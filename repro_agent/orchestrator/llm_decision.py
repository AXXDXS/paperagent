"""主智能体的 LLM 辅助决策器（设计文档 §16 上下文编排 + §19 决策点的落地桥梁）。

背景与动机：
    ``context/builder.py`` 的 ``ContextBuilder`` 与 ``context/budget.py``
    的压缩/优先级机制此前只在 ``MainAgent.__init__`` 里被构造成实例，
    从未被 ``step()`` 主循环里的任何决策点真正消费——§19 伪代码中
    ``main_agent.classify_failure(task)`` 等"决策方法"此前全部由
    ``Replanner``/``ReflectionController`` 里的确定性规则代码实现，
    不涉及 LLM 调用。

    本模块补上"规则无法覆盖时，构造九段上下文 → 交给 LLM 做结构化
    判断 → 解析为领域枚举"这条链路，让 ``ContextBuilder`` 真正为主
    智能体的决策服务，同时不改变既有设计原则：``orchestrator/replanner.py``
    文档明确写着"确定性规则优先于 LLM 自由裁量，只有规则无法覆盖的
    情况才回退到 LLM 辅助判断"——本模块就是那个"回退"分支的具体实现，
    默认规则路径完全不受影响。

设计取舍：
    - LLM 只被要求返回一个受限的结构化 JSON（``{"decision": "...",
      "reason": "..."}``），不是自由文本，解析失败或返回值不在合法
      枚举范围内时一律安全降级为 ``TERMINAL_FAILURE``（升级给用户/
      终止，而不是悄悄重试或拆解——决策不确定时选择最保守的分支）。
    - 复用 ``agents/base.py`` 里子智能体已经在用的
      ``call_with_retry``（token 递减 + 指数退避），主智能体侧的 LLM
      调用健壮性策略与子智能体保持一致，不重新发明一套。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from repro_agent.context.builder import ContextBuilder, UnresolvedIssue
from repro_agent.domain.enums import FailureDecision
from repro_agent.llm_output import DECISION_SCHEMA, StructuredOutputError, parse_structured_json
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.dag import TaskDAG
from repro_agent.domain.task import Task
from repro_agent.providers.base import LLMMessage, LLMProviderError, LLMRequestParams
from repro_agent.providers.prompt_cache import (
    build_stable_system_prompt,
    prompt_cache_key_for_tools,
)
from repro_agent.providers.retry import call_with_retry

logger = logging.getLogger(__name__)

_VALID_DECISIONS = {d.value for d in FailureDecision}

_SYSTEM_PROMPT = (
    "你是论文复现系统的主智能体，负责在确定性规则无法覆盖的失败场景下"
    "做出处置决策。你必须只从以下五种决策中选择一种，并只输出严格的 JSON，"
    "不要输出任何其他文字：\n"
    '{"decision": "retry|split|add_prerequisite|ask_user|terminal_failure", '
    '"reason": "一句话理由"}\n'
    "决策含义：\n"
    "- retry: 值得原样重试（怀疑是偶发/环境抖动）\n"
    "- split: 任务范围过大或上下文过大，应拆解为更小的子任务\n"
    "- add_prerequisite: 缺少某个前置产出，应先创建前置任务补齐\n"
    "- ask_user: 需要人工介入才能继续（权限/资源/数据问题）\n"
    "- terminal_failure: 无法通过以上方式恢复，应终止该任务\n"
    "红线约束：不允许通过修改系统主流程代码来解决失败，只能在任务"
    "调度层面做决策；如果不确定，选择最保守的 terminal_failure。"
)


@dataclass
class LLMDecisionResult:
    decision: FailureDecision
    reason: str
    raw_content: str
    fallback_used: bool = False


class MainAgentLLMDecisionMaker:
    """把 §16 九段上下文编排接入 §19 决策点的通用桥梁。

    只负责"构造上下文 -> 调用 LLM -> 解析为领域决策"，不感知具体是
    哪个决策点在调用它——``classify_failure_with_llm`` 是当前唯一的
    使用方，未来其它决策点（比如是否触发反思）可以复用同一个实例。
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        llm_provider: Any,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 8000,
        max_llm_retries: int = 2,
    ):
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.model = model
        self.max_tokens = max_tokens
        self.max_llm_retries = max_llm_retries

    def classify_failure_with_llm(
        self,
        *,
        job: ReproductionJob,
        dag: TaskDAG,
        task: Task,
        recent_events: list[dict[str, Any]],
    ) -> LLMDecisionResult:
        """规则表未覆盖该失败类型时的兜底分支：编排上下文后询问 LLM。

        呼应 ``Replanner.classify_failure`` 文档："只有在规则无法覆盖
        的情况下才回退到 LLM 辅助判断"——调用方需要自行判断"规则是否
        覆盖"，本方法不重复该判断，只负责"既然要问 LLM，就把上下文
        编排规范做完整"这一件事。
        """

        failure_report = task.failure_report
        decision_desc = (
            f"任务 {task.task_id}（类型={task.definition.task_type}）执行失败，"
            f"当前重试次数 {task.attempt}/{task.definition.max_attempts}。"
            + (
                f"\n失败类型: {failure_report.failure_type.value}\n"
                f"失败步骤: {failure_report.failed_step}\n"
                f"错误信息: {failure_report.error_message}\n"
                f"可能原因: {failure_report.likely_causes}\n"
                f"建议动作: {failure_report.recommended_action}"
                if failure_report is not None
                else "\n(未提供失败报告)"
            )
            + "\n\n请判断应如何处置这个失败任务：retry / split / "
            "add_prerequisite / ask_user / terminal_failure。"
        )

        unresolved = [
            UnresolvedIssue(
                kind="task_failure",
                description=decision_desc,
                related_task_id=task.task_id,
            )
        ]

        ctx = self.context_builder.build(
            job=job,
            dag=dag,
            current_decision=decision_desc,
            recent_events=recent_events,
            unresolved_issues=unresolved,
            max_tokens=self.max_tokens,
        )

        messages = [
            LLMMessage(role="system", content=build_stable_system_prompt(_SYSTEM_PROMPT)),
            LLMMessage(role="user", content=ctx.text),
        ]
        params = LLMRequestParams(
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            prompt_cache_key=prompt_cache_key_for_tools([]),
            response_schema=DECISION_SCHEMA,
            response_schema_name="failure_decision",
        )

        try:
            response = call_with_retry(
                self.llm_provider.complete,
                messages,
                params,
                max_retries=self.max_llm_retries,
            )
        except LLMProviderError as exc:
            logger.error(
                "task %s: LLM-assisted failure classification errored out (%s); "
                "falling back to terminal_failure",
                task.task_id,
                exc,
            )
            return LLMDecisionResult(
                decision=FailureDecision.TERMINAL_FAILURE,
                reason=f"LLM 调用失败，安全降级: {exc}",
                raw_content="",
                fallback_used=True,
            )

        return self._parse_response(response.content, task_id=task.task_id)

    @staticmethod
    def _parse_response(content: str, *, task_id: str) -> LLMDecisionResult:
        try:
            data = parse_structured_json(
                content, DECISION_SCHEMA, label="failure decision output"
            )
            decision_value = data["decision"].strip().lower()
            reason = data["reason"]
        except (StructuredOutputError, AttributeError, TypeError) as exc:
            logger.warning(
                "task %s: failed to parse LLM decision response as JSON (%s); "
                "raw content=%r; falling back to terminal_failure",
                task_id,
                exc,
                content,
            )
            return LLMDecisionResult(
                decision=FailureDecision.TERMINAL_FAILURE,
                reason="LLM 输出无法解析为合法 JSON，安全降级",
                raw_content=content,
                fallback_used=True,
            )

        if decision_value not in _VALID_DECISIONS:
            logger.warning(
                "task %s: LLM returned decision=%r not in valid set %s; "
                "falling back to terminal_failure",
                task_id,
                decision_value,
                _VALID_DECISIONS,
            )
            return LLMDecisionResult(
                decision=FailureDecision.TERMINAL_FAILURE,
                reason=f"LLM 返回了非法决策值 '{decision_value}'，安全降级",
                raw_content=content,
                fallback_used=True,
            )

        return LLMDecisionResult(
            decision=FailureDecision(decision_value),
            reason=reason,
            raw_content=content,
            fallback_used=False,
        )
