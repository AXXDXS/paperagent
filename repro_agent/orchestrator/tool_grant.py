"""主智能体的工具分配与补授裁决（工具分配权上收后的核心决策模块）。

本模块把"给任务配哪些工具"从两层拆成一层决策 + 一层硬边界：

    规划期（派发前）：``ToolAllocationPlanner``
        主智能体根据任务目标、输入、任务类型模板（仅作参考）和工具
        白名单/风险值，为每个任务实例定制 ``allowed_tools``。任务类型
        对应的标准模板（``task_factory.STANDARD_TOOL_TEMPLATES``）从
        "唯一决定者"降级为"参考建议"——主智能体可以增删，但最终结果
        仍要经过 ``ToolAuthorizer.authorize`` 的注册表/风险预算/
        forbidden_actions 三重硬校验，自由裁量永远在 fail-closed 的
        合法包络内。LLM 不可用或输出不合法时回退到模板（旧行为）。

    运行期（执行中）：``ToolGrantDecisionMaker``
        子智能体调用了一个"已注册但未分配给自己"的工具时，升级请求
        到这里裁决：GRANT（补授，子智能体原地继续）/ DENY（明确拒绝）
        / ASK_USER（拿不定主意，转人工介入）。确定性规则先行（存在
        性、风险预算、forbidden_actions 直接 DENY），规则放行后才交
        给 LLM 结合任务上下文判断"任务是否真的需要它"；LLM 失败安全
        降级为 ASK_USER——既不能因为一次网络抖动就放权（GRANT），
        也不能据此杀掉任务（DENY）。

设计原则（与 ``orchestrator/replanner.py`` 一致）：
    确定性规则优先于 LLM 自由裁量。同一 (task, tool) 的裁决会被缓存，
    一个任务对同一工具只裁决一次，避免重试循环中反复弹人工请求。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from repro_agent.domain.enums import ToolGrantDecision
from repro_agent.llm_output import StructuredOutputError, parse_structured_json
from repro_agent.providers.base import LLMMessage, LLMRequestParams
from repro_agent.providers.prompt_cache import (
    build_stable_system_prompt,
    prompt_cache_key_for_tools,
)
from repro_agent.providers.retry import call_with_retry
from repro_agent.tools.authorization import ToolAuthorizer
from repro_agent.tools.base import ToolSpec

logger = logging.getLogger(__name__)


TOOL_GRANT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "reason"],
    "properties": {
        "decision": {"type": "string", "enum": ["grant", "deny", "ask_user"]},
        "reason": {"type": "string"},
    },
}

TOOL_ALLOCATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["allowed_tools", "reason"],
    "properties": {
        "allowed_tools": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "reason": {"type": "string"},
    },
}

_GRANT_SYSTEM_PROMPT = (
    "你是论文复现系统的主智能体，负责裁决子智能体在运行期提出的"
    "\"补授缺失工具\"请求。候选工具已经通过确定性安全边界校验"
    "（已注册、风险在该任务类型预算内、未被 forbidden_actions 禁止），"
    "你只需回答一个问题：这个任务是否真的需要这个工具才能完成。\n"
    "你只能输出严格的 JSON，不要输出任何其他文字：\n"
    '{"decision": "grant|deny|ask_user", "reason": "一句话理由"}\n'
    "裁决含义：\n"
    "- grant: 任务目标确实需要该工具，现有工具无法替代，应立即补授\n"
    "- deny: 该工具与任务目标无关，或任务可以用现有工具完成，"
    "或补授会引入不必要的风险\n"
    "- ask_user: 涉及安全/成本/权限边界你拿不准，需要人类最终仲裁\n"
    "约束：宁缺毋滥，能不补就不补；拿不准时选 ask_user。"
)

_ALLOCATION_SYSTEM_PROMPT = (
    "你是论文复现系统的主智能体，正在为一个即将派发的任务定制下发给"
    "子智能体的工具白名单。任务类型的标准模板只是参考，你应该根据任务"
    "的实际目标和输入判断它真正需要的最小工具子集。\n"
    "只输出严格的 JSON，不要输出任何其他文字：\n"
    '{"allowed_tools": ["tool_a", "tool_b"], "reason": "一句话理由"}\n'
    "约束：\n"
    "- 只能从候选工具列表中选择，任何列表之外的名称都会被丢弃；\n"
    "- 必须包含 write_task_output；\n"
    "- 最小授权原则：宁可少给——子智能体运行期缺少工具时，"
    "可以向主智能体申请补授，不会因此卡死；\n"
    "- 仔细对照任务目标与每个工具的\"何时使用\"说明再决定。"
)

# 工具目录里每条描述的最大长度，控制 prompt 体量。
_CATALOG_ENTRY_MAX_CHARS = 320
_TASK_CONTEXT_MAX_CHARS = 1_500


@dataclass(frozen=True)
class ToolGrantOutcome:
    """一次补授裁决的结果。"""

    decision: ToolGrantDecision
    tool_name: str
    reason: str
    source: str = "rule"  # rule | llm | cache | fallback


def _bounded(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _describe_spec(spec: ToolSpec) -> str:
    """单条工具目录：名称 + 风险 + 一句话描述 + 何时使用。"""

    parts = [f"- {spec.name} [risk={spec.risk_level.value}]"]
    if spec.description:
        parts.append(f"说明: {_bounded(spec.description, 160)}")
    when_to_use = getattr(spec, "when_to_use", "")
    if when_to_use:
        parts.append(f"何时使用: {_bounded(when_to_use, 160)}")
    return " ".join(parts)


def eligible_specs(
    authorizer: ToolAuthorizer,
    *,
    task_type: str,
    forbidden_actions: list[str] | None = None,
) -> list[ToolSpec]:
    """返回该任务类型在硬边界内**可以**被分配的全部已注册工具。

    与 ``ToolAuthorizer.authorize`` 使用同一套校验（注册表 + 风险预算 +
    forbidden_actions），保证规划期 LLM 看到的候选集与运行期授权层
    实际放行的集合一致——LLM 选出的任何名字都必然能通过 authorize。
    """

    eligible = []
    for spec in authorizer.registry.all_specs():
        denials = authorizer.validate_human_approval(
            task_type=task_type,
            tool_names=[spec.name],
            forbidden_actions=forbidden_actions,
        )
        if not denials:
            eligible.append(spec)
    return eligible


class ToolGrantDecisionMaker:
    """运行期缺工具升级请求的裁决器：确定性规则先行 + LLM 辅助判断。"""

    def __init__(
        self,
        tool_authorizer: ToolAuthorizer,
        llm_provider: Any = None,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 8192,
        max_llm_retries: int = 2,
    ):
        self._authorizer = tool_authorizer
        self._llm = llm_provider
        self._model = model
        self._max_tokens = max_tokens
        self._max_llm_retries = max_llm_retries
        # (task_id, tool_name) -> ToolGrantOutcome。一个任务对同一工具
        # 只裁决一次：重试/挂起恢复后再次升级同一工具时直接复用结论，
        # 防止"deny 后重试又 ask_user"的振荡和重复弹人工请求。
        self._cache: dict[tuple[str, str], ToolGrantOutcome] = {}

    def adjudicate(
        self,
        *,
        task_id: str,
        task_type: str,
        objective: str,
        inputs: dict[str, Any] | None,
        allowed_tools: list[str],
        forbidden_actions: list[str] | None,
        tool_name: str,
        rationale: str = "",
    ) -> ToolGrantOutcome:
        """对"给这个任务补授 tool_name"做出 GRANT / DENY / ASK_USER 裁决。"""

        cached = self._cache.get((task_id, tool_name))
        if cached is not None:
            return ToolGrantOutcome(
                decision=cached.decision,
                tool_name=tool_name,
                reason=cached.reason,
                source="cache",
            )

        # ---- 规则层 1：工具必须已注册 ----
        spec = self._authorizer.registry.get(tool_name)
        if spec is None:
            return self._remember(
                task_id,
                ToolGrantOutcome(
                    decision=ToolGrantDecision.DENY,
                    tool_name=tool_name,
                    reason=f"tool '{tool_name}' is not registered in the global registry",
                    source="rule",
                ),
            )

        # ---- 规则层 2：硬安全边界（风险预算 / forbidden_actions / 网络）----
        # 主智能体的裁量与人工批准共享同一条不可逾越的边界：
        # validate_human_approval。超出边界的请求连问 LLM 的资格都没有。
        denials = self._authorizer.validate_human_approval(
            task_type=task_type,
            tool_names=[tool_name],
            forbidden_actions=forbidden_actions,
        )
        if denials:
            return self._remember(
                task_id,
                ToolGrantOutcome(
                    decision=ToolGrantDecision.DENY,
                    tool_name=tool_name,
                    reason=denials[0].reason,
                    source="rule",
                ),
            )

        # ---- LLM 层：任务是否真的需要这个工具 ----
        if self._llm is None:
            # 没有分析能力时不能放权也不能杀任务，转人工是最保守的选择。
            return self._remember(
                task_id,
                ToolGrantOutcome(
                    decision=ToolGrantDecision.ASK_USER,
                    tool_name=tool_name,
                    reason="主智能体未配置 LLM，无法分析任务是否需要该工具",
                    source="fallback",
                ),
            )

        user_prompt = self._build_grant_prompt(
            task_type=task_type,
            objective=objective,
            inputs=inputs,
            allowed_tools=allowed_tools,
            spec=spec,
            rationale=rationale,
        )
        messages = [
            LLMMessage(role="system", content=build_stable_system_prompt(_GRANT_SYSTEM_PROMPT)),
            LLMMessage(role="user", content=user_prompt),
        ]
        params = LLMRequestParams(
            model=self._model,
            temperature=0.0,
            max_tokens=self._max_tokens,
            prompt_cache_key=prompt_cache_key_for_tools([]),
            response_schema=TOOL_GRANT_SCHEMA,
            response_schema_name="tool_grant_decision",
        )
        try:
            response = call_with_retry(
                self._llm.complete,
                messages,
                params,
                max_retries=self._max_llm_retries,
            )
            data = parse_structured_json(
                response.content, TOOL_GRANT_SCHEMA, label="tool grant decision output"
            )
            decision = ToolGrantDecision(str(data["decision"]).strip().lower())
        except Exception as exc:  # noqa: BLE001 - 任何 provider 异常都必须安全降级，不能阻断升级流程
            logger.warning(
                "task %s: LLM-assisted tool grant adjudication failed (%s); "
                "falling back to ask_user for tool '%s'",
                task_id,
                exc,
                tool_name,
            )
            return self._remember(
                task_id,
                ToolGrantOutcome(
                    decision=ToolGrantDecision.ASK_USER,
                    tool_name=tool_name,
                    reason=f"LLM 裁决失败，安全降级转人工: {exc}",
                    source="fallback",
                ),
            )

        reason = _bounded(str(data.get("reason", "")), 500)
        return self._remember(
            task_id,
            ToolGrantOutcome(
                decision=decision,
                tool_name=tool_name,
                reason=reason or "LLM 裁决",
                source="llm",
            ),
        )

    def _remember(self, task_id: str, outcome: ToolGrantOutcome) -> ToolGrantOutcome:
        self._cache[(task_id, outcome.tool_name)] = outcome
        return outcome

    def remember_for_task(self, task_id: str, outcome: ToolGrantOutcome) -> None:
        """显式登记一条裁决缓存（供任务维度的幂等复用）。"""

        self._cache[(task_id, outcome.tool_name)] = outcome

    @staticmethod
    def _build_grant_prompt(
        *,
        task_type: str,
        objective: str,
        inputs: dict[str, Any] | None,
        allowed_tools: list[str],
        spec: ToolSpec,
        rationale: str,
    ) -> str:
        inputs_text = _bounded(
            json.dumps(inputs or {}, ensure_ascii=False, default=str),
            _TASK_CONTEXT_MAX_CHARS,
        )
        granted = ", ".join(sorted(set(allowed_tools))) or "(无)"
        return (
            f"任务类型: {task_type}\n"
            f"任务目标: {_bounded(objective, 600)}\n"
            f"任务输入(摘要): {inputs_text}\n"
            f"当前已授权工具: {granted}\n\n"
            f"子智能体请求补授的工具:\n{_describe_spec(spec)}\n"
            f"请求现场: {rationale or '子智能体在执行任务时尝试调用该工具但未被授权'}\n\n"
            "请裁决是否补授该工具。"
        )


class ToolAllocationPlanner:
    """规划期工具白名单定制器：模板作参考，主智能体按任务内容裁剪。"""

    def __init__(
        self,
        tool_authorizer: ToolAuthorizer,
        llm_provider: Any = None,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 8192,
        max_llm_retries: int = 2,
    ):
        self._authorizer = tool_authorizer
        self._llm = llm_provider
        self._model = model
        self._max_tokens = max_tokens
        self._max_llm_retries = max_llm_retries

    def plan_allowed_tools(
        self,
        *,
        task_type: str,
        objective: str,
        inputs: dict[str, Any] | None,
        template_tools: list[str],
        forbidden_actions: list[str] | None = None,
    ) -> tuple[list[str], str]:
        """为任务定制 ``allowed_tools``，返回 (工具列表, 决策来源说明)。

        失败安全语义：LLM 缺失、调用失败、输出为空或全部是幻觉名称时，
        一律回退到任务类型模板（过滤掉模板里不合法的名字），即改造前
        的旧行为——规划期的定制是"锦上添花"，绝不能让它把任务变得
        比模板基线更不可用。
        """

        eligible = eligible_specs(
            self._authorizer,
            task_type=task_type,
            forbidden_actions=forbidden_actions,
        )
        eligible_names = {spec.name for spec in eligible}
        # 模板基线：模板 ∩ 合法集合。模板里若混入了超风险预算的名字
        # （本不应发生，授权层也会拦），这里就地修正。
        template_baseline = [
            name for name in template_tools if name in eligible_names
        ]
        if "write_task_output" in eligible_names and "write_task_output" not in template_baseline:
            template_baseline.append("write_task_output")

        if self._llm is None:
            return template_baseline, "template-fallback:no-llm"

        user_prompt = self._build_allocation_prompt(
            task_type=task_type,
            objective=objective,
            inputs=inputs,
            template_tools=template_tools,
            eligible=eligible,
        )
        messages = [
            LLMMessage(
                role="system",
                content=build_stable_system_prompt(_ALLOCATION_SYSTEM_PROMPT),
            ),
            LLMMessage(role="user", content=user_prompt),
        ]
        params = LLMRequestParams(
            model=self._model,
            temperature=0.0,
            max_tokens=self._max_tokens,
            prompt_cache_key=prompt_cache_key_for_tools([]),
            response_schema=TOOL_ALLOCATION_SCHEMA,
            response_schema_name="tool_allocation",
        )
        try:
            response = call_with_retry(
                self._llm.complete,
                messages,
                params,
                max_retries=self._max_llm_retries,
            )
            data = parse_structured_json(
                response.content, TOOL_ALLOCATION_SCHEMA, label="tool allocation output"
            )
            requested = [str(name) for name in data.get("allowed_tools", [])]
        except Exception as exc:  # noqa: BLE001 - 任何 provider 异常都回退模板，不阻塞派发
            logger.warning(
                "tool allocation customization failed (%s); falling back to template",
                exc,
            )
            return template_baseline, f"template-fallback:{type(exc).__name__}"

        # 过滤幻觉：只保留候选集合内的名字，强制并上最小能力。
        selected = [name for name in requested if name in eligible_names]
        if "write_task_output" in eligible_names:
            selected.append("write_task_output")
        selected = list(dict.fromkeys(selected))
        if len(selected) <= 1:
            # LLM 实际上一个合法工具都没选上（全是幻觉/越权名）：输出
            # 质量不可信，回退模板而不是让任务裸奔在最小集上。
            logger.warning(
                "tool allocation LLM output contained no eligible tool names; "
                "falling back to template"
            )
            return template_baseline, "template-fallback:empty-selection"
        return selected, "llm-customized"

    @staticmethod
    def _build_allocation_prompt(
        *,
        task_type: str,
        objective: str,
        inputs: dict[str, Any] | None,
        template_tools: list[str],
        eligible: list[ToolSpec],
    ) -> str:
        inputs_text = _bounded(
            json.dumps(inputs or {}, ensure_ascii=False, default=str),
            _TASK_CONTEXT_MAX_CHARS,
        )
        catalog = "\n".join(_describe_spec(spec) for spec in eligible)
        template = ", ".join(template_tools) or "(无)"
        return (
            f"任务类型: {task_type}\n"
            f"任务目标: {_bounded(objective, 600)}\n"
            f"任务输入(摘要): {inputs_text}\n"
            f"该任务类型的参考模板（仅作参考，可增删）: {template}\n\n"
            f"候选工具目录（只能从中选择）:\n{catalog}\n\n"
            "请为该任务选出实际需要的最小工具子集。"
        )


# 失败报告错误消息 -> 请求补授的工具名列表。
# 覆盖两类消息格式：
#   1. ToolAuthorization.call:      ... not authorized to call tool 'xxx'
#   2. ToolAuthorization.describe_granted: requested tool description(s)
#      not granted: aaa, bbb (granted=...)
_QUOTED_TOOL_PATTERN = re.compile(r"tool ['\"]([A-Za-z0-9_.-]+)['\"]")
_NOT_GRANTED_PATTERN = re.compile(r"not granted:\s*([A-Za-z0-9_.,\s-]+?)\s*\(")


def extract_requested_tool_names(message: str) -> list[str]:
    """从权限错误消息中提取子智能体想要但没拿到的工具名。"""

    text = str(message or "")
    found: list[str] = []
    for match in _QUOTED_TOOL_PATTERN.finditer(text):
        found.append(match.group(1))
    for match in _NOT_GRANTED_PATTERN.finditer(text):
        for part in match.group(1).split(","):
            name = part.strip()
            if name:
                found.append(name)
    return list(dict.fromkeys(found))
