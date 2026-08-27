"""重规划器（设计文档 §19 主循环里的失败分类与拆解决策）。

对应主循环伪代码中的：
    ``main_agent.classify_failure(task)``  -> FailureDecision
    ``main_agent.decompose(task)``          -> list[Task]（split 分支）
    ``main_agent.create_prerequisite(task)``-> Task（add_prerequisite 分支）

分类规则直接依据 ``FailureType``（§14）与任务当前 ``attempt``/
``max_attempts`` 计数，是确定性规则而非 LLM 自由发挥——只有在规则
无法覆盖的情况下才回退到 LLM 辅助判断，这样可以保证同一类失败
（比如网络抖动的 TRANSIENT_ERROR）的处理策略是稳定、可预测的，
不会因为 LLM 的随机性导致同一种失败这次重试、下次却直接终止。

LLM 兜底分支（可选注入）：
    ``classify_failure`` 接受一个可选的 ``llm_fallback`` 回调，只有
    当 ``failure_type`` 不在 ``_DEFAULT_DECISION_RULES`` 覆盖范围内
    时才会被调用。``FailureType.UNKNOWN_ERROR`` 特意从规则表中移出
    （不再机械映射为 ``TERMINAL_FAILURE``）——顾名思义，"未知错误"
    正是确定性规则天然无法覆盖、最值得让 LLM 结合 §16 编排上下文
    做进一步判断的场景（例如子智能体日志里其实透露了应该 retry 还是
    ask_user 的线索，只是没有被归到已知分类里）；已被规则表明确覆盖
    的 16 种已知失败类型完全不受影响，调用方不传 ``llm_fallback``
    时会安全降级为 ``TERMINAL_FAILURE``（与改造前效果一致，只是不再
    经由字典默认值实现）。
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from repro_agent.domain.enums import FailureDecision, FailureType
from repro_agent.domain.task import Task, TaskDefinition
from repro_agent.orchestrator.task_factory import build_task_definition

logger = logging.getLogger(__name__)

# 失败类型 -> 默认决策规则（§14 错误类型 + §19 主循环 decision 分支）。
# 这是"确定性优先于 LLM 自由裁量"的一处具体体现：明确属于某一类的
# 失败，不需要每次都问一遍 LLM 该怎么办。
_DEFAULT_DECISION_RULES: dict[FailureType, FailureDecision] = {
    FailureType.TRANSIENT_ERROR: FailureDecision.RETRY,
    FailureType.TOOL_ERROR: FailureDecision.RETRY,
    FailureType.CONTEXT_TOO_LARGE: FailureDecision.SPLIT,
    FailureType.TASK_TOO_BROAD: FailureDecision.SPLIT,
    FailureType.INPUT_MISSING: FailureDecision.ADD_PREREQUISITE,
    FailureType.DEPENDENCY_ERROR: FailureDecision.ADD_PREREQUISITE,
    FailureType.PERMISSION_ERROR: FailureDecision.ASK_USER,
    FailureType.RESOURCE_EXCEEDED: FailureDecision.ASK_USER,
    FailureType.AGENT_STALLED: FailureDecision.RETRY,
    FailureType.PARSING_ERROR: FailureDecision.RETRY,
    FailureType.INVALID_OUTPUT: FailureDecision.RETRY,
    # Experiment environment/code failures have dedicated prerequisite flows
    # in MainAgent.  Other task types must never be "repaired" by launching
    # four copies of the same command.
    FailureType.ENVIRONMENT_ERROR: FailureDecision.ASK_USER,
    FailureType.CODE_ERROR: FailureDecision.RETRY,
    FailureType.DATA_ERROR: FailureDecision.ASK_USER,
    FailureType.MODEL_ERROR: FailureDecision.ASK_USER,
    FailureType.TRAINING_ERROR: FailureDecision.RETRY,
    FailureType.EVALUATION_ERROR: FailureDecision.RETRY,
    # 注意：FailureType.UNKNOWN_ERROR 故意不在此表中——它是唯一被
    # 设计为"交给 LLM 兜底分支处理"的失败类型，见模块顶部说明。
}


class Replanner:
    """依据失败报告分类失败并生成后续任务（拆解/前置任务）。"""

    def classify_failure(
        self,
        task: Task,
        *,
        llm_fallback: Optional[Callable[[Task], FailureDecision]] = None,
    ) -> FailureDecision:
        """§19: ``main_agent.classify_failure(task)``。

        ``llm_fallback`` 为可选注入的兜底回调，仅在失败类型不在
        ``_DEFAULT_DECISION_RULES`` 明确覆盖范围内时才会被调用（见
        模块顶部"LLM 兜底分支"说明）；不传时行为与改造前完全一致。
        """

        if task.failure_report is None:
            return FailureDecision.RETRY

        failure_type = task.failure_report.failure_type

        # PARSING_ERROR 和 INVALID_OUTPUT 是 LLM 输出格式问题（全角字符、
        # 多余/缺失字段等），不是任务逻辑本身有缺陷。每次重试是一次
        # 独立的 LLM 采样，成功的概率与之前失败无关，所以不应该消耗
        # max_attempts 预算。设一个独立上限（max_attempts 的 3 倍）防止
        # 极端情况下无限重试。
        _FREE_RETRY_TYPES = {FailureType.PARSING_ERROR, FailureType.INVALID_OUTPUT}
        if failure_type in _FREE_RETRY_TYPES:
            free_retry_limit = task.definition.max_attempts * 3
            if task.attempt >= free_retry_limit:
                logger.info(
                    "task %s: failure_type=%s reached free-retry limit %d "
                    "(3×max_attempts), escalating to terminal_failure",
                    task.task_id,
                    failure_type.value,
                    free_retry_limit,
                )
                return FailureDecision.TERMINAL_FAILURE
            logger.info(
                "task %s: failure_type=%s does not consume max_attempts "
                "(attempt=%d, max_attempts=%d)",
                task.task_id,
                failure_type.value,
                task.attempt,
                task.definition.max_attempts,
            )
            return FailureDecision.RETRY

        if task.attempt >= task.definition.max_attempts:
            # 达到最大重试次数后，即使规则建议 RETRY 也要升级处理，
            # 避免无限重试同一个必然失败的任务（呼应设计文档 §11.9
            # "防止无限反思和重跑"的预算精神，这里是任务级别的对应）。
            logger.info(
                "task %s reached max_attempts=%d, escalating beyond simple retry",
                task.task_id,
                task.definition.max_attempts,
            )
            return FailureDecision.TERMINAL_FAILURE

        if failure_type in _DEFAULT_DECISION_RULES:
            return _DEFAULT_DECISION_RULES[failure_type]

        if llm_fallback is not None:
            logger.info(
                "task %s: failure_type=%s not covered by deterministic rules, "
                "delegating to LLM-assisted classification",
                task.task_id,
                failure_type.value,
            )
            return llm_fallback(task)

        return FailureDecision.TERMINAL_FAILURE

    def create_code_repair(
        self,
        failed_task: Task,
        *,
        repository_path: str,
    ) -> Task:
        """Create one attempt-bound repair prerequisite for a failed run.

        The repair works from the exact repository snapshot mounted by the
        failed experiment.  It never edits the user's repository in place; the
        coding agent receives another isolated copy and the repaired copy is
        bound to the next experiment attempt only after its regression test
        succeeds.

        修复任务只能引用凭证环境变量名称，不能把框架自身的 API Key、
        API 地址或模型名称复制进任务、数据库、日志、源码或 LLM Prompt。
        """

        failure = failed_task.failure_report
        metadata = dict(failure.metadata or {}) if failure is not None else {}
        command = metadata.get("command") or failed_task.definition.inputs.get(
            "command", []
        )
        tier = metadata.get("tier") or failed_task.definition.inputs.get("tier", "")
        diagnostic = (
            metadata.get("stderr_tail")
            or metadata.get("stdout_tail")
            or (failure.error_message if failure is not None else "")
        )
        diagnostic = str(diagnostic)[-8000:]
        model_configuration_hint = (
            "\n若失败涉及模型连接配置，只能让实验代码在运行时读取"
            " REPRO_AGENT_API_KEY、REPRO_AGENT_API_BASE 和"
            " REPRO_AGENT_MODEL 环境变量；不要把任何凭证或连接值写入"
            "源码、配置模板、修复指令、日志或测试。缺少必填值时应明确"
            "报告需要用户配置，而不是生成占位凭证。\n"
        )
        fix_instructions = (
            "修复实验执行过程中确认的代码错误，并添加能够复现该错误的最小回归测试。\n"
            f"失败层级: {tier}\n"
            f"失败命令: {command}\n"
            f"失败步骤: {failure.failed_step if failure is not None else 'unknown'}\n"
            f"运行诊断:\n{diagnostic}"
            f"{model_configuration_hint}"
        )
        definition = build_task_definition(
            objective=f"修复实验任务 {failed_task.task_id} 的运行时代码错误",
            task_type="coding",
            dependencies=list(failed_task.dependencies),
            inputs={
                "repository_path": repository_path,
                "fix_instructions": fix_instructions,
                "failure_context": diagnostic,
                "failing_command": list(command) if isinstance(command, list) else command,
                "failing_tier": str(tier),
                "source_failed_task_id": failed_task.task_id,
                "source_failed_attempt_id": failed_task.active_attempt_id,
                "execution_image": failed_task.definition.inputs.get(
                    "execution_image", ""
                ),
                "creation_key": (
                    f"code-repair:{failed_task.task_id}:"
                    f"{failed_task.active_attempt_id or failed_task.attempt}"
                ),
            },
            expected_outputs=["output/result.json", "output/candidate_memory.md"],
            completion_criteria=[
                "至少修改一个与失败相关的源码文件",
                "新增最小回归测试且测试通过",
            ],
            parent_task_id=failed_task.task_id,
            max_attempts=min(3, failed_task.definition.max_attempts),
        )
        return Task(job_id=failed_task.job_id, definition=definition)

    def create_environment_repair(
        self,
        failed_task: Task,
        *,
        repository_path: str,
    ) -> Task:
        """Create a real environment prerequisite for one failed experiment."""

        failure = failed_task.failure_report
        metadata = dict(failure.metadata or {}) if failure is not None else {}
        diagnostic = (
            metadata.get("stderr_tail")
            or metadata.get("stdout_tail")
            or (failure.error_message if failure is not None else "")
        )
        definition = build_task_definition(
            objective=f"重建并验证实验任务 {failed_task.task_id} 的运行环境",
            task_type="environment_build",
            dependencies=list(failed_task.dependencies),
            inputs={
                "repository_path": repository_path,
                "dependencies_hint": str(diagnostic)[-8000:],
                "base_image": failed_task.definition.inputs.get(
                    "environment_base_image"
                )
                or failed_task.definition.inputs.get("execution_image", ""),
                "environment_backend": failed_task.definition.inputs.get(
                    "environment_backend", "docker"
                ),
                "python_version": failed_task.definition.inputs.get(
                    "python_version", "3.11"
                ),
                "force_rebuild": True,
                "source_failed_task_id": failed_task.task_id,
                "source_failed_attempt_id": failed_task.active_attempt_id,
                "environment_repair": True,
                "creation_key": (
                    f"environment-repair:{failed_task.task_id}:"
                    f"{failed_task.active_attempt_id or failed_task.attempt}"
                ),
            },
            expected_outputs=["output/result.json", "output/candidate_memory.md"],
            completion_criteria=["运行环境构建成功且 import 自检通过"],
            parent_task_id=failed_task.task_id,
            max_attempts=min(3, failed_task.definition.max_attempts),
        )
        return Task(job_id=failed_task.job_id, definition=definition)

    def decompose(self, task: Task) -> list[Task]:
        """§19: ``main_agent.decompose(task)`` —— TASK_TOO_BROAD / CONTEXT_TOO_LARGE
        场景下把任务拆分为更小粒度的子任务。

        拆分策略是通用的"按 §9 职责边界切分"：如果一个分析任务过大，
        通常意味着它试图一次性覆盖数据+模型+训练+评测多个流程，
        这里按这四个维度做默认拆分，具体项目可以覆盖此方法定制策略。
        """

        if task.definition.task_type not in {"paper_analysis", "code_analysis"}:
            logger.warning(
                "task %s type=%s cannot be behaviorally decomposed",
                task.task_id,
                task.definition.task_type,
            )
            return []

        base_objective = task.objective
        sub_aspects = ["数据流程", "模型流程", "训练流程", "评测流程"]
        subtasks = []
        for aspect in sub_aspects:
            definition = build_task_definition(
                objective=f"{base_objective} —— 聚焦: {aspect}",
                task_type=task.definition.task_type,
                dependencies=list(task.definition.dependencies),
                inputs={
                    **task.definition.inputs,
                    "focus_aspect": aspect,
                    "required_checks": [
                        *list(task.definition.inputs.get("required_checks", []) or []),
                        aspect,
                    ],
                    "audit_hypothesis": (
                        task.definition.inputs.get("audit_hypothesis")
                        or f"将原任务收窄到{aspect}"
                    ),
                    "creation_key": f"subtask:{task.task_id}:{aspect}",
                },
                # 继承原任务已经收窄好的 allowed_tools，而不是重新套用
                # 类型模板全集——否则"任务太大被拆分"反而会让拆出来的
                # 子任务拿到比原任务更宽的工具权限，与"按需最小化授权"
                # 的原则相悖。原任务的 allowed_tools 本身可能已经是
                # "模板 ∩ restrict_tools"的结果，这里直接原样传递即可。
                restrict_tools=list(task.definition.allowed_tools),
                expected_outputs=task.definition.expected_outputs,
                completion_criteria=task.definition.completion_criteria,
                parent_task_id=task.task_id,
            )
            subtasks.append(Task(job_id=task.job_id, definition=definition))
        return subtasks

    def create_prerequisite(self, task: Task) -> Task:
        """§19: ``main_agent.create_prerequisite(task)`` —— INPUT_MISSING /
        DEPENDENCY_ERROR 场景下创建一个前置任务来补齐缺失的输入。
        """

        missing_hint = ""
        if task.failure_report:
            missing_hint = task.failure_report.error_message

        # 没有携带 dataset/model/checkpoint 路径，check_path_resource
        # 不会被 ResourceCheckAgent 触发，只保留固定会调用的探测工具。
        definition = build_task_definition(
            objective=f"为任务 {task.task_id} 补齐缺失的前置条件: {missing_hint}",
            task_type="resource_check",
            inputs={
                "original_task_id": task.task_id,
                "missing_hint": missing_hint,
                "creation_key": f"prerequisite:{task.task_id}:{task.attempt}",
            },
            restrict_tools=["check_gpu", "check_cuda", "check_disk_space"],
        )
        return Task(job_id=task.job_id, definition=definition)
