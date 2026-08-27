"""反思闭环控制器（设计文档 §11.2、§19）。

对应主循环伪代码的三个钩子：
    ``result_gap_detected()``      -> 触发反思
    ``main_agent.plan_audit(...)`` -> 生成审计任务 DAG
    ``summarize_audit_findings()`` -> 汇总多个并行审计任务的结论
    ``audit_issue_confirmed()``    -> 触发修复
    ``main_agent.plan_repair()``   -> 生成修复任务
    ``repair_completed()``         -> 触发最小范围重跑
    ``main_agent.plan_minimum_rerun_scope()`` -> 按 RerunScope 生成重跑任务

反思闭环整体流程（§11.2 原文）：
    复现结果超出容差 → 结果验证智能体确认指标计算 → 主智能体创建反思
    任务 → 反思智能体分析差距 → 生成审计检查清单 → 主智能体构建审计
    任务 DAG → 多个子智能体并行检查 → 汇总审计结果
        ├── 找到错误 → 修复错误 → 最小测试 → 缩小实验 → 重新运行正式实验
        └── 未找到错误 → 确认流程和配置无明显问题 → 向用户报告真实复现差距

关键约束（呼应 §11.8 原文、也是本次改造明确要求的行为）：
    "未找到错误"分支（``summarize_audit_findings`` 判定为
    ``NO_OBVIOUS_ERROR_FOUND``/``RANDOMNESS_LIKELY``/
    ``UNDISCLOSED_DETAIL_LIKELY`` 之一）时，即使复现结果与论文目标
    仍有差距，也**不允许**仅仅为了"对齐论文数字"而盲目重跑——审计已经
    确认流程、配置、数据、代码均无明显问题，重跑大概率只是重复消耗
    GPU/模型调用预算而不会改变结果。这种情况下 ``audit_issue_confirmed``
    返回 ``False``，主循环应直接把 Job 推进到终态
    ``VERIFIED_REPRODUCTION_GAP`` 并生成用户报告，而不是进入
    ``plan_repair``/``plan_minimum_rerun_scope`` 分支（这两个方法只应该
    在"确认存在具体错误"时才被调用）。

预算防护（§11.9，复用 ``domain.job.ReproductionJob.budget_exhausted``）：
    本控制器在触发反思/重跑前都会先检查 Job 预算是否耗尽，耗尽则
    直接跳过并返回空任务列表，交由主循环把 Job 状态推进到
    ``VERIFIED_REPRODUCTION_GAP``（"经审计确认存在可靠复现差距"，
    对应 §20 最终结论选项之一），避免无限反思循环。
"""

from __future__ import annotations

import logging
import json
import shlex
from dataclasses import dataclass, field

from repro_agent.domain.enums import AuditResultType, ExperimentTier, RerunScope
from repro_agent.domain.experiment import ExperimentRun, MetricComparison
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.reflection import AuditFinding, ReflectionHypothesis, ReflectionReport
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.evaluation.tier_gate import TierGate

logger = logging.getLogger(__name__)

_RERUN_SCOPE_COMMAND_HINT = {
    RerunScope.EVALUATION_ONLY: "仅重跑评测程序",
    RerunScope.INFERENCE_AND_EVALUATION: "重跑推理与评测",
    RerunScope.FULL_TRAINING: "重跑完整训练+推理+评测",
    RerunScope.ENVIRONMENT_REBUILD: "先重建环境，再重跑完整流程",
}

_ISSUE_CONFIRMED_RESULTS = {
    AuditResultType.PROCESS_ERROR_CONFIRMED,
    AuditResultType.CONFIG_ERROR_CONFIRMED,
    AuditResultType.CODE_ERROR_CONFIRMED,
    AuditResultType.DATA_ERROR_CONFIRMED,
    AuditResultType.EVALUATION_ERROR_CONFIRMED,
    AuditResultType.ENVIRONMENT_ERROR_CONFIRMED,
}


@dataclass
class ReflectionDecision:
    should_reflect: bool
    reason: str = ""


class ReflectionController:
    """主智能体侧的反思闭环编排逻辑。"""

    def result_gap_detected(
        self, comparisons: list[MetricComparison], job: ReproductionJob
    ) -> ReflectionDecision:
        """§11.1 触发条件：关键指标超出容差即视为 gap（其余触发条件——
        方法与 baseline 排序不一致/消融趋势不一致/结果波动异常等——
        需要更专门的分析器，这里先覆盖最核心、最容易机器判定的一条，
        更复杂的触发条件留给结果验证子智能体在 LLM 分析阶段识别并
        通过任务输出显式报告。
        """

        exhausted, reason = job.budget_exhausted()
        if exhausted:
            return ReflectionDecision(should_reflect=False, reason=f"预算已耗尽: {reason}")

        out_of_tolerance = [c for c in comparisons if not c.within_tolerance]
        if not out_of_tolerance:
            return ReflectionDecision(should_reflect=False, reason="所有指标均在容差范围内")

        return ReflectionDecision(
            should_reflect=True,
            reason=f"{len(out_of_tolerance)} 个指标超出容差: "
            f"{[c.metric for c in out_of_tolerance]}",
        )

    def plan_audit(self, reflection_report: ReflectionReport) -> list[Task]:
        """§19: ``main_agent.plan_audit(reflection_report)``。

        依据反思智能体给出的 ``suggested_audit_tasks``（见
        ``agents/reflection/agent.py``）逐条创建审计任务，每个审计
        任务分配到对应的子智能体类型（论文理解问题 -> paper_analysis，
        代码路径问题 -> code_analysis，参数问题 -> specification，
        数据/模型问题 -> resource_check），按优先级排序。
        """

        dimension_to_task_type = {
            "A": "paper_analysis",
            "B": "code_analysis",
            "C": "specification",
            "D": "resource_check",
            "E": "resource_check",
        }
        context = dict(reflection_report.audit_context or {})
        upstream = dict(context.get("upstream_task_ids", {}) or {})
        audit_tasks: list[Task] = []
        for hypothesis in reflection_report.sorted_hypotheses():
            task_type = dimension_to_task_type.get(hypothesis.category, "code_analysis")
            inputs = {
                "audit_hypothesis_id": hypothesis.hypothesis_id,
                "audit_check_dimension": hypothesis.category,
                "audit_hypothesis": hypothesis.description,
                "required_checks": hypothesis.required_checks,
                "reflection_id": reflection_report.reflection_id,
                "source_run_id": reflection_report.run_id,
                "target_experiments": list(context.get("target_experiments", [])),
                "creation_key": (
                    f"audit:{reflection_report.reflection_id}:"
                    f"{hypothesis.hypothesis_id}"
                ),
            }
            dependencies: list[str] = []
            restrict_tools: list[str] | None = None
            if hypothesis.category == "A":
                inputs.update(
                    {
                        "paper_path": context.get("paper_path", ""),
                        "appendix_paths": list(context.get("appendix_paths", [])),
                        "scope": "body",
                    }
                )
                restrict_tools = ["read_file", "read_pdf_text", "inspect_pdf_page"]
            elif hypothesis.category == "B":
                inputs["repository_path"] = context.get("repository_path", "")
                inputs["focus_aspect"] = hypothesis.description
                restrict_tools = [
                    "get_repository_map",
                    "search_repository_code",
                    "read_file",
                    "hash_path",
                ]
            elif hypothesis.category == "C":
                dependencies = [
                    *list(upstream.get("paper_analysis", [])),
                    *list(upstream.get("code_analysis", [])),
                ]
                inputs.update(
                    {
                        "experiment_id": context.get("experiment_id", "experiment"),
                        "target_claim": "audit_experiment_configuration",
                        "user_overrides": dict(context.get("user_overrides", {}) or {}),
                    }
                )
                restrict_tools = []
            else:
                dependencies = list(upstream.get("specification", []))[-1:]
                inputs.update(
                    {
                        "repository_path": context.get("repository_path", ""),
                        "dataset_paths": list(context.get("dataset_paths", [])),
                        "model_paths": list(context.get("model_paths", [])),
                        "checkpoint_paths": list(context.get("checkpoint_paths", [])),
                        "requested_gpu_count": context.get("requested_gpu_count"),
                        "requested_gpu_memory_gb": context.get(
                            "requested_gpu_memory_gb"
                        ),
                        "requested_disk_mb": context.get("requested_disk_mb"),
                    }
                )
                restrict_tools = [
                    "find_named_resource",
                    "check_gpu",
                    "check_cuda",
                    "check_disk_space",
                ]
                if any(
                    inputs.get(key)
                    for key in ("dataset_paths", "model_paths", "checkpoint_paths")
                ):
                    restrict_tools.append("check_path_resource")
            definition = build_task_definition(
                objective=f"[审计] {hypothesis.description}",
                task_type=task_type,
                dependencies=list(dict.fromkeys(dependencies)),
                inputs=inputs,
                restrict_tools=restrict_tools,
                priority=hypothesis.priority,
            )
            audit_tasks.append(Task(job_id=reflection_report.job_id, definition=definition))
        return audit_tasks

    def summarize_audit_findings(
        self, findings: list[AuditFinding]
    ) -> tuple[AuditResultType, str]:
        """§11.6/§11.2"汇总审计结果"：把多个并行审计任务各自的结论
        汇总成一个整体 ``AuditResultType`` + 人类可读的确认问题描述。

        汇总规则（保持确定性，不引入额外的 LLM 判断——每个具体审计
        任务的结论已经是各自子智能体给出的判断，这里只做"多个结论中
        选择最值得处理的一个"这一层级的确定性合并）：
            1. 只要有任意一条 finding 命中"已确认错误"类型
               （``_ISSUE_CONFIRMED_RESULTS``），就以证据数量最多的
               那一类错误作为整体结论——优先修复"证据最充分"的问题，
               而不是"最先发现"的问题；
            2. 都没有确认错误，但存在 ``RESOURCE_LIMITATION_CONFIRMED``，
               归类为资源限制（不进入 repair 分支，因为这类问题往往
               不是"代码/配置错误"，而是需要用户介入调整资源预算）；
            3. 都没有，但存在 ``UNDISCLOSED_DETAIL_LIKELY``，说明多个
               维度都指向"论文未披露的实现细节"；
            4. 都没有，但存在 ``RANDOMNESS_LIKELY``；
            5. 以上都不满足（包括 findings 为空的情况），判定为
               ``NO_OBVIOUS_ERROR_FOUND``——§11.8 "未找到错误"分支，
               不允许仅为了对齐论文数字而重跑。
        """

        if not findings:
            return AuditResultType.NO_OBVIOUS_ERROR_FOUND, "未产生任何审计结论，视为无明显问题"

        confirmed = [f for f in findings if f.result in _ISSUE_CONFIRMED_RESULTS]
        if confirmed:
            counts: dict[AuditResultType, int] = {}
            for f in confirmed:
                counts[f.result] = counts.get(f.result, 0) + 1
            winner = max(counts.items(), key=lambda kv: kv[1])[0]
            details = [f.detail for f in confirmed if f.result == winner]
            return winner, "; ".join(details)

        for fallback_type in (
            AuditResultType.RESOURCE_LIMITATION_CONFIRMED,
            AuditResultType.UNDISCLOSED_DETAIL_LIKELY,
            AuditResultType.RANDOMNESS_LIKELY,
        ):
            matches = [f for f in findings if f.result == fallback_type]
            if matches:
                return fallback_type, "; ".join(f.detail for f in matches)

        return (
            AuditResultType.NO_OBVIOUS_ERROR_FOUND,
            "已完成的审计任务均未发现明显问题（论文理解/代码路径/参数/"
            "数据/模型均未确认存在错误）",
        )

    def audit_issue_confirmed(self, reflection_report: ReflectionReport) -> bool:
        """§19: ``audit_issue_confirmed()``。"""

        return reflection_report.issue_found and (
            reflection_report.audit_result in _ISSUE_CONFIRMED_RESULTS
        )

    def plan_repair(
        self,
        reflection_report: ReflectionReport,
        *,
        repository_path: str = "",
        base_image: str = "",
        dataset_paths: list[str] | None = None,
        model_paths: list[str] | None = None,
        checkpoint_paths: list[str] | None = None,
    ) -> list[Task]:
        """§19: ``main_agent.plan_repair()``——依据确认的问题类型创建
        对应的修复任务（代码问题 -> coding，配置/流程问题 -> specification，
        环境问题 -> environment_build）。
        """

        result_to_task_type = {
            AuditResultType.CODE_ERROR_CONFIRMED: "coding",
            AuditResultType.CONFIG_ERROR_CONFIRMED: "specification",
            AuditResultType.PROCESS_ERROR_CONFIRMED: "specification",
            AuditResultType.DATA_ERROR_CONFIRMED: "resource_check",
            AuditResultType.EVALUATION_ERROR_CONFIRMED: "coding",
            AuditResultType.ENVIRONMENT_ERROR_CONFIRMED: "environment_build",
        }
        task_type = result_to_task_type.get(reflection_report.audit_result, "coding")
        # 修复任务的 inputs 同样只携带问题描述文本，没有具体路径：
        #   - specification 类型的 ExperimentSpecificationAgent 完全不
        #     调用 call_tool/call_llm，不需要任何工具；
        #   - resource_check（DATA_ERROR_CONFIRMED）同理没有
        #     dataset/model/checkpoint 路径，check_path_resource 用不上；
        #   - coding/environment_build 是多阶段确定性流程，模板里的每个
        #     工具在某个阶段都会被用到，保留完整模板（不传 restrict_tools）。
        resource_tools = ["check_gpu", "check_cuda", "check_disk_space"]
        if dataset_paths or model_paths or checkpoint_paths:
            resource_tools.append("check_path_resource")
        repair_restrict_tools: dict[str, list[str] | None] = {
            "specification": [],
            "resource_check": resource_tools,
        }
        definition = build_task_definition(
            objective=f"修复已确认问题: {reflection_report.confirmed_issue}",
            task_type=task_type,
            inputs={
                "reflection_id": reflection_report.reflection_id,
                "confirmed_issue": reflection_report.confirmed_issue,
                "fix_instructions": reflection_report.confirmed_issue,
                "repository_path": repository_path,
                "base_image": base_image,
                "dependencies_hint": reflection_report.confirmed_issue,
                "dataset_paths": dataset_paths or [],
                "model_paths": model_paths or [],
                "checkpoint_paths": checkpoint_paths or [],
                "creation_key": (
                    f"repair:{reflection_report.reflection_id}:{task_type}"
                ),
            },
            restrict_tools=repair_restrict_tools.get(task_type),
        )
        return [Task(job_id=reflection_report.job_id, definition=definition)]

    def repair_completed(self, repair_tasks: list[Task]) -> bool:
        """§19: ``repair_completed()``。"""

        return bool(repair_tasks) and all(t.status.value == "SUCCEEDED" for t in repair_tasks)

    def plan_minimum_rerun_scope(
        self,
        reflection_report: ReflectionReport,
        job: ReproductionJob,
        *,
        runs: list[ExperimentRun] | None = None,
        execution_image: str = "",
        repository_path: str = "",
        execution_manifest: dict | None = None,
    ) -> list[Task]:
        """§19: ``main_agent.plan_minimum_rerun_scope()``——按
        ``RerunScope``（§11.7）生成最小重跑范围的任务，而不是无脑重跑
        整个正式实验，节省 GPU 预算。
        """

        exhausted, reason = job.budget_exhausted()
        if exhausted:
            logger.warning("job %s budget exhausted (%s), skipping rerun", job.job_id, reason)
            return []

        scope = reflection_report.recommended_rerun_scope or RerunScope.FULL_TRAINING
        hint = _RERUN_SCOPE_COMMAND_HINT[scope]
        experiment_id = (job.inputs.target_experiments or ["main_experiment"])[0]
        available_runs = runs or []
        parent = next(
            (
                run
                for run in reversed(available_runs)
                if run.experiment_id == experiment_id
                and run.tier == ExperimentTier.FULL_EXPERIMENT
                and run.exit_code == 0
            ),
            None,
        )
        if job.inputs.user_run_commands:
            command = shlex.split(job.inputs.user_run_commands[0])
        elif parent is not None:
            try:
                parsed = json.loads(parent.command)
                command = parsed if isinstance(parsed, list) else shlex.split(parent.command)
            except (json.JSONDecodeError, TypeError):
                command = shlex.split(parent.command)
        else:
            command = []
        if not command:
            logger.error("job %s cannot create rerun without a concrete command", job.job_id)
            return []

        tier = (
            ExperimentTier.FULL_EXPERIMENT
            if scope in {RerunScope.FULL_TRAINING, RerunScope.ENVIRONMENT_REBUILD}
            else ExperimentTier.SMOKE_TEST
        )
        if tier == ExperimentTier.FULL_EXPERIMENT and not TierGate().can_run_full_experiment(
            experiment_id, available_runs
        ):
            logger.error("job %s cannot rerun full experiment before prior tiers pass", job.job_id)
            return []
        manifest = execution_manifest or {
            "config_digest": parent.config_digest if parent is not None else "",
            "model_identifier": parent.model_identifier if parent is not None else "",
            "seed": parent.seed if parent is not None else None,
            "hardware_identifier": parent.hardware_identifier if parent is not None else "",
        }
        definition = build_task_definition(
            objective=f"最小范围重跑: {hint}",
            task_type="experiment_execution",
            dependencies=list(reflection_report.repair_task_ids),
            inputs={
                "rerun_scope": scope.value,
                "reflection_id": reflection_report.reflection_id,
                "experiment_id": experiment_id,
                "tier": tier.value,
                "command": command,
                "repository_path": repository_path or job.inputs.repository_path,
                "dataset_paths": job.inputs.dataset_paths,
                "model_paths": job.inputs.model_paths,
                "checkpoint_paths": job.inputs.checkpoint_paths,
                "metrics_output_path": "output://metrics.json",
                "execution_manifest": manifest,
                "cpu_cores": job.inputs.cpu_cores or 1.0,
                "memory_mb": job.inputs.memory_mb or 1024,
                "disk_mb": job.inputs.disk_mb or 4096,
                "gpu_count": job.inputs.gpu_count or 0,
                "gpu_memory_gb": job.inputs.gpu_memory_gb or 0.0,
                "network_enabled": False,
                "network_hosts": [],
                "timeout_seconds": job.inputs.max_runtime_seconds or 600,
                "tier_command_verified": bool(
                    parent is not None and parent.tier_command_verified
                ),
                "parent_run_id": parent.run_id if parent is not None else "",
                "parent_container_digest": parent.container_digest if parent is not None else "",
                "execution_image": execution_image,
                "creation_key": f"rerun:{reflection_report.reflection_id}:{tier.value}",
            },
            restrict_tools=["execute_command", "read_file", "hash_path"],
        )
        return [Task(job_id=job.job_id, definition=definition)]
