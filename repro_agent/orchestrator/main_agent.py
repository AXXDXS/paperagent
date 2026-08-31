"""主智能体（设计文档 §5、§19 主循环）：整个系统的唯一总控制器。

本类是"主智能体按任务下发受限工具集给子智能体"这一需求的顶层组装点：
它持有全局唯一的 ``ToolAuthorizer``（间接持有 ``ToolRegistry``）、
``SandboxManager``、``MemoryManager`` + ``MainAgentCapability``，
这些"高权限对象"只在 ``MainAgent`` 内部流转，从不传递给子智能体
运行时代码——子智能体只会收到 ``AgentDispatcher`` 为它精心构造的
``ToolAuthorization`` 实例。

主循环严格对齐设计文档 §19 的伪代码结构，逐行对应：

    scheduler.refresh_task_states()      -> self.scheduler.refresh_task_states()
    scheduler/reporting deadline check   -> self._check_subagent_reporting()
    scheduler.check_timeouts()           -> self.scheduler.check_timeouts()
    main_agent.validate_outputs(...)     -> self.validate_outputs(...)
    main_agent.classify_failure(task)    -> self.replanner.classify_failure(task)
    main_agent.decompose(task)           -> self.replanner.decompose(task)
    main_agent.create_prerequisite(task) -> self.replanner.create_prerequisite(task)
    result_gap_detected()                -> self._check_result_gap()
    reflection_agent.analyze()           -> 通过 AgentDispatcher 派发 reflection 任务
    main_agent.plan_audit(...)           -> self.reflection_controller.plan_audit(...)
    audit_issue_confirmed()              -> self.reflection_controller.audit_issue_confirmed(...)
    main_agent.plan_repair()             -> self.reflection_controller.plan_repair(...)
    repair_completed()                   -> self.reflection_controller.repair_completed(...)
    main_agent.plan_minimum_rerun_scope()-> self.reflection_controller.plan_minimum_rerun_scope(...)
    scheduler.select_tasks(...)          -> self.scheduler.select_tasks(...)
    scheduler.dispatch(...)              -> self.scheduler.dispatch(...) + self.dispatcher.dispatch_and_run(...)
    context_manager.save_snapshot()      -> self.snapshot_store.save(...)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from repro_agent.context.builder import ContextBuilder, UnresolvedIssue
from repro_agent.context.snapshot import SnapshotStore
from repro_agent.domain.common import utc_now
from repro_agent.domain.enums import (
    AuditResultType,
    ExperimentTier,
    FailureDecision,
    FailureType,
    InterventionStatus,
    JobStatus,
    ReproductionStatus,
    TaskStatus,
    ToleranceType,
    ToolGrantDecision,
)
from repro_agent.domain.experiment import MetricComparison
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.reflection import AuditFinding, ReflectionHypothesis, ReflectionReport
from repro_agent.domain.task import (
    AgentReport,
    AgentReportType,
    FailureReport,
    Heartbeat,
    Task,
)
from repro_agent.domain.verification import VerificationRecord
from repro_agent.dynamic_tools.lifecycle import DynamicToolLifecycleManager
from repro_agent.execution import (
    ColimaExecutionBackend,
    CondaExecutionBackend,
    DockerExecutionBackend,
    MockExecutionBackend,
)
from repro_agent.execution.environment_naming import managed_environment_name
from repro_agent.memory.candidate import CandidateMemory
from repro_agent.memory.manager import MainAgentCapability, MemoryManager
from repro_agent.orchestrator.dispatcher import AgentDispatcher
from repro_agent.orchestrator.execution_parameters import (
    ExecutionParameterValidationError,
    execution_plan_fingerprint,
    execution_plan_snapshot,
    has_current_execution_parameter_approval,
)
from repro_agent.orchestrator.llm_decision import MainAgentLLMDecisionMaker
from repro_agent.orchestrator.interventions import InterventionService
from repro_agent.orchestrator.planner import InitialPlanner
from repro_agent.orchestrator.phases import PhaseCoordinator
from repro_agent.orchestrator.reflection_controller import ReflectionController
from repro_agent.orchestrator.runtime_accounting import RuntimeAccountingService
from repro_agent.orchestrator.replanner import Replanner
from repro_agent.orchestrator.recovery import RecoveryOutcome, RecoveryService
from repro_agent.orchestrator.runtime_configuration import (
    materialize_runtime_configuration,
    missing_requirements,
    normalize_requirements,
    runtime_network_configuration,
)
from repro_agent.orchestrator.result_tools import GetJobResultTool
from repro_agent.observability.result_query import JobResultService
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.orchestrator.tool_grant import (
    ToolAllocationPlanner,
    ToolGrantDecisionMaker,
    extract_requested_tool_names,
)
from repro_agent.orchestrator.validator import OutputValidator
from repro_agent.providers.metered import MeteredLLMProvider
# Backward-compatible re-export: older integrations monkeypatch this symbol on
# main_agent even though creation now runs through orchestrator.agent_tools.
from repro_agent.orchestrator.artifacts import ArtifactResolver
from repro_agent.orchestrator.agent_tools import CreateSubagentsTool
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.scheduler.scheduler import SchedulerConfig, TaskScheduler
from repro_agent.scheduler.agent_reporting import (
    ReportingOutcome,
    TerminationMode,
    TerminationRecord,
)
from repro_agent.schemas.results import ResultValidationError, TaskResultEnvelope
from repro_agent.storage.database import Database
from repro_agent.storage.repository import (
    EvidenceRepository,
    ExperimentRunRepository,
    InterventionRepository,
    JobRepository,
    ReflectionRepository,
    TaskCheckpointRepository,
    TaskRepository,
    ToolInvocationRepository,
    DynamicToolRepository,
    VerificationRepository,
)
from repro_agent.tools.authorization import ToolAuthorizer
from repro_agent.tools.registry import create_builtin_registry
from repro_agent.tools.result_sanitization import redact_sensitive_text

logger = logging.getLogger(__name__)

@dataclass
class MainAgentConfig:
    memory_root: str = "./project_memory"
    sandbox_root: str = "./sandbox"
    snapshot_root: str = "./context_snapshots"
    db_path: str = "./repro_agent.db"
    model: str = "gpt-4o-mini"
    # 推理型模型或兼容网关返回的思考 token
    # 同样计入 max_tokens：额度太小（4096/1024）时思考阶段就会耗尽预算，
    # content 一个字都不产出（finish_reason=length + 空 content），
    # 表现为确定性的 "empty response" 结构化输出校验失败。因此默认
    # 给出宽裕的输出上限，并同步放宽单次调用的网络超时。
    model_max_tokens: int = 32768
    llm_timeout_seconds: float = 600.0
    mock_execution: bool = False
    # Colima is the default macOS runtime.  It exposes a Docker-compatible CLI,
    # allowing the existing fail-closed container policy to remain unchanged.
    container_runtime: str = "colima"
    # Optional runtime-neutral selector. Empty preserves the historical
    # container_runtime behavior; "conda" enables trusted-local Conda mode.
    environment_backend: str = ""
    conda_executable: str = "conda"
    # ~/.conda/envs is a standard Conda environment directory, so environments
    # appear by name in `conda env list` instead of as anonymous custom paths.
    conda_env_root: str = "~/.conda/envs"
    conda_python_version: str = "3.11"
    # Empty delegates to REPRO_AGENT_MIRROR_POLICY (defaulting to auto).
    mirror_policy: str = ""
    pip_index_urls: tuple[str, ...] = ()
    conda_channels: tuple[str, ...] = ()
    execution_image: str = "python:3.11-slim"
    main_loop_wait_seconds: float = 0.05
    timeout_cancel_grace_seconds: float = 5.0
    intervention_timeout_seconds: int | None = None
    # The normal workflow confirms the complete run plan after resource
    # checking and before environment construction.  Persisted/legacy standalone
    # experiment tasks retain an exact per-dispatch fallback confirmation.
    require_execution_parameter_confirmation: bool = True
    enable_dynamic_tool_growth: bool = True
    model_input_cost_per_million_usd: float = 0.0
    model_output_cost_per_million_usd: float = 0.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass(frozen=True)
class RunLoopOutcome:
    """主循环的显式退出结果，避免把迭代耗尽误报为任务完成。"""

    completed: bool
    terminal_status: JobStatus | None
    reason: str
    iterations: int
    paused: bool = False


class MainAgent:
    """整个 ReproAgent 系统的唯一总控制器（设计文档 §5 原文）。"""

    @staticmethod
    def _create_execution_backend(config: MainAgentConfig):
        if config.mock_execution:
            return MockExecutionBackend()
        runtime = (
            config.environment_backend.strip().lower()
            or config.container_runtime.strip().lower()
        )
        if runtime == "conda":
            return CondaExecutionBackend(
                environment_root=config.conda_env_root,
                conda_binary=config.conda_executable,
                mirror_policy=config.mirror_policy,
                pip_index_urls=config.pip_index_urls,
                conda_channels=config.conda_channels,
            )
        if runtime == "colima":
            return ColimaExecutionBackend()
        if runtime == "docker":
            return DockerExecutionBackend()
        raise ValueError(
            "environment backend must be one of: colima, docker, conda"
        )

    def __init__(self, job: ReproductionJob, config: MainAgentConfig, llm_provider):
        self.job = job
        self.config = config

        self.db = Database(config.db_path)
        self.job_repo = JobRepository(self.db)
        self.task_repo = TaskRepository(self.db)
        self.evidence_repo = EvidenceRepository(self.db)
        self.checkpoint_repo = TaskCheckpointRepository(self.db)
        self.tool_invocation_repo = ToolInvocationRepository(self.db)
        self.experiment_run_repo = ExperimentRunRepository(self.db)
        self.verification_repo = VerificationRepository(self.db)
        self.reflection_repo = ReflectionRepository(self.db)
        self.intervention_repo = InterventionRepository(self.db)
        self.dynamic_tool_repo = DynamicToolRepository(self.db)
        self.runtime_accounting = RuntimeAccountingService(
            self.job,
            job_repo=self.job_repo,
            task_repo=self.task_repo,
            evidence_repo=self.evidence_repo,
            experiment_run_repo=self.experiment_run_repo,
            mock_execution=config.mock_execution,
            input_cost_per_million_usd=config.model_input_cost_per_million_usd,
            output_cost_per_million_usd=config.model_output_cost_per_million_usd,
        )
        metered_llm_provider = MeteredLLMProvider(
            llm_provider, self.runtime_accounting.record_model_usage
        )

        self.scheduler = TaskScheduler(job.job_id, self.task_repo, config.scheduler)
        execution_backend = self._create_execution_backend(config)
        self.sandbox_manager = SandboxManager(
            f"{config.sandbox_root}/{job.job_id}",
            execution_backend=execution_backend,
            execution_image=config.execution_image,
        )
        # Use a fresh built-in registry for this workspace. Generated tools are
        # loaded only from this database, while built-ins remain permanent and
        # never participate in lifecycle decay.
        tool_registry = create_builtin_registry()
        self.dynamic_tool_lifecycle = DynamicToolLifecycleManager(
            self.dynamic_tool_repo, tool_registry
        )
        if config.enable_dynamic_tool_growth:
            self._apply_persisted_dynamic_tool_decisions()
            self.dynamic_tool_lifecycle.load_active_tools()
        self.tool_authorizer = ToolAuthorizer(
            tool_registry,
            invocation_observer=(
                self.dynamic_tool_lifecycle.observe_invocation
                if config.enable_dynamic_tool_growth
                else None
            ),
        )
        # ---- 工具分配权上收（主智能体决策层） ----
        # 运行期：子智能体缺工具时由裁决器决定 GRANT/DENY/ASK_USER；
        # 规划期：派发前按任务内容定制 allowed_tools（模板仅作参考）。
        # 两者都复用 metered provider，成本计入 job 预算。
        self.tool_grant_decision_maker = ToolGrantDecisionMaker(
            self.tool_authorizer, metered_llm_provider, model=config.model, max_tokens=config.model_max_tokens
        )
        self.tool_allocation_planner = ToolAllocationPlanner(
            self.tool_authorizer, metered_llm_provider, model=config.model, max_tokens=config.model_max_tokens
        )
        self.intervention_service = InterventionService(
            self.db, tool_authorizer=self.tool_authorizer
        )
        self.dispatcher = AgentDispatcher(
            self.sandbox_manager,
            self.tool_authorizer,
            metered_llm_provider,
            self.task_repo,
            model=config.model,
            max_tokens=config.model_max_tokens,
            llm_timeout_seconds=config.llm_timeout_seconds,
            # 子智能体窄通道在 MainAgent 中拆分为业务 AgentReport 与
            # 底层 ActivitySignal，再分别持久化。
            on_progress_push=self._on_subagent_progress_push,
            # 运行期缺工具升级通道：裁决器 + 人工介入创建回调。
            tool_grant_decision_maker=self.tool_grant_decision_maker,
            request_human_tool_grant=self._request_human_tool_grant,
        )
        # Main-agent-only orchestration tool.  It is intentionally not part of
        # the child-facing ToolRegistry, so sub-agents cannot create descendants.
        self.create_subagents_tool = CreateSubagentsTool(
            self.scheduler,
            self.dispatcher,
            tool_allocation_planner=self.tool_allocation_planner,
        )
        self.get_job_result_tool = GetJobResultTool(
            JobResultService(self.db, Path(config.db_path).parent),
            current_job_id=job.job_id,
        )
        # Dispatcher 是逐次工具审计的实际写入者；这里公开同一 Repository
        # 供恢复服务、报告与 API 查询使用，避免形成两个不一致的视图。
        self.checkpoint_repo = self.dispatcher.checkpoint_repo
        self.tool_invocation_repo = self.dispatcher.tool_invocation_repo
        self.validator = OutputValidator(self.sandbox_manager)
        self.replanner = Replanner()
        self.reflection_controller = ReflectionController()
        effective_environment_backend = (
            config.environment_backend.strip().lower()
            or config.container_runtime.strip().lower()
        )
        self.planner = InitialPlanner(
            config.execution_image,
            environment_backend=effective_environment_backend,
            conda_python_version=config.conda_python_version,
        )
        self.phase_coordinator = PhaseCoordinator()
        # 调度器和主 Agent 共享同一个动态报备策略；底层活动信号不参与
        # 截止时间续期，避免与业务报备形成两套冲突的判定。
        self.reporting_policy = self.scheduler.reporting_policy
        self._termination_log: list[TerminationRecord] = []

        # 记忆系统：MainAgentCapability 只在这里被构造一次，永远不会
        # 传递给子智能体（§3 原则 12-14 的工程落地，见 memory/manager.py）。
        self.memory_manager = MemoryManager(job.job_id, config.memory_root)
        self._capability = MainAgentCapability(job_id=job.job_id)
        self.context_builder = ContextBuilder(self.memory_manager, self._capability)
        self.snapshot_store = SnapshotStore(config.snapshot_root)
        # 把 §16 九段上下文编排真正接入 §19 决策点的桥梁：目前唯一的
        # 消费方是 _handle_failed_task 里对 UNKNOWN_ERROR 的兜底分类
        # （见 replanner.py 顶部说明与 llm_decision.py 模块文档）。
        self.llm_decision_maker = MainAgentLLMDecisionMaker(
            self.context_builder,
            metered_llm_provider,
            model=config.model,
            max_tokens=config.model_max_tokens,
        )

        self._reflection_reports: list[ReflectionReport] = []
        # reflection_id -> 该轮反思仍在等待完成的审计任务 id 集合。
        # 只有集合清空（该轮审计任务全部结束）时才会调用
        # ``summarize_audit_findings`` 汇总，避免"部分审计任务还没跑完
        # 就仓促下结论"（§11.2："多个子智能体并行检查" -> "汇总审计
        # 结果"是两个独立阶段，后者必须等前者全部完成）。
        self._pending_audit_task_ids: dict[str, set[str]] = {}
        self._decision_version = 0
        self._last_snapshot_fingerprint: str | None = None
        # 已结束线程、结果自称成功、但尚未通过 validate_outputs 独立
        # 校验的任务 id 集合——句柄在这个集合里逗留期间绝不会被回收，
        # 是"验证通过后才能关闭子agent"约束的核心状态。
        self._pending_validation: set[str] = set()
        self._pending_validation_attempts: dict[str, str] = {}
        # 子智能体报备、活动或完成回调会唤醒主循环；超时等待仍作为
        # 容错兜底，从而避免无任务变化时持续空转和刷写快照。
        self._wake_event = threading.Event()
        self._timeout_cancellations: dict[str, float] = {}
        # 与上面的既有取消计时表配套，仅保存终止语义，不再建立第二套
        # 取消状态机。缺项表示普通硬/软超时取消，兼容旧快照和测试注入。
        self._termination_requests: dict[str, dict[str, Any]] = {}

        self.job_repo.save(self.job)

    @classmethod
    def resume_from_storage(
        cls, job_id: str, config: MainAgentConfig, llm_provider
    ) -> "MainAgent":
        """从同一 SQLite work-dir 加载既有 Job，但不立即修改任务状态。"""

        database = Database(config.db_path)
        try:
            job = JobRepository(database).get(job_id)
            tasks = TaskRepository(database).list_by_job(job_id)
        finally:
            database.close()
        if job is None:
            raise ValueError(f"job {job_id} was not found in {config.db_path}")
        if not config.environment_backend.strip():
            environment_tasks = [
                task
                for task in tasks
                if task.definition.task_type == "environment_build"
            ]
            if environment_tasks:
                inputs = environment_tasks[-1].definition.inputs
                persisted_backend = str(inputs.get("environment_backend", "")).strip()
                persisted_python = str(inputs.get("python_version", "")).strip()
                if persisted_backend in {"conda", "colima", "docker"}:
                    config = replace(
                        config,
                        environment_backend=persisted_backend,
                        conda_python_version=(
                            persisted_python or config.conda_python_version
                        ),
                    )
        return cls(job, config, llm_provider)

    def recover_interrupted_tasks(self) -> RecoveryOutcome:
        """安全处理进程中断时处于 DISPATCHED/RUNNING 的任务。"""

        return RecoveryService(self).recover()

    def pending_intervention(self):
        """返回当前待处理的人工介入请求；没有则返回 ``None``。"""

        return self.intervention_repo.get_pending_for_job(self.job.job_id)

    def resolve_intervention(
        self,
        request_id: str,
        payload: dict[str, Any],
        *,
        responded_by: str = "user",
    ):
        """回答请求并同步当前 MainAgent 的内存视图，之后可直接续跑。"""

        # 工具升级（事中审批）路径优先：子智能体线程仍挂起等待，批准后
        # 必须原地唤醒而不是重置 PENDING 重启。
        if self.dispatcher.get_pending_escalation_by_request(request_id) is not None:
            return self._resolve_tool_grant_intervention(
                request_id, payload, responded_by=responded_by
            )

        pending_request = self.intervention_repo.get(request_id)
        resolution = self.intervention_service.resolve(
            request_id, payload, responded_by=responded_by
        )
        if (
            pending_request is not None
            and pending_request.metadata.get("response_mode")
            == "dynamic_tool_activation"
        ):
            self._apply_dynamic_tool_decision(resolution.request)
        self.job = resolution.job
        self.runtime_accounting.job = self.job
        if resolution.task is not None:
            self.scheduler.dag.replace_task(resolution.task)
        if (
            pending_request is not None
            and pending_request.metadata.get("response_mode")
            == "pre_environment_execution_plan"
            and resolution.request.status == InterventionStatus.RESOLVED
        ):
            self._recheck_resources_if_confirmed_plan_changed(resolution.task)
        self._wake_event.set()
        return resolution

    def _resolve_tool_grant_intervention(
        self,
        request_id: str,
        payload: dict[str, Any],
        *,
        responded_by: str,
    ):
        """工具升级人工审批的 resolve：注入裁决并唤醒挂起中的子智能体。

        这是"主 agent 追加工具后子智能体继续运行而不是重启"的人工
        分支：``resume_task=False`` 保证任务不被重置 PENDING，随后
        ``dispatcher.resume_escalation`` 把批准的工具直接并入挂起中
        子智能体手里的 ``ToolAuthorization`` 并 set 事件唤醒它——线程
        从阻塞点继续执行原工具调用，运行时状态零丢失。拒绝则携带
        DENY 裁决唤醒，子智能体按已裁决的权限错误失败。
        """

        escalation = self.dispatcher.get_pending_escalation_by_request(request_id)
        resolution = self.intervention_service.resolve(
            request_id, payload, responded_by=responded_by, resume_task=False
        )
        request = resolution.request
        approved = request.status == InterventionStatus.APPROVED
        response_payload = request.response or {}
        approved_tools = list(response_payload.get("approved_tools") or [])

        task_id = escalation.task_id if escalation is not None else None
        if task_id is not None:
            if approved:
                self.dispatcher.resume_escalation(
                    task_id,
                    approved_tools=approved_tools,
                    reason=str(response_payload.get("reason") or "人工批准补授"),
                )
            else:
                self.dispatcher.fail_escalation(
                    task_id,
                    reason=str(response_payload.get("reason") or "人工拒绝了补授请求"),
                )

        self.job = resolution.job
        self.runtime_accounting.job = self.job
        if resolution.task is not None:
            self.scheduler.dag.replace_task(resolution.task)
        self._wake_event.set()
        return resolution

    def _request_human_tool_grant(self, task: Task, tool_name: str, reason: str):
        """dispatcher 的 ASK_USER 回调：为缺工具升级创建人工介入请求。

        返回 request_id；若当前 Job 已有**其他**挂起中的介入（无法再
        建新的），返回 None——调用方会 fail closed 地把本次升级按拒绝
        处理，绝不无限挂起子智能体线程。
        """

        request = self.intervention_service.create_for_tool_grant(
            self.job,
            task,
            tool_name=tool_name,
            reason=reason,
            timeout_seconds=self.config.intervention_timeout_seconds,
        )
        if (
            request.metadata.get("response_mode") != "tool_grant_escalation"
            or request.metadata.get("escalation_task_id") != task.task_id
        ):
            # 拿到的是其他挂起中的介入（create 的去重返回），不能占用。
            return None
        return request.request_id

    def reject_intervention(
        self,
        request_id: str,
        *,
        reason: str = "user rejected the intervention",
        responded_by: str = "user",
    ):
        """拒绝请求并以 fail-closed 方式终止 Job。"""

        escalation = self.dispatcher.get_pending_escalation_by_request(request_id)
        resolution = self.intervention_service.reject(
            request_id, reason=reason, responded_by=responded_by
        )
        if escalation is not None:
            # 唤醒挂起中的子智能体线程（携带拒绝裁决），避免线程
            # 永远阻塞在已关闭的审批请求上。
            self.dispatcher.fail_escalation(
                escalation.task_id, reason=f"人工介入被拒绝: {reason}"
            )
        self.job = resolution.job
        self.runtime_accounting.job = self.job
        if resolution.task is not None:
            self.scheduler.dag.replace_task(resolution.task)
        self._wake_event.set()
        return resolution

    # ---- 启动：初始规划 ----

    def bootstrap(self) -> None:
        """Job 创建后的首次规划（§6：把用户输入拆解为初始任务 DAG）。"""

        if self.job.status != JobStatus.JOB_CREATED:
            return
        initial_tasks = self.planner.plan_initial_tasks(self.job)
        self.scheduler.add_tasks(initial_tasks)
        self.job.status = JobStatus.PAPER_ANALYSIS_RUNNING
        self.job_repo.save(self.job)

    # ---- 主循环（§19）----

    def run_until_finished(self, *, max_iterations: int = 10_000) -> RunLoopOutcome:
        iteration = 0
        self._expire_overdue_intervention()
        while (
            not self._job_finished()
            and not self._job_waiting_for_user()
            and iteration < max_iterations
        ):
            self.step()
            iteration += 1
            if not self._job_finished() and not self._job_waiting_for_user():
                self._wake_event.wait(timeout=max(0.0, self.config.main_loop_wait_seconds))
                self._wake_event.clear()

        if self._job_finished():
            return RunLoopOutcome(
                completed=True,
                terminal_status=self.job.status,
                reason="terminal_status",
                iterations=iteration,
            )

        if self._job_waiting_for_user():
            return RunLoopOutcome(
                completed=False,
                terminal_status=None,
                reason="waiting_for_user",
                iterations=iteration,
                paused=True,
            )

        logger.warning("job %s reached max_iterations=%d without finishing", self.job.job_id, max_iterations)
        return RunLoopOutcome(
            completed=False,
            terminal_status=None,
            reason="iteration_limit",
            iterations=iteration,
        )

    def step(self) -> None:
        """执行主循环的一次迭代，严格对齐 §19 伪代码顺序。"""

        self._expire_overdue_intervention()
        if self._job_finished() or self._job_waiting_for_user():
            return
        if self.config.enable_dynamic_tool_growth:
            # Pick up tools activated/expired by another Job sharing this
            # workspace database before planning or dispatching new work.
            self.dynamic_tool_lifecycle.load_active_tools()
            if self._request_pending_dynamic_tool_approval():
                self.job_repo.save(self.job)
                return

        self.scheduler.refresh_task_states()
        self._check_subagent_reporting()
        timeout_result = self.scheduler.check_timeouts()
        for task in timeout_result.hard_timeout_tasks:
            self.task_repo.record_event(self.job.job_id, task.task_id, "hard_timeout_observed", {})
            self._begin_timeout_termination(task)
        self._settle_timeout_terminations()

        self._collect_finished_subagents()

        completed_tasks = self._new_completed_tasks()
        self.validate_outputs(completed_tasks)

        failed_tasks = self.scheduler.get_failed_tasks()
        for task in failed_tasks:
            if self._handle_failed_task(task):
                self._save_snapshot()
                self.job_repo.save(self.job)
                return

        self._sync_required_experiment_configuration()

        gap_decision = self._check_result_gap()
        if gap_decision is not None and gap_decision.should_reflect:
            self._trigger_reflection(gap_decision.reason)

        self._advance_reflection_pipeline()
        if self._advance_phases():
            self._save_snapshot()
            self.job_repo.save(self.job)
            return

        budget_reason = self._runtime_budget_limit_reason()
        if budget_reason:
            running = any(
                task.status in {TaskStatus.DISPATCHED, TaskStatus.RUNNING}
                for task in self.scheduler.dag.all_tasks()
            )
            self.task_repo.record_event(
                self.job.job_id,
                None,
                "runtime_budget_exhausted",
                {
                    "reason": budget_reason,
                    "waiting_for_running_tasks": running,
                    "gpu_hours_used": self.job.gpu_hours_used,
                    "model_call_cost_usd": self.job.model_call_cost_usd,
                    "model_calls_made": self.job.model_calls_made,
                },
                event_key=f"budget-exhausted:{budget_reason}",
            )
            if not running:
                self.job.status = JobStatus.FAILED
                self.job_repo.save(self.job)
            self._save_snapshot()
            return

        ready_tasks = self.scheduler.get_ready_tasks()
        selected = self.scheduler.select_tasks(ready_tasks, self.config.scheduler.max_parallel_agents)
        if self._pause_for_required_experiment_configuration(selected):
            self._save_snapshot()
            self.job_repo.save(self.job)
            return
        if self._pause_for_pre_environment_execution_plan(selected):
            self._save_snapshot()
            self.job_repo.save(self.job)
            return
        if self._pause_for_execution_parameter_confirmation(selected):
            self._save_snapshot()
            self.job_repo.save(self.job)
            return
        dispatched = self.scheduler.dispatch(selected)
        if dispatched:
            # One tool call launches the complete READY batch for this step.
            self.create_subagents_tool.call(
                task_ids=[task.task_id for task in dispatched]
            )

        self._save_snapshot()
        self.job_repo.save(self.job)

    def _sync_required_experiment_configuration(self) -> None:
        """Persist spec-normalized must-have runtime configuration."""

        spec_tasks = [
            task
            for task in self.scheduler.dag.all_tasks()
            if task.definition.task_type == "specification"
            and task.status == TaskStatus.SUCCEEDED
            and not task.definition.inputs.get("reflection_id")
        ]
        payload: dict[str, Any] = {}
        if spec_tasks:
            payload = self._read_task_result_json(spec_tasks[-1]) or {}
        if not payload:
            # Compatibility for persisted DAGs created before the specification
            # became the canonical execution-resource contract.
            code_tasks = [
                task
                for task in self.scheduler.dag.all_tasks()
                if task.definition.task_type == "code_analysis"
                and task.status == TaskStatus.SUCCEEDED
            ]
            if not code_tasks:
                return
            payload = self._read_task_result_json(code_tasks[-1]) or {}
        requirements = normalize_requirements(
            payload.get("required_user_configuration", [])
        )
        if requirements != self.job.inputs.required_experiment_configurations:
            self.job.inputs.required_experiment_configurations = requirements
            self.job.inputs.confirmed_execution_plan = {}
            self.job_repo.save(self.job)

    def _pause_for_required_experiment_configuration(
        self, selected_tasks: list[Task]
    ) -> bool:
        """Pause at the resource gate when a required runtime value is absent."""

        requirements = normalize_requirements(
            self.job.inputs.required_experiment_configurations
        )
        if not requirements:
            return False
        for task in selected_tasks:
            if task.definition.task_type != "experiment_execution":
                continue
            missing = missing_requirements(
                requirements,
                self.job.inputs.experiment_runtime_config,
            )
            task.definition.inputs["required_experiment_configurations"] = requirements
            if missing:
                request = (
                    self.intervention_service.create_for_required_experiment_configuration(
                        self.job,
                        task,
                        missing,
                        timeout_seconds=self.config.intervention_timeout_seconds,
                    )
                )
                self.scheduler.dag.replace_task(task)
                self.task_repo.record_event(
                    self.job.job_id,
                    task.task_id,
                    "required_experiment_configuration_requested",
                    {
                        "request_id": request.request_id,
                        "configuration_names": [item["name"] for item in missing],
                        "credential_env_vars": [
                            item["environment_variable"]
                            for item in missing
                            if item["kind"] == "credential_env"
                        ],
                    },
                )
                return True

            command, environment, secret_environment = materialize_runtime_configuration(
                command=list(task.definition.inputs.get("command", [])),
                requirements=requirements,
                runtime_values=self.job.inputs.experiment_runtime_config,
            )
            network_enabled, network_hosts = runtime_network_configuration(
                requirements, self.job.inputs.experiment_runtime_config
            )
            execution_manifest = dict(
                task.definition.inputs.get("execution_manifest", {}) or {}
            )
            for requirement in requirements:
                if requirement["kind"] != "model_name":
                    continue
                model_name = self.job.inputs.experiment_runtime_config.get(
                    requirement["name"]
                )
                if model_name:
                    execution_manifest["model_identifier"] = str(model_name)
                    break
            task.definition.inputs.update(
                {
                    "command": command,
                    "experiment_runtime_config": dict(
                        self.job.inputs.experiment_runtime_config
                    ),
                    "experiment_environment": environment,
                    "experiment_secret_env_vars": secret_environment,
                    "network_enabled": network_enabled,
                    "network_hosts": network_hosts,
                    "execution_manifest": execution_manifest,
                }
            )
            self.task_repo.save(task)
        return False

    def _pause_for_pre_environment_execution_plan(
        self, selected_tasks: list[Task]
    ) -> bool:
        """Confirm the complete run plan before resource/env tasks dispatch."""

        if not self.config.require_execution_parameter_confirmation:
            return False
        all_tasks = list(self.scheduler.dag.all_tasks())
        environment_tasks = [
            task
            for task in all_tasks
            if task.definition.task_type == "environment_build"
            and not task.definition.inputs.get("reflection_id")
        ]
        for task in selected_tasks:
            is_environment_gate = task.definition.task_type == "environment_build"
            if task.definition.task_type == "resource_check" and any(
                task.task_id in env.dependencies for env in environment_tasks
            ):
                self._apply_unconfirmed_resource_defaults(task)
                continue
            if not is_environment_gate:
                continue

            plan = self._confirmed_execution_plan()
            if plan is not None:
                self._apply_execution_plan_to_prerequisite(task, plan)
                continue
            plan_inputs = self._build_pre_environment_execution_plan(
                environment_task=task
            )
            if plan_inputs is None:
                request = self.intervention_service.create_for_missing_command(
                    self.job,
                    timeout_seconds=self.config.intervention_timeout_seconds,
                )
                self.task_repo.record_event(
                    self.job.job_id,
                    task.task_id,
                    "pre_environment_command_requested",
                    {"request_id": request.request_id},
                )
                return True
            request = (
                self.intervention_service.create_for_pre_environment_execution_plan(
                    self.job,
                    task,
                    plan_inputs,
                    requirements=normalize_requirements(
                        self.job.inputs.required_experiment_configurations
                    ),
                    missing=missing_requirements(
                        self.job.inputs.required_experiment_configurations,
                        self.job.inputs.experiment_runtime_config,
                    ),
                    default_execution_image=self.config.execution_image,
                    timeout_seconds=self.config.intervention_timeout_seconds,
                )
            )
            self.scheduler.dag.replace_task(task)
            self.task_repo.record_event(
                self.job.job_id,
                task.task_id,
                "pre_environment_execution_plan_requested",
                {
                    "request_id": request.request_id,
                    "plan_fingerprint": request.metadata.get("plan_fingerprint", ""),
                },
            )
            return True
        return False

    def _apply_unconfirmed_resource_defaults(self, task: Task) -> None:
        """Run the first resource probe with current/default requested limits."""

        task.definition.inputs.update(
            {
                "requested_cpu_cores": self.job.inputs.cpu_cores or 1.0,
                "requested_memory_mb": self.job.inputs.memory_mb or 1024,
                "requested_disk_mb": self.job.inputs.disk_mb or 4096,
                "requested_gpu_count": self.job.inputs.gpu_count or 0,
                "requested_gpu_memory_gb": self.job.inputs.gpu_memory_gb or 0.0,
            }
        )
        self.task_repo.save(task)

    def _recheck_resources_if_confirmed_plan_changed(
        self, environment_task: Task | None
    ) -> None:
        if environment_task is None:
            return
        plan = self._confirmed_execution_plan()
        if plan is None:
            return
        resource_tasks = [
            self.scheduler.dag.get(task_id)
            for task_id in environment_task.dependencies
        ]
        resource = next(
            (
                task
                for task in resource_tasks
                if task is not None and task.definition.task_type == "resource_check"
            ),
            None,
        )
        if resource is None:
            return
        expected = {
            "requested_cpu_cores": plan["cpu_cores"],
            "requested_memory_mb": plan["memory_mb"],
            "requested_disk_mb": plan["disk_mb"],
            "requested_gpu_count": plan["gpu_count"],
            "requested_gpu_memory_gb": plan["gpu_memory_gb"],
        }
        if all(resource.definition.inputs.get(key) == value for key, value in expected.items()):
            return
        resource.definition.inputs.update(expected)
        self.scheduler.retry(
            resource,
            guidance=(
                "用户在环境前合并确认中修改了资源参数；按最终确认值重新检查，"
                "通过后再允许环境构建。"
            ),
        )
        self.task_repo.record_event(
            self.job.job_id,
            resource.task_id,
            "resource_recheck_after_execution_plan_change",
            {"plan_fingerprint": execution_plan_fingerprint(plan)},
        )

    def _build_pre_environment_execution_plan(
        self, *, environment_task: Task
    ) -> dict[str, Any] | None:
        tasks = list(self.scheduler.dag.all_tasks())
        requirements = normalize_requirements(
            self.job.inputs.required_experiment_configurations
        )
        tier_commands: dict[str, list[str]] = {}
        experiment_environment: dict[str, str] = {}
        secret_environment: list[str] = []
        for tier in ExperimentTier:
            command = self.phase_coordinator._command_for(self.job, tasks, tier)
            if not command:
                return None
            command, environment, secrets = materialize_runtime_configuration(
                command=command,
                requirements=requirements,
                runtime_values=self.job.inputs.experiment_runtime_config,
            )
            tier_commands[tier.value] = command
            experiment_environment.update(environment)
            secret_environment.extend(secrets)
        network_enabled, network_hosts = runtime_network_configuration(
            requirements, self.job.inputs.experiment_runtime_config
        )
        return {
            "tier_commands": tier_commands,
            "base_image": environment_task.definition.inputs.get("base_image")
            or self.config.execution_image,
            "working_dir": "workspace://repository",
            "timeout_seconds": self.job.inputs.max_runtime_seconds or 600,
            "cpu_cores": self.job.inputs.cpu_cores or 1.0,
            "memory_mb": self.job.inputs.memory_mb or 1024,
            "disk_mb": self.job.inputs.disk_mb or 4096,
            "gpu_count": self.job.inputs.gpu_count or 0,
            "gpu_memory_gb": self.job.inputs.gpu_memory_gb or 0.0,
            "metrics_output_path": "output://metrics.json",
            "experiment_environment": experiment_environment,
            "experiment_secret_env_vars": sorted(set(secret_environment)),
            "network_enabled": network_enabled,
            "network_hosts": network_hosts,
        }

    def _confirmed_execution_plan(self) -> dict[str, Any] | None:
        raw = self.job.inputs.confirmed_execution_plan
        if not raw:
            return None
        try:
            return execution_plan_snapshot(
                raw, default_execution_image=self.config.execution_image
            )
        except ExecutionParameterValidationError:
            # Persisted plans from an older/incomplete schema must be reviewed
            # again instead of silently authorizing environment construction.
            self.job.inputs.confirmed_execution_plan = {}
            self.job_repo.save(self.job)
            return None

    def _apply_execution_plan_to_prerequisite(
        self, task: Task, plan: dict[str, Any]
    ) -> None:
        if task.definition.task_type == "resource_check":
            task.definition.inputs.update(
                {
                    "requested_cpu_cores": plan["cpu_cores"],
                    "requested_memory_mb": plan["memory_mb"],
                    "requested_disk_mb": plan["disk_mb"],
                    "requested_gpu_count": plan["gpu_count"],
                    "requested_gpu_memory_gb": plan["gpu_memory_gb"],
                }
            )
        elif task.definition.task_type == "environment_build":
            task.definition.inputs.update(
                {
                    "base_image": plan["base_image"],
                    "cpu_cores": plan["cpu_cores"],
                    "memory_mb": plan["memory_mb"],
                    "disk_mb": plan["disk_mb"],
                    "gpu_count": plan["gpu_count"],
                    "gpu_memory_gb": plan["gpu_memory_gb"],
                }
            )
        task.definition.inputs["pre_environment_plan_fingerprint"] = (
            execution_plan_fingerprint(plan)
        )
        self.task_repo.save(task)

    def _experiment_matches_confirmed_execution_plan(self, task: Task) -> bool:
        plan = self._confirmed_execution_plan()
        if plan is None:
            return False
        tier = task.definition.inputs.get("tier")
        if tier not in plan["tier_commands"]:
            return False
        expected = {
            "command": plan["tier_commands"][tier],
            "working_dir": plan["working_dir"],
            "timeout_seconds": plan["timeout_seconds"],
            "cpu_cores": plan["cpu_cores"],
            "memory_mb": plan["memory_mb"],
            "disk_mb": plan["disk_mb"],
            "gpu_count": plan["gpu_count"],
            "gpu_memory_gb": plan["gpu_memory_gb"],
            "metrics_output_path": plan["metrics_output_path"],
            "experiment_environment": plan["experiment_environment"],
            "experiment_secret_env_vars": plan["experiment_secret_env_vars"],
            "network_enabled": plan["network_enabled"],
            "network_hosts": plan["network_hosts"],
        }
        return all(task.definition.inputs.get(key) == value for key, value in expected.items())

    def _pause_for_execution_parameter_confirmation(self, selected_tasks: list[Task]) -> bool:
        """Create the pre-run confirmation request for an unapproved experiment.

        ``TaskScheduler.dispatch`` increments ``task.attempt``.  The approval
        is therefore bound to ``attempt + 1`` here, and a retry necessarily
        returns through this gate instead of reusing a confirmation for a
        failed run.
        """

        if not self.config.require_execution_parameter_confirmation:
            return False
        for task in selected_tasks:
            if task.definition.task_type != "experiment_execution":
                continue
            if self._experiment_matches_confirmed_execution_plan(task):
                task.definition.inputs.update(
                    {
                        "tier_command_verified": True,
                        "pre_environment_plan_fingerprint": execution_plan_fingerprint(
                            self.job.inputs.confirmed_execution_plan
                        ),
                    }
                )
                self.task_repo.save(task)
                continue
            if has_current_execution_parameter_approval(
                task.definition.inputs,
                next_attempt=task.attempt + 1,
                default_execution_image=self.config.execution_image,
            ):
                continue
            request = self.intervention_service.create_for_execution_parameters(
                self.job,
                task,
                default_execution_image=self.config.execution_image,
                timeout_seconds=self.config.intervention_timeout_seconds,
            )
            self.scheduler.dag.replace_task(task)
            self.task_repo.record_event(
                self.job.job_id,
                task.task_id,
                "execution_parameter_confirmation_requested",
                {
                    "request_id": request.request_id,
                    "parameter_fingerprint": request.metadata.get("parameter_fingerprint", ""),
                    "attempt_to_approve": task.attempt + 1,
                },
            )
            return True
        return False

    # ---- validate_outputs（§19）----

    def validate_outputs(self, tasks: list[Task]) -> None:
        """校验任务产出；无论通过与否，本方法都是"关闭子agent"的唯一收口点。

        呼应用户需求"主agent对子agent返回的结果必须要验证，验证通过后，
        才可以关闭子agent"：``discard_handle`` 只在这里调用（校验分支
        执行完毕之后），``_collect_finished_subagents`` 发现线程结束后
        绝不会自己 discard 一个"成功"的句柄——必须先走到这里完成独立
        校验。校验失败的任务同样在这里回收句柄：失败任务的后续重试会
        通过 ``scheduler.retry`` 重新走一次完整的
        ``sandbox_manager.create_sandbox`` + ``dispatcher.start_async``
        流程，产生一个全新的句柄，旧句柄没有继续保留的意义。
        """

        for task in tasks:
            self._pending_validation.discard(task.task_id)
            pending_attempt = self._pending_validation_attempts.pop(task.task_id, None)
            if pending_attempt != task.active_attempt_id:
                self.task_repo.record_event(
                    self.job.job_id,
                    task.task_id,
                    "stale_attempt_result_rejected",
                    {
                        "result_attempt_id": pending_attempt,
                        "active_attempt_id": task.active_attempt_id,
                    },
                )
                self.dispatcher.discard_handle(
                    task.task_id, attempt_id=pending_attempt
                )
                continue
            validation = self.validator.validate(task, agent_succeeded=True)
            if validation.passed:
                self.scheduler.mark_succeeded(task, validation.outputs)
                self._resume_experiments_after_code_repair(task)
                self._resume_experiments_after_environment_repair(task)
                self.runtime_accounting.persist_task_evidence(task)
                self._request_resource_intervention_if_needed(task)
                self._request_spec_approval_if_needed(task)
                self._persist_experiment_run(task)
                self._persist_verification_record(task)
                self._promote_candidate_memory(task)
                self._process_dynamic_tool_lifecycle(task)
                self.task_repo.record_event(
                    self.job.job_id, task.task_id, "subagent_result_validated", {"passed": True}
                )
                # 反思闭环胶水逻辑（§11.2）：一个任务"成功且通过独立校验"
                # 之后，才是反思闭环可以安全消费其产出的时机——校验失败
                # 的 reflection/审计任务不应该被当作有效结论纳入汇总。
                self._on_task_validated_for_reflection(task)
            else:
                logger.info("task %s failed output validation: %s", task.task_id, validation.reasons)
                self.scheduler.mark_failed(task, TaskStatus.VALIDATION_FAILED)
                self.task_repo.record_event(
                    self.job.job_id,
                    task.task_id,
                    "subagent_result_validated",
                    {"passed": False, "reasons": validation.reasons},
                )
            # 无论校验通过与否，子智能体的本次运行到这里都已经有了
            # 明确的最终裁决，句柄可以安全释放。
            self.dispatcher.discard_handle(
                task.task_id, attempt_id=task.active_attempt_id
            )

    def _runtime_budget_limit_reason(self) -> str:
        return self.runtime_accounting.budget_limit_reason()

    def _persist_task_evidence(self, task: Task) -> None:
        self.runtime_accounting.persist_task_evidence(task)

    def _request_spec_approval_if_needed(self, task: Task) -> bool:
        if task.definition.task_type != "specification":
            return False
        if task.definition.inputs.get("audit_hypothesis_id"):
            return False
        payload = self._read_task_result_json(task) or {}
        raw_conflicts = list(payload.get("unresolved_conflicts") or [])
        conflict_fields = [
            item.get("field", "") if isinstance(item, dict) else str(item)
            for item in raw_conflicts
        ]
        conflict_fields = [name for name in conflict_fields if name]
        if not conflict_fields:
            return False
        fields = payload.get("fields", {}) or {}
        primary_values = {
            name: fields[name].get("value")
            for name in conflict_fields
            if isinstance(fields.get(name), dict) and "value" in fields[name]
        }
        self.intervention_service.create_for_spec_conflicts(
            self.job,
            task,
            conflict_fields=conflict_fields,
            primary_values=primary_values,
            timeout_seconds=self.config.intervention_timeout_seconds,
        )
        self.scheduler.dag.replace_task(task)
        return True

    def _request_resource_intervention_if_needed(self, task: Task) -> bool:
        if task.definition.task_type != "resource_check":
            return False
        payload = self._read_task_result_json(task) or {}
        blocking_issues = list(payload.get("blocking_issues") or [])
        if not blocking_issues:
            return False
        self.intervention_service.create_for_missing_resources(
            self.job,
            task,
            blocking_issues,
            missing_required_resources=list(
                payload.get("missing_required_resources") or []
            ),
            timeout_seconds=self.config.intervention_timeout_seconds,
        )
        self.scheduler.dag.replace_task(task)
        return True

    def _collect_finished_subagents(self) -> None:
        """轮询所有仍处于 RUNNING 的任务对应的后台线程句柄是否已经跑完。

        这是切换到"子智能体后台线程异步执行"模型后，替代原先"同步
        dispatch_and_run 后立即校验"的轮询入口：``_run_dispatched_task``
        现在只负责 ``start_async`` 后立即返回，不再阻塞主循环；真正的
        "任务是否已经产出结果"由本方法在每次 ``step()`` 迭代里检查。

        找到已结束的线程后，从句柄取出 ``AgentRunResult``：
            - 成功 -> 记录到 ``self._pending_validation``，交给
              ``_new_completed_tasks``/``validate_outputs`` 做独立校验
              （校验通过前绝不 ``discard_handle``，呼应"验证通过后才能
              关闭子agent"）；
            - 失败 -> 直接按失败路径处理并在这里回收句柄（失败任务不需要
              输出校验，``_handle_failed_task`` 分类后即可安全关闭）。
        """

        for task in list(self.scheduler.dag.all_tasks()):
            if task.status != TaskStatus.RUNNING:
                continue
            handle = self.dispatcher.get_handle(task.task_id)
            if handle is None or not handle.is_finished():
                continue

            try:
                result = handle.collect_result()
            except Exception as exc:  # noqa: BLE001 - 线程内未捕获异常的兜底
                if handle.attempt_id != task.active_attempt_id:
                    self._reject_stale_attempt(task, handle.attempt_id)
                    continue
                logger.exception("task %s sub-agent thread raised unhandled exception", task.task_id)
                from repro_agent.domain.enums import FailureType
                from repro_agent.domain.task import FailureReport

                self.scheduler.mark_failed(
                    task,
                    TaskStatus.FAILED_RETRYABLE,
                    FailureReport(
                        failure_type=FailureType.UNKNOWN_ERROR,
                        failed_step="subagent_thread",
                        error_message=str(exc),
                        likely_causes=["子智能体线程内出现未捕获异常"],
                        recommended_action="检查子智能体实现日志，必要时重试",
                    ),
                )
                self.dispatcher.discard_handle(
                    task.task_id, attempt_id=task.active_attempt_id
                )
                continue

            self._accept_attempt_result(task, handle, result)

    def _accept_attempt_result(self, task: Task, handle, result) -> bool:
        """仅接纳当前活跃 attempt 的结果，晚到结果只记审计事件。"""

        if handle.attempt_id != task.active_attempt_id:
            self._reject_stale_attempt(task, handle.attempt_id)
            return False

        if result.succeeded:
            # 成功任务先放进"待校验"集合，句柄暂不回收——只有
            # validate_outputs 判定通过之后才允许 discard_handle。
            self._pending_validation.add(task.task_id)
            self._pending_validation_attempts[task.task_id] = handle.attempt_id
        else:
            self.scheduler.mark_failed(
                task, TaskStatus.FAILED_RETRYABLE, result.failure_report
            )
            self.dispatcher.discard_handle(
                task.task_id, attempt_id=handle.attempt_id
            )
        return True

    def _reject_stale_attempt(self, task: Task, attempt_id: str | None) -> None:
        self.task_repo.record_event(
            self.job.job_id,
            task.task_id,
            "stale_attempt_result_rejected",
            {
                "result_attempt_id": attempt_id,
                "active_attempt_id": task.active_attempt_id,
            },
        )
        self.dispatcher.discard_handle(task.task_id, attempt_id=attempt_id)

    def _new_completed_tasks(self) -> list[Task]:
        """返回本轮"子智能体线程已结束且自称成功、尚待独立校验"的任务。

        真正的"线程是否跑完"判定在 ``_collect_finished_subagents`` 里
        完成，本方法只是把待校验队列转换成 ``Task`` 列表交给
        ``validate_outputs``，两者拆分是为了让"轮询已完成线程"和
        "决定要不要校验"两个关注点保持独立、可分别单测。
        """

        tasks = []
        for task_id in list(self._pending_validation):
            task = self.scheduler.dag.get(task_id)
            if task is not None:
                tasks.append(task)
            else:
                self._pending_validation.discard(task_id)
        return tasks

    # ---- 子智能体动态报备：业务租约 + 活动信号 + 到期主动查询 ----

    def _on_subagent_progress_push(
        self,
        task: Task,
        progress: float,
        current_step: str,
        eta_seconds,
    ) -> None:
        """接收子 Agent 的窄通道消息并按语义分流。

        所有消息都会形成活动快照；只有业务报告才更新下一报备截止
        时间。``activity:`` 前缀专用于工具/等待状态，绝不续期。
        """

        activity = Heartbeat(
            progress=progress,
            current_step=current_step,
            last_completed_step=(
                task.heartbeat.last_completed_step if task.heartbeat else ""
            ),
            eta_seconds=eta_seconds,
            reported_by="push",
        )
        self.scheduler.report_heartbeat(task.task_id, activity)
        if current_step.startswith("activity:"):
            self._wake_event.set()
            return

        report_type = {
            "started": AgentReportType.STARTED,
            "completed": AgentReportType.COMPLETED,
            "failed": AgentReportType.FAILED,
        }.get(current_step, AgentReportType.PROGRESS)
        report = AgentReport(
            attempt_id=task.active_attempt_id,
            report_type=report_type,
            progress=min(1.0, max(0.0, float(progress))),
            current_step=current_step,
            eta_seconds=eta_seconds,
            next_report_after_seconds=eta_seconds,
            reported_by="push",
        )
        self.scheduler.report_agent(task.task_id, report)
        self._wake_event.set()

    def get_subagent_status(self, task_id: str):
        """主智能体主动查询（pull）子智能体当前状态。

        这是需求原文"拉取（pull）：主 Agent 调用
        get_subagent_status(agent_id)，返回子 Agent 的当前状态...
        进度及最近活动时间"对应的公开接口，任何时候都可以调用（不需要
        等到"超时未汇报"才能用），但只有在超时未汇报的场景下，主循环
        才会把它的结果用于延期/终止裁决（见
        ``_check_subagent_reporting``），避免把"随手查一下"和"报备
        到期后的关键查询"混为一谈。
        """

        handle = self.dispatcher.get_handle(task_id)
        if handle is None:
            return None
        heartbeat = handle.pull_status()
        task = self.scheduler.dag.get(task_id)
        if task is not None:
            self.scheduler.report_heartbeat(task_id, heartbeat)
        return heartbeat

    def _check_subagent_reporting(self) -> None:
        """Enforce dynamic report deadlines without fixed heartbeat polling.

        At the promised time, a finished handle is left to the normal result
        collector.  Otherwise the main Agent actively pulls status.  A live
        response creates exactly one EXTENSION report; the third extension
        exhausts the reporting budget and starts terminal cancellation.
        """

        for task in list(self.scheduler.dag.all_tasks()):
            if task.status != TaskStatus.RUNNING:
                continue
            if task.task_id in self._timeout_cancellations:
                continue
            handle = self.dispatcher.get_handle(task.task_id)
            if handle is None:
                continue
            if handle.is_finished():
                continue

            decision = self.reporting_policy.evaluate(task)
            if decision.outcome != ReportingOutcome.DUE:
                continue

            logger.warning(
                "task %s reached report deadline (%s); pulling status",
                task.task_id,
                decision.detail,
            )
            self.task_repo.record_event(
                self.job.job_id,
                task.task_id,
                "agent_report_deadline_reached",
                {
                    "detail": decision.detail,
                    "due_at": decision.due_at.isoformat()
                    if decision.due_at
                    else None,
                    "seconds_overdue": decision.seconds_overdue,
                    "overrun_report_count": task.overrun_report_count,
                },
            )

            heartbeat = self.get_subagent_status(task.task_id)
            if heartbeat is None or not handle.is_alive():
                task.failure_report = FailureReport(
                    failure_type=FailureType.AGENT_STALLED,
                    failed_step="report_deadline_status_pull",
                    error_message="status pull could not confirm the sub-agent is alive",
                    recommended_action="确认执行后端已停止后再重试",
                )
                self.task_repo.save(task)
                self._terminate_subagent(
                    task,
                    handle,
                    reason="status pull could not confirm the sub-agent is alive",
                    terminal=False,
                )
                continue

            extension = AgentReport(
                attempt_id=task.active_attempt_id,
                report_type=AgentReportType.EXTENSION,
                progress=heartbeat.progress,
                current_step=heartbeat.current_step,
                eta_seconds=heartbeat.eta_seconds,
                next_report_after_seconds=heartbeat.eta_seconds,
                reason="promised completion time reached; main agent pulled live status",
                evidence={"thread_alive": True},
                reported_by="pull",
            )
            self.scheduler.report_agent(task.task_id, extension)
            limit = max(1, int(task.definition.max_overrun_reports))
            if task.overrun_report_count < limit:
                continue

            task.reporting_exhausted = True
            task.failure_report = FailureReport(
                failure_type=FailureType.AGENT_STALLED,
                failed_step="reporting_budget_exhausted",
                last_successful_step=heartbeat.last_completed_step,
                error_message=(
                    f"sub-agent remained unfinished after {limit} overdue reports"
                ),
                likely_causes=["执行持续超过多次由子 Agent 给出的完成预估"],
                recommended_action="保留证据并如实上报；需要时由用户显式创建新任务",
                metadata={
                    "overrun_report_count": task.overrun_report_count,
                    "max_overrun_reports": limit,
                },
            )
            self.task_repo.save_with_event(
                task,
                "agent_reporting_budget_exhausted",
                task.failure_report.to_dict(),
                event_key=(
                    f"report-budget-exhausted:{task.task_id}:"
                    f"{task.active_attempt_id}"
                ),
            )
            self._terminate_subagent(
                task,
                handle,
                reason=task.failure_report.error_message,
                terminal=True,
            )

    def _terminate_subagent(
        self,
        task: Task,
        handle,
        *,
        reason: str,
        terminal: bool = False,
    ) -> None:
        """Start non-blocking cancellation through the shared termination path."""

        self._begin_subagent_termination(
            task, reason=reason, terminal=terminal, handle=handle
        )

    @staticmethod
    def _now():
        from repro_agent.domain.common import utc_now

        return utc_now()

    # ---- 失败分类与处理（§19）----

    def _handle_failed_task(self, task: Task) -> bool:
        if task.status == TaskStatus.TERMINAL_FAILURE:
            # Terminal means no automatic retry/replan.  In particular, a task
            # that exhausted three report extensions must not re-enter the same
            # loop merely because AGENT_STALLED is normally retryable.
            return False
        if task.task_id in self._timeout_cancellations:
            # 重试必须等当前执行句柄明确退出；否则旧/新 attempt 会并行运行。
            return False
        if (
            task.definition.task_type == "experiment_execution"
            and task.failure_report is not None
            and task.failure_report.failure_type == FailureType.CODE_ERROR
        ):
            self._plan_execution_code_repair(task)
            return False
        if (
            task.definition.task_type == "experiment_execution"
            and task.failure_report is not None
            and task.failure_report.failure_type == FailureType.ENVIRONMENT_ERROR
        ):
            self._plan_execution_environment_repair(task)
            return False
        # ---- 工具分配权上收：权限类失败先由主智能体裁决能否补授 ----
        # 覆盖子智能体线程已死（未走挂起通道）的场景，比如
        # describe_granted 在暴露工具名时直接抛错：先裁决，
        # GRANT 则把工具写进白名单后重试；拿不准才转人工。
        adjudication = self._adjudicate_permission_failure(task)
        if adjudication == "retry":
            self.scheduler.retry(
                task,
                guidance="主智能体已补授缺失工具，请重新执行并正常使用它们。",
            )
            return False
        if adjudication == "ask_user":
            request = self.intervention_service.create_for_failure(
                self.job,
                task,
                timeout_seconds=self.config.intervention_timeout_seconds,
            )
            self.scheduler.dag.replace_task(task)
            logger.info(
                "job %s paused for tool-grant intervention %s (task=%s)",
                self.job.job_id,
                request.request_id,
                task.task_id,
            )
            return True

        decision = self.replanner.classify_failure(
            task, llm_fallback=self._classify_failure_via_llm
        )
        self.task_repo.record_event(
            self.job.job_id, task.task_id, "failure_classified", {"decision": decision.value}
        )

        if decision.value == "retry":
            self.scheduler.retry(
                task,
                guidance=self._summarize_retry_guidance(task),
            )
        elif decision.value == "split":
            new_tasks = self.replanner.decompose(task)
            if new_tasks:
                self.scheduler.replace_with_subtasks(task, new_tasks)
            else:
                self.scheduler.retry(
                    task,
                    guidance=(
                        "该任务类型不能安全拆成多个等价副本；请收窄当前步骤的"
                        "输入和上下文后重试。"
                    ),
                )
        elif decision.value == "add_prerequisite":
            prerequisite = self.replanner.create_prerequisite(task)
            self.scheduler.block_until(task, prerequisite)
        elif decision.value == "ask_user":
            request = self.intervention_service.create_for_failure(
                self.job,
                task,
                timeout_seconds=self.config.intervention_timeout_seconds,
            )
            self.scheduler.dag.replace_task(task)
            logger.info(
                "job %s paused for intervention %s (task=%s)",
                self.job.job_id,
                request.request_id,
                task.task_id,
            )
            return True
        else:
            self.scheduler.mark_terminal_failure(task, reason="classified as terminal_failure")
        return False

    def _plan_execution_code_repair(self, task: Task) -> None:
        """Block a failed run on one isolated, attempt-bound coding repair."""

        if task.attempt >= task.definition.max_attempts:
            self.scheduler.mark_terminal_failure(
                task,
                reason="code repair attempt budget exhausted",
            )
            return
        source_repository = self._repository_used_by_failed_execution(task)
        repair = self.replanner.create_code_repair(
            task,
            repository_path=source_repository,
        )
        task.definition.inputs["code_repair_source_attempt_id"] = (
            task.active_attempt_id
        )
        canonical = self.scheduler.block_until(
            task,
            repair,
            prerequisite_input_key="code_repair_task_id",
        )
        self.task_repo.record_event(
            self.job.job_id,
            task.task_id,
            "execution_code_repair_planned",
            {
                "repair_task_id": canonical.task_id,
                "failed_attempt_id": task.active_attempt_id,
                "failed_tier": task.definition.inputs.get("tier", ""),
            },
            event_key=(
                f"execution-code-repair:{task.task_id}:"
                f"{task.active_attempt_id}"
            ),
        )

    def _plan_execution_environment_repair(self, task: Task) -> None:
        """Block the failed run on a forced environment rebuild prerequisite."""

        if task.attempt >= task.definition.max_attempts:
            self.scheduler.mark_terminal_failure(
                task,
                reason="environment repair attempt budget exhausted",
            )
            return
        repair = self.replanner.create_environment_repair(
            task,
            repository_path=self._repository_used_by_failed_execution(task),
        )
        # The logical environment identity belongs to the job, not to an
        # attempt-specific staged repository path.  Legacy experiment tasks may
        # not carry it, so bind the stable job-level name before scheduling.
        repair.definition.inputs["environment_name"] = managed_environment_name(
            task.definition.inputs.get("environment_name")
            or self.job.inputs.environment_name,
            self.job.inputs.repository_path,
        )
        canonical = self.scheduler.block_until(
            task,
            repair,
            prerequisite_input_key="environment_repair_task_id",
        )
        self.task_repo.record_event(
            self.job.job_id,
            task.task_id,
            "execution_environment_repair_planned",
            {
                "environment_task_id": canonical.task_id,
                "failed_attempt_id": task.active_attempt_id,
            },
            event_key=(
                f"execution-environment-repair:{task.task_id}:"
                f"{task.active_attempt_id}"
            ),
        )

    def _repository_used_by_failed_execution(self, task: Task) -> str:
        sandbox = self.sandbox_manager.get(task.task_id)
        if sandbox is not None:
            repository = sandbox.workspace_dir / "repository"
            if repository.is_dir():
                return str(repository.resolve())
        return self.job.inputs.repository_path

    def _resume_experiments_after_code_repair(self, repair_task: Task) -> None:
        """Bind a validated repair snapshot to blocked experiment retries."""

        if (
            repair_task.definition.task_type != "coding"
            or not repair_task.definition.inputs.get("source_failed_task_id")
        ):
            return
        repair_sandbox = self.sandbox_manager.get(repair_task.task_id)
        if repair_sandbox is not None:
            repository = repair_sandbox.workspace_dir / "repository"
        else:
            repository = (
                self.sandbox_manager.sandbox_root
                / f"task_{repair_task.task_id}"
                / repair_task.active_attempt_id
                / "workspace"
                / "repository"
            )
        repair_payload = self._read_task_result_json(repair_task) or {}

        for experiment in self.scheduler.dag.all_tasks():
            if (
                experiment.definition.task_type != "experiment_execution"
                or experiment.definition.inputs.get("code_repair_task_id")
                != repair_task.task_id
                or experiment.status != TaskStatus.BLOCKED
            ):
                continue
            if not repository.is_dir():
                self.scheduler.mark_terminal_failure(
                    experiment,
                    reason=(
                        "validated code repair repository is missing: "
                        f"{repository}"
                    ),
                )
                continue
            experiment.definition.inputs.update(
                {
                    "repository_path": str(repository.resolve()),
                    "dataset_paths": list(self.job.inputs.dataset_paths),
                    "model_paths": list(self.job.inputs.model_paths),
                    "checkpoint_paths": list(self.job.inputs.checkpoint_paths),
                    "applied_code_repair_task_id": repair_task.task_id,
                    "applied_code_repair_digest": repair_payload.get(
                        "modified_repository_digest", ""
                    ),
                }
            )
            experiment.definition.inputs.pop("repository_workdir", None)
            self.scheduler.retry(
                experiment,
                guidance=(
                    f"代码修复任务 {repair_task.task_id} 已通过回归测试；"
                    "使用修复后的隔离仓库，从原失败层级重新执行。"
                ),
            )
            self.task_repo.record_event(
                self.job.job_id,
                experiment.task_id,
                "execution_resumed_after_code_repair",
                {
                    "repair_task_id": repair_task.task_id,
                    "repository_path": str(repository.resolve()),
                },
                event_key=(
                    f"execution-resumed-after-repair:{experiment.task_id}:"
                    f"{repair_task.task_id}"
                ),
            )

    def _resume_experiments_after_environment_repair(self, repair_task: Task) -> None:
        """Bind a validated rebuilt environment to the blocked experiment."""

        if (
            repair_task.definition.task_type != "environment_build"
            or not repair_task.definition.inputs.get("environment_repair")
        ):
            return
        payload = self._read_task_result_json(repair_task) or {}
        environment_ref = str(
            payload.get("environment_ref") or payload.get("image_ref", "")
        )
        if not environment_ref:
            return
        for experiment in self.scheduler.dag.all_tasks():
            if (
                experiment.definition.task_type != "experiment_execution"
                or experiment.definition.inputs.get("environment_repair_task_id")
                != repair_task.task_id
                or experiment.status != TaskStatus.BLOCKED
            ):
                continue
            experiment.definition.inputs.update(
                {
                    "execution_image": environment_ref,
                    "applied_environment_repair_task_id": repair_task.task_id,
                    "applied_environment_fingerprint": payload.get(
                        "environment_fingerprint", ""
                    ),
                }
            )
            self.scheduler.retry(
                experiment,
                guidance=(
                    f"环境任务 {repair_task.task_id} 已强制重建并通过 import 自检；"
                    "使用新环境从原失败层级重新执行。"
                ),
            )
            self.task_repo.record_event(
                self.job.job_id,
                experiment.task_id,
                "execution_resumed_after_environment_repair",
                {
                    "environment_task_id": repair_task.task_id,
                    "environment_ref": environment_ref,
                },
                event_key=(
                    f"execution-resumed-after-environment-repair:"
                    f"{experiment.task_id}:{repair_task.task_id}"
                ),
            )

    def _adjudicate_permission_failure(self, task: Task) -> Optional[str]:
        """对权限类失败先做一次主智能体裁决（工具分配权上收的兑底分支）。

        返回值：
            - ``"retry"``：裁决补授了至少一个工具并已写入任务白名单，
              调用方应直接重试（重启后新工具会随授权生效）；
            - ``"ask_user"``：裁决器拿不准，应创建人工介入；
            - ``None``：不适用（非权限失败 / 已裁决过 / 无可识别的
              工具名 / 重试次数已耗尽），继续走原有分类规则。

        适用场景：子智能体线程已经死掉的路径（例如 ``describe_granted``
        抛出的 ToolPermissionError 直接终止了任务）——挂起升级通道
        （``ToolAuthorization.call`` 内）只覆盖实际执行场景；这条兑底
        路径保证"没走成挂起通道"的缺工具失败同样先由主智能体判断，
        而不是直接弹人工。
        """

        failure = task.failure_report
        if failure is None or failure.failure_type != FailureType.PERMISSION_ERROR:
            return None
        if failure.metadata.get("tool_grant_adjudicated"):
            # 挂起通道已裁决拒绝后线程失败：人类是最后仲裁者，
            # 落回原有规则（PERMISSION_ERROR -> ASK_USER）。
            return None
        if task.attempt >= task.definition.max_attempts:
            return None
        if self.tool_grant_decision_maker is None:
            return None
        requested = extract_requested_tool_names(failure.error_message)
        if not requested:
            return None

        granted_tools: list[str] = []
        ask_user_tools: list[str] = []
        denied_tools: list[str] = []
        for tool_name in requested:
            outcome = self.tool_grant_decision_maker.adjudicate(
                task_id=task.task_id,
                task_type=task.definition.task_type,
                objective=task.definition.objective,
                inputs=task.definition.inputs,
                allowed_tools=task.definition.allowed_tools,
                forbidden_actions=task.definition.forbidden_actions,
                tool_name=tool_name,
                rationale=(
                    "任务失败后主智能体复核权限错误: "
                    + (failure.error_message or "")[:300]
                ),
            )
            self.task_repo.record_event(
                self.job.job_id,
                task.task_id,
                "tool_grant_adjudicated",
                {
                    "attempt_id": task.active_attempt_id,
                    "tool_name": tool_name,
                    "decision": outcome.decision.value,
                    "reason": outcome.reason,
                    "source": outcome.source,
                    "context": "failure_followup",
                },
            )
            if outcome.decision == ToolGrantDecision.GRANT:
                # 写入白名单即可：重试时会经过 ToolAuthorizer 硬校验，
                # 不可能绕过风险预算（裁决器已先验过一次，这里信任
                # 但不重新实现边界检查）。
                if tool_name not in task.definition.allowed_tools:
                    task.definition.allowed_tools.append(tool_name)
                granted_tools.append(tool_name)
            elif outcome.decision == ToolGrantDecision.ASK_USER:
                ask_user_tools.append(tool_name)
            else:
                denied_tools.append(tool_name)

        if granted_tools:
            self.task_repo.save(task)
            return "retry"
        if ask_user_tools:
            return "ask_user"
        if denied_tools:
            # 全部被明确拒绝：打上已裁决标记后落回原有规则，
            # 避免下一轮重试对同一批工具重复裁决。
            failure.metadata["tool_grant_adjudicated"] = True
            failure.metadata["tool_grant_denied_tools"] = denied_tools
            self.task_repo.save(task)
        return None

    @staticmethod
    def _summarize_retry_guidance(task: Task) -> str:
        """Create one bounded instruction before the failure report is cleared."""

        failure = task.failure_report
        if failure is None:
            return "重试注意事项：重新执行前先核对输入和前置条件，不要直接重复上次操作。"

        action = failure.recommended_action.strip()
        if not action and failure.likely_causes:
            action = f"优先排查可能原因：{failure.likely_causes[0]}"
        if not action:
            action = "先定位并修正上次失败原因，再执行对应步骤"

        failed_step = failure.failed_step.strip() or "未知步骤"
        error = " ".join(failure.error_message.split()).strip()
        error = error[:240] if error else failure.failure_type.value
        guidance = (
            f"重试注意事项：上次在“{failed_step}”失败（{error}）；"
            f"本次请{action.rstrip('。')}，避免重复同一错误。"
        )
        redacted, _ = redact_sensitive_text(guidance)
        return redacted[:600]

    def _begin_timeout_termination(self, task: Task) -> None:
        reason = (
            task.failure_report.error_message
            if task.failure_report is not None
            else f"task entered timeout state {task.status.value}"
        )
        self._begin_subagent_termination(task, reason=reason, terminal=False)

    def _begin_subagent_termination(
        self,
        task: Task,
        *,
        reason: str,
        terminal: bool,
        handle=None,
    ) -> None:
        """Use one asynchronous graceful/forced cancellation state machine."""

        handle = handle or self.dispatcher.get_handle(task.task_id)
        if handle is None:
            self.scheduler.lease_manager.release(task)
            if terminal:
                self.scheduler.mark_terminal_failure(task, reason=reason)
            else:
                self.task_repo.save(task)
            return
        if task.task_id in self._timeout_cancellations:
            if terminal:
                request = self._termination_requests.setdefault(task.task_id, {})
                request.update({"terminal": True, "reason": reason})
            return
        handle.request_graceful_cancel()
        self.scheduler.lease_manager.request_cancel(task.task_id)
        self._timeout_cancellations[task.task_id] = time.monotonic()
        self._termination_requests[task.task_id] = {
            "terminal": terminal,
            "reason": reason,
            "record": TerminationRecord(
                task_id=task.task_id,
                mode=TerminationMode.GRACEFUL,
                reason=reason,
            ),
        }
        self.task_repo.record_event(
            self.job.job_id,
            task.task_id,
            "subagent_cancellation_requested",
            {
                "attempt_id": handle.attempt_id,
                "status": task.status.value,
                "terminal": terminal,
                "reason": reason,
            },
        )

    def _settle_timeout_terminations(self) -> None:
        for task_id, requested_at in list(self._timeout_cancellations.items()):
            task = self.scheduler.dag.get(task_id)
            handle = self.dispatcher.get_handle(task_id)
            request = self._termination_requests.get(task_id, {})
            if task is None or handle is None:
                self._timeout_cancellations.pop(task_id, None)
                self._termination_requests.pop(task_id, None)
                continue
            if handle.is_finished():
                # Only after confirmed thread/backend termination may this attempt
                # release its lease and become eligible for retry.
                if request.get("terminal", False):
                    self.scheduler.mark_terminal_failure(
                        task, reason=str(request.get("reason", "reporting exhausted"))
                    )
                else:
                    self.scheduler.mark_failed(
                        task, TaskStatus.FAILED_RETRYABLE, task.failure_report
                    )
                record = request.get("record")
                if isinstance(record, TerminationRecord):
                    record.mode = TerminationMode.GRACEFUL
                    record.completed_at = self._now()
                    self._termination_log.append(record)
                self.dispatcher.discard_handle(task_id, attempt_id=handle.attempt_id)
                self._timeout_cancellations.pop(task_id, None)
                self._termination_requests.pop(task_id, None)
                self._pending_validation.discard(task_id)
                self.task_repo.record_event(
                    self.job.job_id,
                    task_id,
                    "subagent_termination_confirmed",
                    {
                        "attempt_id": handle.attempt_id,
                        "terminal": bool(request.get("terminal", False)),
                        "termination": record.to_dict()
                        if isinstance(record, TerminationRecord)
                        else None,
                    },
                )
                continue
            if time.monotonic() - requested_at < self.config.timeout_cancel_grace_seconds:
                continue

            # Python cannot prove an unresponsive thread was killed. Fail the task
            # terminally instead of releasing it into a concurrent retry.
            handle.force_kill()
            self.scheduler.lease_manager.release(task)
            reason = str(
                request.get(
                    "reason",
                    "execution termination could not be confirmed after timeout",
                )
            )
            self.scheduler.mark_terminal_failure(
                task, reason=reason
            )
            record = request.get("record")
            if isinstance(record, TerminationRecord):
                record.mode = TerminationMode.FORCED
                record.completed_at = self._now()
                self._termination_log.append(record)
            self.dispatcher.discard_handle(task_id, attempt_id=handle.attempt_id)
            self._timeout_cancellations.pop(task_id, None)
            self._termination_requests.pop(task_id, None)
            self._pending_validation.discard(task_id)
            self.task_repo.record_event(
                self.job.job_id,
                task_id,
                "subagent_termination_forced",
                {
                    "attempt_id": handle.attempt_id,
                    "reason": reason,
                    "termination": record.to_dict()
                    if isinstance(record, TerminationRecord)
                    else None,
                },
            )

    def _classify_failure_via_llm(self, task: Task) -> FailureDecision:
        """``Replanner.classify_failure`` 的 LLM 兜底回调（§16 编排 + §19 决策的落地）。

        只有 ``task.failure_report.failure_type == UNKNOWN_ERROR``（确定性
        规则表明确不覆盖的唯一类型）时才会走到这里，具体见
        ``replanner.py`` 顶部"LLM 兜底分支"说明。本方法负责：
            1. 从 ``task_repo`` 取最近事件，连同 job/DAG 一起交给
               ``ContextBuilder`` 编排出九段决策上下文；
            2. 委托 ``MainAgentLLMDecisionMaker`` 调用 LLM 并解析为
               ``FailureDecision``；
            3. 把 LLM 的原始决策理由记录为审计事件，便于事后复盘"这次
               为什么是 LLM 而不是规则做的决定"。
        """

        recent_events = self.task_repo.list_events(self.job.job_id)
        result = self.llm_decision_maker.classify_failure_with_llm(
            job=self.job,
            dag=self.scheduler.dag,
            task=task,
            recent_events=recent_events,
        )
        self.task_repo.record_event(
            self.job.job_id,
            task.task_id,
            "failure_classified_by_llm",
            {
                "decision": result.decision.value,
                "reason": result.reason,
                "fallback_used": result.fallback_used,
            },
        )
        return result.decision

    # ---- 反思闭环（§19 + §11.2）----
    #
    # 完整链路（本节所有方法按调用顺序排列，串联起来就是 §11.2 原文的
    # 完整闭环）：
    #   verification 任务成功 -> _latest_full_experiment_comparisons()
    #       从其 result.json 里的 comparisons 反解析 MetricComparison
    #   -> _check_result_gap() 判定是否超出容差
    #   -> _trigger_reflection() 派发 reflection 任务（真正入队，不再是
    #      空调用）
    #   -> reflection 任务成功 -> _on_task_validated_for_reflection()
    #      构造 ReflectionReport，并立即通过 plan_audit 派发审计任务
    #   -> 审计任务逐个成功 -> _on_task_validated_for_reflection() 从其
    #      输出推断 AuditFinding，追加到对应 ReflectionReport
    #   -> 该轮审计任务全部完成 -> _advance_reflection_pipeline() 调用
    #      summarize_audit_findings() 汇总出整体 audit_result
    #       -> 确认存在具体错误 -> plan_repair -> repair_completed ->
    #          plan_minimum_rerun_scope（唯一允许重跑的分支）
    #       -> 未确认任何错误（NO_OBVIOUS_ERROR_FOUND /
    #          RANDOMNESS_LIKELY / UNDISCLOSED_DETAIL_LIKELY）->
    #          不重跑，直接推进 Job 到 VERIFIED_REPRODUCTION_GAP
    #          终态（§11.8：流程本身没有问题就不要为了硬凑论文数字
    #          而反复重跑，浪费 GPU/模型调用预算）。

    def _check_result_gap(self):
        if self.job.status in {
            JobStatus.REFLECTION_REQUIRED,
            JobStatus.REFLECTION_PLANNING,
            JobStatus.AUDIT_RUNNING,
            JobStatus.ISSUE_FOUND,
            JobStatus.REPAIR_RUNNING,
            JobStatus.RERUN_REQUIRED,
            JobStatus.NO_ISSUE_FOUND,
            JobStatus.VERIFIED_REPRODUCTION_GAP,
        }:
            return None
        latest_run = self._latest_full_experiment_comparisons()
        if latest_run is None:
            return None
        comparisons = latest_run
        return self.reflection_controller.result_gap_detected(comparisons, self.job)

    def _latest_full_experiment_comparisons(self) -> Optional[list[MetricComparison]]:
        """从最近一次成功的 ``verification`` 任务输出中取出指标对比结果。

        ``ResultVerificationAgent``（agents/verification/agent.py）成功
        执行后会在 ``result.json`` 里写出 ``comparisons`` 列表（每项含
        metric/paper_value/reproduced_value/tolerance_type/tolerance/
        within_tolerance，字段名与 ``MetricComparison.to_dict()`` 一一
        对应），这里只需要找到 DAG 中最新完成的一个 verification 任务，
        读取其 output 并反解析回 ``MetricComparison`` 领域对象。

        找不到任何已成功的 verification 任务（尚未跑到这一步）时返回
        ``None``，交给 ``_check_result_gap`` 判定为"暂不触发反思"。
        """

        candidates = [
            t
            for t in self.scheduler.dag.all_tasks()
            if t.definition.task_type == "verification" and t.status == TaskStatus.SUCCEEDED
        ]
        if not candidates:
            return None
        # completed_at 为 None 的理论上不会出现在 SUCCEEDED 里
        # （mark_succeeded 总是先写 completed_at），但仍兜底用
        # updated_at 排序，避免因为极端情况下的 None 排序异常。
        latest = max(candidates, key=lambda t: t.completed_at or t.updated_at)

        payload = self._read_task_result_json(latest)
        if payload is None:
            return None
        raw_comparisons = payload.get("comparisons")
        if not raw_comparisons:
            return None

        comparisons: list[MetricComparison] = []
        for item in raw_comparisons:
            try:
                comparisons.append(
                    MetricComparison(
                        metric=item["metric"],
                        paper_value=float(item["paper_value"]),
                        reproduced_value=float(item["reproduced_value"]),
                        tolerance_type=ToleranceType(item.get("tolerance_type", ToleranceType.ABSOLUTE.value)),
                        tolerance=float(item.get("tolerance", 0.0)),
                        within_tolerance=bool(item.get("within_tolerance", False)),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "task %s produced malformed metric comparison entry %r: %s",
                    latest.task_id,
                    item,
                    exc,
                )
        return comparisons or None

    def _read_task_result_json(self, task: Task) -> Optional[dict[str, Any]]:
        """读取任务 ``outputs`` 中登记的 ``result.json`` 并解析为字典。

        ``task.outputs`` 由 ``OutputValidator.validate`` 通过
        ``sandbox.collect_outputs()`` 填充，键是相对 ``output/`` 目录的
        路径，值是宿主机上的**绝对路径**（不是文件内容本身，见
        ``sandbox/workspace.py::collect_outputs`` 的文档字符串），因此
        这里需要真正打开文件读取，而不能直接把 ``outputs`` 当 JSON 用。
        """

        result_path = task.outputs.get("result.json")
        if not result_path:
            return None
        try:
            return TaskResultEnvelope.from_file(
                result_path,
                expected_task_id=task.task_id,
                expected_attempt_id=task.active_attempt_id,
                expected_task_type=task.definition.task_type,
            ).payload
        except (OSError, json.JSONDecodeError, ResultValidationError) as exc:
            logger.warning("failed to read result.json for task %s: %s", task.task_id, exc)
            return None

    def _trigger_reflection(self, reason: str) -> None:
        """§19 ``reflection_agent.analyze()`` 的派发入口：真正创建
        ``reflection`` 任务并加入调度器，而不只是切换 Job 状态。

        触发指标（``trigger_metrics``）通过 inputs 传给
        ``ReflectionAgent``（见 agents/reflection/agent.py::run），
        ``available_evidence`` 目前提供任务 DAG 摘要作为最小可用证据，
        更丰富的证据（日志片段、ExperimentRun 记录等）留待
        evidence/ 模块进一步补充时再扩展这里的组装逻辑。
        """

        exhausted, budget_reason = self.job.budget_exhausted()
        if exhausted:
            logger.warning(
                "job %s budget exhausted (%s), skip triggering reflection despite gap: %s",
                self.job.job_id,
                budget_reason,
                reason,
            )
            return
        if self.job.reflection_round >= self.job.budget.max_reflection_rounds:
            logger.warning("job %s reflection round budget exhausted, skipping", self.job.job_id)
            return

        comparisons = self._latest_full_experiment_comparisons() or []
        trigger_metrics = [c.to_dict() for c in comparisons if not c.within_tolerance]

        self.job.reflection_round += 1
        self.job.status = JobStatus.REFLECTION_REQUIRED
        self.job_repo.save(self.job)
        logger.info("job %s entering reflection round %d: %s", self.job.job_id, self.job.reflection_round, reason)

        definition = build_task_definition(
            objective=f"反思第 {self.job.reflection_round} 轮: {reason}",
            task_type="reflection",
            inputs={
                "trigger_metrics": trigger_metrics,
                "available_evidence": self._reflection_evidence_bundle(),
                "reflection_round": self.job.reflection_round,
                "creation_key": f"reflection:{self.job.reflection_round}",
            },
            # ReflectionAgent 完全基于 inputs 中已提供的证据做纯文本
            # 推理（call_llm 显式声明 tool_names=[]），不调用任何
            # call_tool，因此不需要授予 read_file。
            restrict_tools=[],
        )
        reflection_task = Task(job_id=self.job.job_id, definition=definition)
        reflection_task = self.scheduler.add_tasks([reflection_task])[0]
        self.task_repo.record_event(
            self.job.job_id,
            reflection_task.task_id,
            "reflection_triggered",
            {"reason": reason, "reflection_round": self.job.reflection_round},
            event_key=f"reflection-triggered:{self.job.reflection_round}",
        )

    def _reflection_evidence_bundle(self) -> dict[str, Any]:
        """Collect bounded, redacted facts that can explain a metric gap."""

        tasks = list(self.scheduler.dag.all_tasks())

        def latest(task_type: str, *, tier: str = "") -> Task | None:
            candidates = [
                task
                for task in tasks
                if task.definition.task_type == task_type
                and task.status == TaskStatus.SUCCEEDED
                and (not tier or task.definition.inputs.get("tier") == tier)
            ]
            return candidates[-1] if candidates else None

        full = latest(
            "experiment_execution", tier=ExperimentTier.FULL_EXPERIMENT.value
        )
        verification = latest("verification")
        evidence: dict[str, Any] = {
            "full_experiment": self._redacted_payload_excerpt(full, 2600),
            "verification": self._redacted_payload_excerpt(verification, 1600),
            "specification": self._redacted_payload_excerpt(
                latest("specification"), 900
            ),
            "resource_check": self._redacted_payload_excerpt(
                latest("resource_check"), 700
            ),
            "code_analysis": self._redacted_payload_excerpt(
                latest("code_analysis"), 900
            ),
            "dag_summary": self.scheduler.summary(),
        }
        return evidence

    def _redacted_payload_excerpt(self, task: Task | None, limit: int) -> str:
        if task is None:
            return ""
        payload = self._read_task_result_json(task) or {}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        redacted, _ = redact_sensitive_text(encoded)
        return redacted[:limit]

    def _build_reflection_audit_context(self) -> dict[str, Any]:
        tasks = list(self.scheduler.dag.all_tasks())
        upstream: dict[str, list[str]] = {}
        for task_type in (
            "paper_analysis",
            "code_analysis",
            "specification",
            "resource_check",
        ):
            upstream[task_type] = [
                task.task_id
                for task in tasks
                if task.definition.task_type == task_type
                and task.status == TaskStatus.SUCCEEDED
                and not task.definition.inputs.get("reflection_id")
            ]
        successful_runs = [
            task
            for task in tasks
            if task.definition.task_type == "experiment_execution"
            and task.status == TaskStatus.SUCCEEDED
        ]
        repository_path = (
            str(successful_runs[-1].definition.inputs.get("repository_path", ""))
            if successful_runs
            else ""
        ) or self.job.inputs.repository_path
        return {
            "paper_path": self.job.inputs.paper_path,
            "appendix_paths": [
                *self.job.inputs.appendix_paths,
                *self.job.inputs.supplementary_paths,
            ],
            "repository_path": repository_path,
            "dataset_paths": list(self.job.inputs.dataset_paths),
            "model_paths": list(self.job.inputs.model_paths),
            "checkpoint_paths": list(self.job.inputs.checkpoint_paths),
            "target_experiments": list(self.job.inputs.target_experiments),
            "experiment_id": (
                self.job.inputs.target_experiments or ["main_experiment"]
            )[0],
            "user_overrides": dict(self.job.inputs.experiment_runtime_config),
            "requested_gpu_count": self.job.inputs.gpu_count,
            "requested_gpu_memory_gb": self.job.inputs.gpu_memory_gb,
            "requested_disk_mb": self.job.inputs.disk_mb,
            "upstream_task_ids": upstream,
        }

    def _on_task_validated_for_reflection(self, task: Task) -> None:
        """任务通过独立校验后的反思闭环钩子：分流到 reflection / 审计两类。"""

        if task.definition.task_type == "reflection":
            self._on_reflection_task_succeeded(task)
        elif task.definition.inputs.get("reflection_id"):
            # plan_audit/plan_repair 派发的任务都会在 inputs 里带上
            # reflection_id，用它反查归属的 ReflectionReport，而不是
            # 用 task_type 白名单——修复任务的 task_type 可能是
            # coding/specification/resource_check/environment_build
            # 中的任意一种（见 reflection_controller.py::plan_repair），
            # 用固定白名单去猜测容易漏判。
            if task.definition.inputs.get("audit_hypothesis_id"):
                self._on_audit_task_succeeded(task)

    def _on_reflection_task_succeeded(self, task: Task) -> None:
        """``ReflectionAgent`` 完成分析后：构造 ReflectionReport 并立即派发审计任务。"""

        payload = self._read_task_result_json(task)
        if payload is None:
            logger.warning("reflection task %s succeeded but result.json is unreadable", task.task_id)
            return

        reflection_round = int(
            task.definition.inputs.get("reflection_round", self.job.reflection_round)
        )
        # The callback can be replayed during crash recovery.  Prefer an old
        # report for the same round (migration compatibility); otherwise use a
        # task-derived id so an interrupted callback is naturally idempotent.
        report = next(
            (item for item in self._reflection_reports if item.round == reflection_round),
            None,
        )
        if report is not None and report.audit_result is not None:
            return

        hypotheses = [
            ReflectionHypothesis(
                hypothesis_id=f"{task.task_id}.{idx}",
                category=h.get("category", "B"),
                description=h.get("description", ""),
                confidence=float(h.get("confidence", 0.5)),
                required_checks=list(h.get("required_checks", [])),
                priority=int(h.get("priority", 0)),
            )
            for idx, h in enumerate(payload.get("hypotheses", []))
        ]

        trigger_metric_dicts = task.definition.inputs.get("trigger_metrics", [])
        trigger_metrics = [
            MetricComparison(
                metric=m["metric"],
                paper_value=float(m["paper_value"]),
                reproduced_value=float(m["reproduced_value"]),
                tolerance_type=ToleranceType(m.get("tolerance_type", ToleranceType.ABSOLUTE.value)),
                tolerance=float(m.get("tolerance", 0.0)),
                within_tolerance=bool(m.get("within_tolerance", False)),
            )
            for m in trigger_metric_dicts
        ]
        if report is None:
            report = ReflectionReport(
                job_id=self.job.job_id,
                round=reflection_round,
                trigger_metrics=trigger_metrics,
                run_id=(
                    latest.run_id
                    if (
                        latest := self.experiment_run_repo.latest_full_run(
                            self.job.job_id,
                            (self.job.inputs.target_experiments or ["main_experiment"])[0],
                        )
                    )
                    else ""
                ),
                likely_source=payload.get("likely_source", "unknown"),
                reflection_id=f"reflection_{task.task_id}",
                hypotheses=hypotheses,
                audit_context=self._build_reflection_audit_context(),
            )
            self._reflection_reports.append(report)
            self.reflection_repo.save(report)
        self.task_repo.record_event(
            self.job.job_id,
            task.task_id,
            "reflection_report_created",
            {"reflection_id": report.reflection_id, "hypothesis_count": len(hypotheses)},
            event_key=f"reflection-report-created:{task.task_id}",
        )

        audit_tasks = self.reflection_controller.plan_audit(report)
        if not audit_tasks:
            # 反思智能体没有给出任何审计建议：视为"无更多可检查的线索"，
            # 直接按无问题处理，避免这一轮反思无限期挂起。
            report.audit_result = AuditResultType.NO_OBVIOUS_ERROR_FOUND
            report.confirmed_issue = "反思智能体未提出任何可执行的审计检查项"
            self.reflection_repo.save(report)
            self.job.status = JobStatus.VERIFIED_REPRODUCTION_GAP
            self.job.final_reproduction_status = ReproductionStatus.VERIFIED_REPRODUCTION_GAP
            self.job_repo.save(self.job)
            return

        # 把这些审计任务 inputs 里补上 reflection_id 之外，还需要能
        # 反查回同一个 report——task_factory.build_task_definition 生成
        # 的 inputs 已经在 plan_audit 里写好了 reflection_id，这里只需
        # 记录"这一轮还有哪些审计任务在跑"，供后续判断"是否全部完成"。
        audit_tasks = self.scheduler.add_tasks(audit_tasks)
        report.audit_task_ids = [t.task_id for t in audit_tasks]
        self.reflection_repo.save(report)
        self._pending_audit_task_ids[report.reflection_id] = {
            t.task_id for t in audit_tasks if t.status != TaskStatus.SUCCEEDED
        }
        self.task_repo.record_event(
            self.job.job_id,
            task.task_id,
            "audit_tasks_dispatched",
            {"reflection_id": report.reflection_id, "audit_task_ids": [t.task_id for t in audit_tasks]},
            event_key=f"audit-tasks-dispatched:{report.reflection_id}",
        )

    def _on_audit_task_succeeded(self, task: Task) -> None:
        """单个审计任务完成后：推断 ``AuditFinding`` 并登记到对应 ReflectionReport。

        审计任务复用现有的 paper_analysis/code_analysis/specification/
        resource_check 四类子智能体（见 reflection_controller.py::
        plan_audit 的维度映射），它们各自已有成熟的领域输出 schema，
        本方法不要求它们额外输出专门的"审计结论"字段，而是从每类
        agent 已经产出的确定性信号里推断是否发现问题——这样完全不需要
        改动这些子智能体的 Prompt/解析逻辑，也不会让"审计"这个临时
        身份污染它们本职的领域职责。
        """

        reflection_id = task.definition.inputs.get("reflection_id")
        report = self._find_reflection_report(reflection_id)
        if report is None:
            logger.warning("audit task %s has unknown reflection_id=%s", task.task_id, reflection_id)
            return

        payload = self._read_task_result_json(task) or {}
        if not any(item.audit_task_id == task.task_id for item in report.audit_findings):
            finding = self._infer_audit_finding(task, payload)
            report.audit_findings.append(finding)
            self.reflection_repo.save(report)
            self.task_repo.record_event(
                self.job.job_id,
                task.task_id,
                "audit_finding_recorded",
                finding.to_dict(),
                event_key=f"audit-finding-recorded:{task.task_id}",
            )

        pending = self._pending_audit_task_ids.get(reflection_id)
        if pending is not None:
            pending.discard(task.task_id)

    def _infer_audit_finding(self, task: Task, payload: dict[str, Any]) -> AuditFinding:
        """规则优先的确定性审计结论推断（不引入额外 LLM 调用）。

        各任务类型的判据直接取自该类型子智能体已有的、本来就会产出的
        字段：
            - resource_check -> blocking_issues 非空即
              RESOURCE_LIMITATION_CONFIRMED；
            - specification  -> unresolved_conflicts 非空即
              CONFIG_ERROR_CONFIRMED（未解决的字段冲突就是配置问题）；
            - code_analysis  -> 未能定位到任何 entry_points/
              matched_run_scripts 视为 CODE_ERROR_CONFIRMED（代码路径
              没对上，符合该维度"代码路径是否正确"的检查目的）；
            - paper_analysis -> 本身没有"是否有问题"的直接信号，任务
              成功执行即视为论文理解没有发现新的矛盾，归入
              UNDISCLOSED_DETAIL_LIKELY 让主智能体知道"复核过了，但
              没排除论文遗漏细节的可能"，而不是武断地宣布无问题。
        找不到匹配任务类型时，统一按"任务成功但无法进一步判定"处理，
        归入 NO_OBVIOUS_ERROR_FOUND，保持保守。
        """

        dimension = task.definition.inputs.get("audit_check_dimension", "")
        task_type = task.definition.task_type

        if task_type == "resource_check":
            blocking = payload.get("blocking_issues") or []
            if blocking:
                return AuditFinding(
                    audit_task_id=task.task_id,
                    check_dimension=dimension or "D",
                    result=AuditResultType.RESOURCE_LIMITATION_CONFIRMED,
                    detail=f"资源检查发现阻塞性问题: {blocking}",
                    evidence_refs=[f"task:{task.task_id}"],
                )
            return AuditFinding(
                audit_task_id=task.task_id,
                check_dimension=dimension or "D",
                result=AuditResultType.NO_OBVIOUS_ERROR_FOUND,
                detail="资源检查未发现阻塞性问题",
                evidence_refs=[f"task:{task.task_id}"],
            )

        if task_type == "specification":
            unresolved = payload.get("unresolved_conflicts") or []
            if unresolved:
                return AuditFinding(
                    audit_task_id=task.task_id,
                    check_dimension=dimension or "C",
                    result=AuditResultType.CONFIG_ERROR_CONFIRMED,
                    detail=f"实验规格存在未解决的字段冲突: {unresolved}",
                    evidence_refs=[f"task:{task.task_id}"],
                )
            return AuditFinding(
                audit_task_id=task.task_id,
                check_dimension=dimension or "C",
                result=AuditResultType.NO_OBVIOUS_ERROR_FOUND,
                detail="实验规格字段均无未解决冲突",
                evidence_refs=[f"task:{task.task_id}"],
            )

        if task_type == "code_analysis":
            entry_points = payload.get("entry_points") or []
            matched_scripts = payload.get("matched_run_scripts") or {}
            if not entry_points and not matched_scripts:
                return AuditFinding(
                    audit_task_id=task.task_id,
                    check_dimension=dimension or "B",
                    result=AuditResultType.CODE_ERROR_CONFIRMED,
                    detail="复核代码路径未能定位到入口文件或对应实验的运行脚本",
                    evidence_refs=[f"task:{task.task_id}"],
                )
            return AuditFinding(
                audit_task_id=task.task_id,
                check_dimension=dimension or "B",
                result=AuditResultType.NO_OBVIOUS_ERROR_FOUND,
                detail="代码路径复核未发现明显问题",
                evidence_refs=[f"task:{task.task_id}"],
            )

        if task_type == "paper_analysis":
            return AuditFinding(
                audit_task_id=task.task_id,
                check_dimension=dimension or "A",
                result=AuditResultType.UNDISCLOSED_DETAIL_LIKELY,
                detail="已复核论文理解，未发现新的矛盾，但不能排除论文未披露的实现细节",
                evidence_refs=[f"task:{task.task_id}"],
            )

        return AuditFinding(
            audit_task_id=task.task_id,
            check_dimension=dimension or "E",
            result=AuditResultType.NO_OBVIOUS_ERROR_FOUND,
            detail=f"审计任务 {task_type} 已完成，未给出可判定的异常信号",
            evidence_refs=[f"task:{task.task_id}"],
        )

    def _find_reflection_report(self, reflection_id: Optional[str]) -> Optional[ReflectionReport]:
        if not reflection_id:
            return None
        for report in self._reflection_reports:
            if report.reflection_id == reflection_id:
                return report
        return None

    def _advance_reflection_pipeline(self) -> None:
        """§11.2 反思闭环的后续阶段：汇总审计 → (确认问题 ? 修复 → 最小重跑 : 终局)。

        每一轮反思只会经过下面两个分支之一：
            1. 汇总结果确认存在具体错误 -> plan_repair -> 修复完成后
               plan_minimum_rerun_scope（唯一允许重跑的路径，且始终是
               "最小范围"重跑而不是无脑重跑全部正式实验）；
            2. 汇总结果为"无明显问题"（或论文未披露细节/随机性）->
               不重跑，直接把 Job 推进到 VERIFIED_REPRODUCTION_GAP
               终态——这正是用户要求的"审查确定流程没问题就不要为了
               对齐论文而重跑"的落地点。
        """

        for report in list(self._reflection_reports):
            if report.audit_result is None:
                pending = self._pending_audit_task_ids.get(report.reflection_id)
                if pending is None or pending:
                    continue  # 该轮审计任务还没派发完 / 还有任务未完成

                audit_result, detail = self.reflection_controller.summarize_audit_findings(
                    report.audit_findings
                )
                report.audit_result = audit_result
                report.confirmed_issue = detail
                self.reflection_repo.save(report)
                self.task_repo.record_event(
                    self.job.job_id,
                    report.reflection_id,
                    "audit_findings_summarized",
                    {"audit_result": audit_result.value, "detail": detail},
                    event_key=f"audit-findings-summarized:{report.reflection_id}",
                )
            else:
                audit_result = report.audit_result

            if self.reflection_controller.audit_issue_confirmed(report):
                if report.repair_task_ids:
                    continue
                self.job.status = JobStatus.ISSUE_FOUND
                self.job_repo.save(self.job)
                repair_tasks = self.reflection_controller.plan_repair(
                    report,
                    repository_path=self.job.inputs.repository_path,
                    base_image=self.config.execution_image,
                    dataset_paths=self.job.inputs.dataset_paths,
                    model_paths=self.job.inputs.model_paths,
                    checkpoint_paths=self.job.inputs.checkpoint_paths,
                )
                self._bind_repair_dependencies(repair_tasks)
                repair_tasks = self.scheduler.add_tasks(repair_tasks)
                report.repair_task_ids = [t.task_id for t in repair_tasks]
                self.reflection_repo.save(report)
                self.job.status = JobStatus.REPAIR_RUNNING
                self.job_repo.save(self.job)

            else:
                if self.job.status == JobStatus.VERIFIED_REPRODUCTION_GAP:
                    continue
                # §11.8 明确要求：流程/配置/代码/数据均未发现明显问题时，
                # 不允许仅仅为了对齐论文数字而重跑，直接进入终态，
                # 交给最终报告如实呈现"经审计确认的真实复现差距"。
                logger.info(
                    "job %s reflection %s: no issue confirmed (%s), skip rerun and report gap directly",
                    self.job.job_id,
                    report.reflection_id,
                    audit_result.value,
                )
                self.job.status = JobStatus.NO_ISSUE_FOUND
                self.job_repo.save(self.job)
                self.job.status = JobStatus.VERIFIED_REPRODUCTION_GAP
                self.job.final_reproduction_status = ReproductionStatus.VERIFIED_REPRODUCTION_GAP
                self.job_repo.save(self.job)

        self._advance_repair_completion()

    def _bind_repair_dependencies(self, repair_tasks: list[Task]) -> None:
        """Give deterministic repair agents the validated artifacts they need."""

        existing_tasks = list(self.scheduler.dag.all_tasks())
        for repair in repair_tasks:
            if repair.definition.task_type != "specification":
                continue
            dependencies: list[str] = []
            for task_type in ("paper_analysis", "code_analysis", "resource_check"):
                if task_type == "paper_analysis":
                    # 论文分析拆为正文 + 附录（可能多片）并行任务后，
                    # 修复用的规格任务必须绑定全部 paper 依赖，
                    # ArtifactResolver 才能把它们合并成完整 paper_findings。
                    dependencies.extend(
                        task.task_id
                        for task in existing_tasks
                        if task.definition.task_type == "paper_analysis"
                        and task.status == TaskStatus.SUCCEEDED
                        and not task.definition.inputs.get("reflection_id")
                    )
                    continue
                matched = next(
                    (
                        candidate
                        for candidate in reversed(existing_tasks)
                        if candidate.definition.task_type == task_type
                        and candidate.status == TaskStatus.SUCCEEDED
                        and not candidate.definition.inputs.get("reflection_id")
                    ),
                    None,
                )
                if matched is not None:
                    dependencies.append(matched.task_id)
            repair.definition.dependencies = list(dict.fromkeys(dependencies))

    def _advance_repair_completion(self) -> None:
        """修复任务全部完成后，触发最小范围重跑（唯一允许重跑的入口）。"""

        for report in list(self._reflection_reports):
            if report.audit_result is None or not self.reflection_controller.audit_issue_confirmed(report):
                continue
            if report.rerun_triggered:
                existing_reruns = [
                    task
                    for task in self.scheduler.dag.all_tasks()
                    if task.definition.inputs.get("reflection_id") == report.reflection_id
                    and str(task.definition.inputs.get("creation_key", "")).startswith(
                        "rerun:"
                    )
                ]
                if existing_reruns and self.job.status in {
                    JobStatus.ISSUE_FOUND,
                    JobStatus.REPAIR_RUNNING,
                }:
                    self.job.full_experiment_rerun_count = sum(
                        1
                        for candidate in self.scheduler.dag.all_tasks()
                        if str(candidate.definition.inputs.get("creation_key", "")).startswith(
                            "rerun:"
                        )
                    )
                    self.job.status = JobStatus.RERUN_REQUIRED
                    self.job_repo.save(self.job)
                continue
            if not report.repair_task_ids:
                continue

            repair_tasks = [
                t for tid in report.repair_task_ids if (t := self.scheduler.dag.get(tid)) is not None
            ]
            if len(repair_tasks) != len(report.repair_task_ids):
                continue  # 有修复任务还没落到 DAG（理论上不会发生），保守等待
            if not self.reflection_controller.repair_completed(repair_tasks):
                continue

            rerun_tasks = self.reflection_controller.plan_minimum_rerun_scope(
                report,
                self.job,
                runs=self.experiment_run_repo.list_by_job(self.job.job_id),
                execution_image=self._latest_execution_image(),
                repository_path=self._repository_after_repairs(report),
                execution_manifest=self._latest_execution_manifest(),
            )
            if rerun_tasks:
                rerun_tasks = self.scheduler.add_tasks(rerun_tasks)
                # Derive the counter from durable logical rerun tasks.  This is
                # safe whether the process died before or after saving the job.
                self.job.full_experiment_rerun_count = sum(
                    1
                    for candidate in self.scheduler.dag.all_tasks()
                    if str(candidate.definition.inputs.get("creation_key", "")).startswith(
                        "rerun:"
                    )
                )
                self.job.status = JobStatus.RERUN_REQUIRED
            else:
                self.job.status = JobStatus.BLOCKED_BY_MISSING_RESOURCE
                self.job.final_reproduction_status = ReproductionStatus.BLOCKED_BY_MISSING_RESOURCE
            report.rerun_triggered = True
            self.reflection_repo.save(report)
            self.job_repo.save(self.job)
            self.task_repo.record_event(
                self.job.job_id,
                report.reflection_id,
                "minimum_rerun_scope_dispatched",
                {"rerun_task_ids": [t.task_id for t in rerun_tasks]},
                event_key=f"minimum-rerun-dispatched:{report.reflection_id}",
            )

    def _repository_after_repairs(self, report: ReflectionReport) -> str:
        for task_id in reversed(report.repair_task_ids):
            task = self.scheduler.dag.get(task_id)
            if task is None or task.definition.task_type != "coding":
                continue
            sandbox = self.sandbox_manager.get(task_id)
            if sandbox is not None:
                repository = sandbox.workspace_dir / "repository"
                if repository.is_dir():
                    return str(repository)
            # SandboxManager intentionally keeps only current-process objects.
            # After a controller restart, recover the validated attempt-scoped
            # repository from its durable directory instead of falling back to
            # the user's original (unrepaired) source tree.
            repository = (
                self.sandbox_manager.sandbox_root
                / f"task_{task_id}"
                / task.active_attempt_id
                / "workspace"
                / "repository"
            )
            if repository.is_dir():
                return str(repository.resolve())
        return self.job.inputs.repository_path

    def _latest_execution_image(self) -> str:
        candidates = [
            task
            for task in self.scheduler.dag.all_tasks()
            if task.definition.task_type == "environment_build"
            and task.status == TaskStatus.SUCCEEDED
        ]
        for task in reversed(candidates):
            payload = self._read_task_result_json(task) or {}
            environment_ref = payload.get("environment_ref") or payload.get("image_ref")
            if environment_ref:
                return str(environment_ref)
        return self.config.execution_image

    def _latest_execution_manifest(self) -> dict[str, Any]:
        spec_tasks = [
            task
            for task in self.scheduler.dag.all_tasks()
            if task.definition.task_type == "specification"
            and task.status == TaskStatus.SUCCEEDED
        ]
        spec = self._read_task_result_json(spec_tasks[-1]) if spec_tasks else {}
        fields = (spec or {}).get("fields", {}) or {}

        def value(*names: str):
            for name in names:
                item = fields.get(name)
                if isinstance(item, dict) and "value" in item:
                    return item["value"]
            return None

        latest_runs = self.experiment_run_repo.list_by_job(self.job.job_id)
        parent = latest_runs[-1] if latest_runs else None
        seed = value("seed", "random_seed", "random seed")
        try:
            seed = int(seed) if seed is not None else (parent.seed if parent else None)
        except (TypeError, ValueError):
            seed = parent.seed if parent else None
        return {
            "config_digest": (spec or {}).get("spec_digest", "")
            or (parent.config_digest if parent else ""),
            "model_identifier": str(
                value("model_identifier", "model_name", "model")
                or (parent.model_identifier if parent else "")
            ),
            "seed": seed,
            "hardware_identifier": parent.hardware_identifier if parent else "",
        }

    # ---- 任务运行 ----

    def _run_dispatched_task(self, task: Task) -> None:
        """Compatibility wrapper around the main-agent creation tool.

        The normal scheduler path invokes ``create_subagents`` once with the
        whole batch.  Tests and older callers may still launch one dispatched
        task through this method; it now uses the same tool boundary.
        """

        self.create_subagents_tool.call(task_ids=[task.task_id])

    def _persist_experiment_run(self, task: Task) -> None:
        """Persist a validated experiment result before phase advancement."""

        if task.definition.task_type != "experiment_execution":
            return
        payload = self._read_task_result_json(task)
        if not payload:
            return
        self.runtime_accounting.persist_experiment_run(task, payload)

    def _persist_verification_record(self, task: Task) -> None:
        if task.definition.task_type != "verification":
            return
        payload = self._read_task_result_json(task)
        if payload is None:
            return
        comparisons: list[MetricComparison] = []
        for item in payload.get("comparisons", []):
            try:
                comparisons.append(
                    MetricComparison(
                        metric=item["metric"],
                        paper_value=float(item["paper_value"]),
                        reproduced_value=float(item["reproduced_value"]),
                        tolerance_type=ToleranceType(item["tolerance_type"]),
                        tolerance=float(item["tolerance"]),
                        within_tolerance=bool(item["within_tolerance"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        experiment_id = (self.job.inputs.target_experiments or ["main_experiment"])[0]
        latest_full_run = self.experiment_run_repo.latest_full_run(
            self.job.job_id, experiment_id
        )
        self.verification_repo.save(
            VerificationRecord(
                job_id=self.job.job_id,
                task_id=task.task_id,
                run_id=latest_full_run.run_id if latest_full_run is not None else "",
                expected_metric_names=list(payload.get("expected_metric_names", [])),
                observed_metric_names=list(payload.get("observed_metric_names", [])),
                missing_metrics=list(payload.get("missing_metrics", [])),
                comparisons=comparisons,
                run_actually_executed=bool(payload.get("run_actually_executed", False)),
                provenance_verified=bool(payload.get("provenance_verified", False)),
                anti_cheat_passed=bool(payload.get("anti_cheat_passed", False)),
                is_fully_traceable=bool(payload.get("is_fully_traceable", False)),
                mock=bool(payload.get("mock", False)),
                verification_valid=bool(payload.get("verification_valid", False)),
            )
        )

    def _advance_phases(self) -> bool:
        decision = self.phase_coordinator.advance(
            self.job,
            list(self.scheduler.dag.all_tasks()),
            self.experiment_run_repo.list_by_job(self.job.job_id),
        )
        if (
            decision.job_status == JobStatus.BLOCKED_BY_MISSING_RESOURCE
            and decision.reason == "resource check reported blocking issues"
        ):
            resource_tasks = [
                task
                for task in self.scheduler.dag.all_tasks()
                if task.definition.task_type == "resource_check"
                and task.status == TaskStatus.SUCCEEDED
            ]
            resource_task = resource_tasks[-1] if resource_tasks else None
            payload = self._read_task_result_json(resource_task) if resource_task else None
            if resource_task is not None and payload and payload.get("blocking_issues"):
                self.intervention_service.create_for_missing_resources(
                    self.job,
                    resource_task,
                    list(payload["blocking_issues"]),
                    missing_required_resources=list(
                        payload.get("missing_required_resources") or []
                    ),
                    timeout_seconds=self.config.intervention_timeout_seconds,
                )
                self.scheduler.dag.replace_task(resource_task)
                return True
        if (
            decision.job_status == JobStatus.BLOCKED_BY_MISSING_RESOURCE
            and decision.reason == "experiment specification has unresolved conflicts"
        ):
            spec_tasks = [
                task
                for task in self.scheduler.dag.all_tasks()
                if task.definition.task_type == "specification"
                and task.status == TaskStatus.SUCCEEDED
            ]
            spec_task = spec_tasks[-1] if spec_tasks else None
            if spec_task is not None and self._request_spec_approval_if_needed(spec_task):
                return True
        if (
            decision.job_status == JobStatus.BLOCKED_BY_MISSING_RESOURCE
            and decision.reason == "no executable experiment command was discovered or supplied"
        ):
            self.intervention_service.create_for_missing_command(
                self.job,
                timeout_seconds=self.config.intervention_timeout_seconds,
            )
            return True
        if decision.job_status is not None:
            self.job.status = decision.job_status
        if decision.reproduction_status is not None:
            self.job.final_reproduction_status = decision.reproduction_status
        if decision.tasks_to_create:
            materialized_tasks = self.scheduler.add_tasks(decision.tasks_to_create)
            for created in materialized_tasks:
                self.task_repo.record_event(
                    self.job.job_id,
                    created.task_id,
                    "phase_task_created",
                    {"creation_key": created.definition.inputs.get("creation_key", "")},
                    event_key=f"phase-task-created:{created.task_id}",
                )
        if decision.reason:
            self.task_repo.record_event(
                self.job.job_id,
                None,
                "phase_decision",
                {"status": self.job.status.value, "reason": decision.reason},
            )
        return self._job_waiting_for_user()

    # ---- 候选记忆转正（§15.2）----

    def _promote_candidate_memory(self, task: Task) -> None:
        sandbox = self.sandbox_manager.get(task.task_id)
        if sandbox is None:
            return
        candidate_path = sandbox.output_dir / "candidate_memory.md"
        if not candidate_path.exists():
            return
        content = candidate_path.read_text(encoding="utf-8")
        candidate = CandidateMemory(
            task_id=task.task_id,
            topic=f"{task.definition.task_type}.{task.task_id}",
            summary=content[:500],
            details={"task_type": task.definition.task_type},
        )
        result = self.memory_manager.promote_candidate(candidate)
        if not result.accepted:
            logger.warning(
                "candidate memory from task %s rejected: %s", task.task_id, result.rejection_reason
            )

    # ---- 动态工具孵化与衰减 ----

    def _process_dynamic_tool_lifecycle(self, task: Task) -> None:
        """Consume proposals only after the task result has been validated."""

        if not self.config.enable_dynamic_tool_growth:
            return
        sandbox = self.sandbox_manager.get(task.task_id)
        if sandbox is None:
            return
        outcome = self.dynamic_tool_lifecycle.ingest_sidecar(
            sandbox.output_dir / "reusable_code_candidates.json",
            job_id=self.job.job_id,
            task_id=task.task_id,
            attempt_id=task.active_attempt_id,
            task_type=task.definition.task_type,
            sandbox_ctx=sandbox,
        )
        invocations = self.tool_invocation_repo.list_by_task(task.task_id)
        successful_dynamic_names = {
            str(item.get("tool_name"))
            for item in invocations
            if item.get("attempt_id") == task.active_attempt_id
            and item.get("succeeded")
            and str(item.get("tool_name", "")).startswith("dynamic_")
        }
        expired = self.dynamic_tool_lifecycle.advance_relevant_task_event(
            task.definition.task_type,
            refreshed_tool_names=outcome.refreshed_pending_tool_names,
            successful_tool_names=successful_dynamic_names,
        )
        for tool_id in outcome.accepted_tool_ids:
            record = self.dynamic_tool_repo.get(tool_id)
            if (
                record is not None
                and record.get("status") == "AWAITING_APPROVAL"
            ):
                request = self.intervention_service.create_for_dynamic_tool(
                    self.job,
                    record,
                    source_task_id=task.task_id,
                    timeout_seconds=self.config.intervention_timeout_seconds,
                )
                if (
                    request.metadata.get("response_mode")
                    == "dynamic_tool_activation"
                    and request.metadata.get("dynamic_tool_id") == tool_id
                ):
                    record["approval_request_id"] = request.request_id
                    record["approval_job_id"] = request.job_id
                    self.dynamic_tool_repo.save(record)
                break
        if outcome.accepted_tool_ids or outcome.rejected or expired:
            self.task_repo.record_event(
                self.job.job_id,
                task.task_id,
                "dynamic_tool_lifecycle_updated",
                {
                    "accepted_tool_ids": outcome.accepted_tool_ids,
                    "activated_tool_ids": outcome.activated_tool_ids,
                    "rejected": outcome.rejected,
                    "expired_tool_ids": expired,
                },
                event_key=f"dynamic-tool-lifecycle:{task.task_id}:{task.active_attempt_id}",
            )

    def list_dynamic_tools(self, *, include_code: bool = False) -> list[dict[str, Any]]:
        """Return workspace-generated tools; built-ins are intentionally absent."""

        return self.dynamic_tool_lifecycle.list_records(include_code=include_code)

    def approve_dynamic_tool(
        self, tool_id: str, *, source_task_id: str | None = None
    ) -> bool:
        """Explicit approval hook for a non-read-only generated tool."""

        sandbox = self.sandbox_manager.get(source_task_id) if source_task_id else None
        return self.dynamic_tool_lifecycle.approve(tool_id, sandbox)

    def _apply_persisted_dynamic_tool_decisions(self) -> None:
        for request in self.intervention_repo.list_by_job(self.job.job_id):
            if request.metadata.get("response_mode") != "dynamic_tool_activation":
                continue
            if request.status in {
                InterventionStatus.APPROVED,
                InterventionStatus.REJECTED,
                InterventionStatus.EXPIRED,
            }:
                self._apply_dynamic_tool_decision(request)

    def _request_pending_dynamic_tool_approval(self) -> bool:
        if self.intervention_repo.get_pending_for_job(self.job.job_id) is not None:
            return False
        for record in self.dynamic_tool_repo.list_all():
            if record.get("status") != "AWAITING_APPROVAL":
                continue
            existing_request_id = str(record.get("approval_request_id", ""))
            if existing_request_id:
                existing_request = self.intervention_repo.get(existing_request_id)
                if existing_request is not None:
                    if existing_request.status == InterventionStatus.PENDING:
                        # Another Job may own the one workspace-wide review.
                        return False
                    self._apply_dynamic_tool_decision(existing_request)
                    continue
            evidence = self.dynamic_tool_repo.list_evidence(str(record["tool_id"]))
            source_task_id = str(evidence[-1].get("task_id", "")) if evidence else ""
            request = self.intervention_service.create_for_dynamic_tool(
                self.job,
                record,
                source_task_id=source_task_id,
                timeout_seconds=self.config.intervention_timeout_seconds,
            )
            record["approval_request_id"] = request.request_id
            record["approval_job_id"] = request.job_id
            self.dynamic_tool_repo.save(record)
            return True
        return False

    def _apply_dynamic_tool_decision(self, request) -> None:
        tool_id = str(request.metadata.get("dynamic_tool_id", ""))
        if not tool_id:
            return
        if request.status == InterventionStatus.APPROVED:
            self.dynamic_tool_lifecycle.approve(tool_id)
        elif request.status in {
            InterventionStatus.REJECTED,
            InterventionStatus.EXPIRED,
        }:
            self.dynamic_tool_lifecycle.reject(
                tool_id,
                reason=str(request.response.get("reason") or request.status.value),
            )

    # ---- 上下文快照（§16.2）----

    def _save_snapshot(self) -> None:
        state = {
            "job_status": self.job.status.value,
            "final_reproduction_status": (
                self.job.final_reproduction_status.value
                if self.job.final_reproduction_status is not None
                else None
            ),
            "reflection_round": self.job.reflection_round,
            "budget": self.job.budget.to_dict(),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "attempt": task.attempt,
                    "active_attempt_id": task.active_attempt_id,
                    "dependencies": task.definition.dependencies,
                    "outputs": sorted(task.outputs),
                    "failure": (
                        task.failure_report.to_dict()
                        if task.failure_report is not None
                        else None
                    ),
                }
                for task in sorted(
                    self.scheduler.dag.all_tasks(), key=lambda item: item.task_id
                )
            ],
        }
        fingerprint = json.dumps(state, sort_keys=True, ensure_ascii=False)
        if fingerprint == self._last_snapshot_fingerprint:
            return

        self._decision_version += 1
        self.snapshot_store.save(
            job_id=self.job.job_id,
            dag_version=self._decision_version,
            task_state_version=self._decision_version,
            memory_version=self._decision_version,
            active_issues=[],
            evidence_refs=[],
            main_agent_decision=f"iteration_{self._decision_version}",
            reflection_round=self.job.reflection_round,
            budget_snapshot=self.job.budget.to_dict(),
        )
        self._last_snapshot_fingerprint = fingerprint
        self.snapshot_store.prune_old_snapshots(
            self.job.job_id, keep_last_n_full=3
        )

    def _job_finished(self) -> bool:
        return self.job.status in {
            JobStatus.USER_REPORT_READY,
            JobStatus.FULLY_REPRODUCED,
            JobStatus.VERIFIED_REPRODUCTION_GAP,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
            JobStatus.BLOCKED_BY_MISSING_RESOURCE,
        }

    def _job_waiting_for_user(self) -> bool:
        return self.job.status in {
            JobStatus.WAITING_FOR_USER_DATA,
            JobStatus.WAITING_FOR_MODEL,
            JobStatus.WAITING_FOR_PERMISSION,
        }

    def _expire_overdue_intervention(self) -> None:
        resolutions = self.intervention_service.expire_overdue(self.job.job_id)
        if not resolutions:
            return
        for resolution in resolutions:
            task = resolution.task
            if task is not None:
                # 工具升级请求过期：唤醒挂起中的子智能体线程（拒绝裁决）。
                self.dispatcher.fail_escalation(
                    task.task_id, reason="工具补授审批已过期（fail closed）"
                )
        latest = resolutions[-1]
        self.job = latest.job
        if latest.task is not None:
            self.scheduler.dag.replace_task(latest.task)
