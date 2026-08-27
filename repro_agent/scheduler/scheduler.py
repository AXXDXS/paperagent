"""确定性任务调度器（设计文档 §8、§13）。

这是主智能体主循环（§19）依赖的核心基础设施：
    - 维护任务 DAG 的就绪判定（依赖 domain.dag.TaskDAG）；
    - 控制并发槽位数量（``max_parallel_agents``）；
    - 按 §8.3 的优先级规则选择要派发的任务；
    - 检查心跳与软/硬超时（§13.2-§13.4）；
    - 维护执行租约（借鉴 DeerFlow 的 Lease+心跳模型，见 lease.py）。

调度器本身**不执行**任务（不调用任何子智能体/LLM），只负责"选择
下一步该跑什么、检测该终止什么"，实际派发执行是
``orchestrator.main_loop`` 通过 ``AgentDispatcher`` 完成的——这是
设计文档 §5.1 强调的"主智能体不直接处理具体内容"原则在工程上的落地：
调度器是纯状态机，可以脱离任何 LLM 独立单元测试。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from repro_agent.domain.common import utc_now
from repro_agent.domain.common import new_id
from repro_agent.domain.dag import TaskDAG
from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import FailureReport, Task
from repro_agent.domain.enums import FailureType
from repro_agent.scheduler.lease import LeaseManager
from repro_agent.scheduler.priority import rank_ready_tasks
from repro_agent.scheduler.timeout_policy import TimeoutOutcome, TimeoutPolicy
from repro_agent.storage.repository import TaskRepository

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    max_parallel_agents: int = 8
    lease_duration_seconds: float = 60.0
    heartbeat_stale_multiplier: float = 3.0


@dataclass
class SchedulingResult:
    """一次调度轮询的结果摘要，供主循环记录到日志/事件流。"""

    newly_ready: list[Task] = field(default_factory=list)
    newly_blocked: list[Task] = field(default_factory=list)
    dispatched: list[Task] = field(default_factory=list)
    soft_timeout_tasks: list[Task] = field(default_factory=list)
    hard_timeout_tasks: list[Task] = field(default_factory=list)
    stalled_tasks: list[Task] = field(default_factory=list)


class TaskScheduler:
    """单个 Job 的任务调度器实例。"""

    def __init__(
        self,
        job_id: str,
        task_repo: TaskRepository,
        config: Optional[SchedulerConfig] = None,
    ):
        self.job_id = job_id
        self.task_repo = task_repo
        self.config = config or SchedulerConfig()
        self.dag = TaskDAG(job_id=job_id)
        self.lease_manager = LeaseManager(self.config.lease_duration_seconds)
        self.timeout_policy = TimeoutPolicy(
            heartbeat_stale_multiplier=self.config.heartbeat_stale_multiplier
        )
        self._running_count = 0
        self._load_from_storage()

    def _load_from_storage(self) -> None:
        for task in self.task_repo.list_by_job(self.job_id):
            self.dag.add_task(task)
        self._running_count = sum(
            1
            for t in self.dag.all_tasks()
            if t.status in {TaskStatus.DISPATCHED, TaskStatus.RUNNING}
        )

    # ---- 任务生命周期管理 ----

    def add_tasks(self, tasks: list[Task]) -> list[Task]:
        """Ensure logical tasks exist and return their canonical persisted objects.

        A caller may be replaying a workflow transition after a crash.  In that
        case a freshly planned ``Task`` has a new random id even though its
        ``creation_key`` already exists.  Returning the persisted object keeps
        downstream dependency lists and in-memory wait sets tied to the real id.
        """

        materialized: list[Task] = []
        for task in tasks:
            creation_key = str(task.definition.inputs.get("creation_key", "") or "")
            existing = self.task_repo.get_by_creation_key(self.job_id, creation_key)
            if existing is not None:
                logger.info(
                    "skip duplicate task creation_key=%s; existing task=%s",
                    creation_key,
                    existing.task_id,
                )
                canonical = self.dag.get(existing.task_id)
                if canonical is None:
                    self.dag.add_task(existing)
                    canonical = existing
                materialized.append(canonical)
                continue
            self.dag.add_task(task)
            self.task_repo.save_with_event(
                task,
                "task_created",
                {"creation_key": creation_key, "task_type": task.definition.task_type},
                event_key=f"task-created:{task.task_id}",
            )
            materialized.append(task)
        return materialized

    def replace_with_subtasks(self, task: Task, subtasks: list[Task]) -> None:
        """Replace one broad task without poisoning the Job as a failure.

        Downstream edges are rewired to every canonical subtask.  The original
        node remains as an audit tombstone (CANCELLED), not TERMINAL_FAILURE.
        """

        if not subtasks:
            raise ValueError("cannot replace a task with an empty subtask set")
        child_ids = list(self.dag.children_of(task.task_id))
        canonical = self.add_tasks(subtasks)
        replacement_ids = [item.task_id for item in canonical]
        for child_id in child_ids:
            child = self.dag.get(child_id)
            if child is None:
                continue
            self.dag.replace_dependency(
                child_id, task.task_id, replacement_ids
            )
            self.task_repo.save_with_event(
                child,
                "task_dependency_rewired_after_decomposition",
                {
                    "replaced_dependency": task.task_id,
                    "replacement_dependencies": replacement_ids,
                },
                event_key=f"task-dependency-rewired:{child_id}:{task.task_id}",
            )
        task.status = TaskStatus.CANCELLED
        self.task_repo.save_with_event(
            task,
            "task_decomposed",
            {"replaced_by": replacement_ids},
            event_key=f"task-decomposed:{task.task_id}:{task.active_attempt_id}",
        )
        self.dag.replace_task(task)

    def mark_waiting_for_user(self, task: Task) -> None:
        task.status = TaskStatus.WAITING_FOR_INPUT
        self.task_repo.save(task)
        self.dag.replace_task(task)

    def mark_terminal_failure(self, task: Task, reason: str = "") -> None:
        task.status = TaskStatus.TERMINAL_FAILURE
        self.task_repo.save_with_event(
            task,
            "terminal_failure",
            {"reason": reason, "attempt_id": task.active_attempt_id},
            event_key=f"terminal-failure:{task.task_id}:{task.active_attempt_id}",
        )
        self.dag.replace_task(task)

    def retry(self, task: Task, *, guidance: str | None = None) -> None:
        """Requeue a task with exactly one main-agent retry instruction.

        ``guidance`` is persisted in task inputs before the previous failure
        report is cleared.  Replacing the single string on each retry prevents
        an ever-growing list of historical instructions, while changing the
        task checkpoint scope so a failed attempt's LLM response is not replayed.
        """

        normalized_guidance = " ".join(str(guidance or "").split()).strip()
        if not normalized_guidance:
            normalized_guidance = (
                "重试注意事项：重新执行前先核对上次失败步骤及其前置条件，"
                "不要直接重复相同操作。"
            )
        task.definition.inputs["retry_guidance"] = normalized_guidance

        task.status = TaskStatus.PENDING
        task.assigned_agent = None
        task.started_at = None
        task.dispatched_at = None
        task.completed_at = None
        task.heartbeat = None
        task.last_push_heartbeat = None
        task.last_pull_heartbeat = None
        task.last_activity_signature = ""
        task.failure_report = None
        self.lease_manager.release(task)
        self.task_repo.save_with_event(
            task,
            "task_requeued",
            {
                "attempt_id": task.active_attempt_id,
                "retry_guidance": normalized_guidance,
            },
            event_key=f"task-requeued:{task.task_id}:{task.active_attempt_id}",
        )
        self.dag.replace_task(task)

    def block_until(
        self,
        task: Task,
        prerequisite: Task,
        *,
        prerequisite_input_key: str | None = None,
    ) -> Task:
        """§19 决策 add_prerequisite：给任务追加一个新的前置依赖任务。"""

        canonical = self.add_tasks([prerequisite])[0]
        if canonical.task_id not in task.definition.dependencies:
            self.dag.add_dependency(task.task_id, canonical.task_id)
        if prerequisite_input_key:
            task.definition.inputs[prerequisite_input_key] = canonical.task_id
        task.status = TaskStatus.BLOCKED
        self.task_repo.save_with_event(
            task,
            "task_prerequisite_added",
            {"prerequisite_task_id": canonical.task_id},
            event_key=f"task-prerequisite:{task.task_id}:{canonical.task_id}",
        )
        self.dag.replace_task(task)
        return canonical

    # ---- 核心调度步骤（对应 §19 主循环的调度器调用序列） ----

    def refresh_task_states(self) -> SchedulingResult:
        """刷新所有 PENDING/BLOCKED 任务状态（依赖满足性判定）。"""

        result = SchedulingResult()
        changed = self.dag.unblock_or_block()
        for task in changed:
            self.task_repo.save_with_event(
                task,
                "task_ready" if task.status == TaskStatus.READY else "task_blocked",
                {"status": task.status.value},
                event_key=f"task-state:{task.task_id}:{task.status.value}:{task.attempt}",
            )
            if task.status == TaskStatus.READY:
                result.newly_ready.append(task)
            elif task.status == TaskStatus.BLOCKED:
                result.newly_blocked.append(task)
        return result

    def check_heartbeats(self) -> None:
        """§19 ``scheduler.check_heartbeats()``：仅刷新心跳新鲜度判定的前置状态。

        真正的心跳数据由子智能体通过 ``report_heartbeat`` 写入，这里
        不主动拉取（避免和沙箱执行耦合），只是一个显式的调用点，
        便于未来接入"主动探测子进程存活"的实现而不改变主循环代码。
        """

        return None

    def report_heartbeat(self, task_id: str, heartbeat) -> None:
        task = self.dag.get(task_id)
        if task is None:
            return
        if heartbeat.reported_by == "push":
            task.last_push_heartbeat = heartbeat
            task.heartbeat = heartbeat
        else:
            task.last_pull_heartbeat = heartbeat
        self.task_repo.save(task)

    def check_timeouts(self) -> SchedulingResult:
        """检查所有 RUNNING/DISPATCHED 任务的心跳与超时（§13.2-§13.4）。"""

        result = SchedulingResult()
        now = utc_now()
        for task in self.dag.all_tasks():
            if task.status not in {TaskStatus.RUNNING, TaskStatus.DISPATCHED}:
                continue
            decision = self.timeout_policy.evaluate(task, now=now)
            if decision.outcome == TimeoutOutcome.HARD_TIMEOUT:
                self._handle_hard_timeout(task, decision.detail)
                result.hard_timeout_tasks.append(task)
            elif decision.outcome == TimeoutOutcome.STALLED:
                task.status = TaskStatus.SOFT_TIMEOUT
                task.failure_report = FailureReport(
                    failure_type=FailureType.AGENT_STALLED,
                    failed_step=(
                        task.heartbeat.current_step if task.heartbeat else "unknown"
                    ),
                    last_successful_step=(
                        task.heartbeat.last_completed_step if task.heartbeat else ""
                    ),
                    error_message=decision.detail,
                    likely_causes=["软超时后心跳或执行活动没有继续增长"],
                    recommended_action="确认旧 attempt 已终止后再重试",
                )
                self.task_repo.save_with_event(
                    task,
                    "soft_timeout_stalled",
                    {"detail": decision.detail, "attempt_id": task.active_attempt_id},
                    event_key=f"soft-timeout:{task.task_id}:{task.active_attempt_id}",
                )
                self.dag.replace_task(task)
                result.stalled_tasks.append(task)
                result.soft_timeout_tasks.append(task)
            elif decision.outcome == TimeoutOutcome.SLOW_BUT_ALIVE:
                result.soft_timeout_tasks.append(task)
                logger.info("task %s slow but alive: %s", task.task_id, decision.detail)
        return result

    def _handle_hard_timeout(self, task: Task, detail: str) -> None:
        """§13.4 硬超时处理：停止任务、保存证据、标记 HARD_TIMEOUT。"""

        task.status = TaskStatus.HARD_TIMEOUT
        task.failure_report = FailureReport(
            failure_type=FailureType.AGENT_STALLED,
            failed_step=task.heartbeat.current_step if task.heartbeat else "unknown",
            last_successful_step=task.heartbeat.last_completed_step
            if task.heartbeat
            else "",
            error_message=detail,
            likely_causes=["任务执行超过硬超时限制"],
            recommended_action="由主智能体判断重试或拆分",
        )
        self.task_repo.save_with_event(
            task,
            "hard_timeout",
            {"detail": detail, "attempt_id": task.active_attempt_id},
            event_key=f"hard-timeout:{task.task_id}:{task.active_attempt_id}",
        )
        self.dag.replace_task(task)

    def get_ready_tasks(self) -> list[Task]:
        return self.dag.ready_tasks()

    def get_completed_tasks(self) -> list[Task]:
        """本轮新增的 SUCCEEDED 任务留给上层通过事件表判断；这里返回全部
        SUCCEEDED 任务，去重交由调用方（主智能体已验证过的任务不会重复处理）。
        """

        return [t for t in self.dag.all_tasks() if t.status == TaskStatus.SUCCEEDED]

    def get_failed_tasks(self) -> list[Task]:
        return [t for t in self.dag.all_tasks() if t.status.is_failure]

    def available_slots(self) -> int:
        running = sum(
            1
            for t in self.dag.all_tasks()
            if t.status in {TaskStatus.DISPATCHED, TaskStatus.RUNNING}
        )
        return max(0, self.config.max_parallel_agents - running)

    def select_tasks(self, ready_tasks: list[Task], max_parallel_agents: Optional[int] = None) -> list[Task]:
        """按 §8.3 优先级规则，从 ready_tasks 中选出本轮可派发的任务。"""

        limit = max_parallel_agents or self.config.max_parallel_agents
        slots = min(self.available_slots(), limit)
        if slots <= 0:
            return []
        ranked = rank_ready_tasks(self.dag, ready_tasks)
        return ranked[:slots]

    def dispatch(self, tasks: list[Task], owner_prefix: str = "agent") -> list[Task]:
        """把选中的任务标记为 DISPATCHED 并分配执行租约。"""

        dispatched = []
        now = utc_now()
        for idx, task in enumerate(tasks):
            owner = f"{owner_prefix}-{task.task_id}"
            if not self.lease_manager.acquire(task, owner, now=now):
                continue
            task.attempt += 1
            task.active_attempt_id = new_id("attempt")
            task.status = TaskStatus.DISPATCHED
            task.dispatched_at = now
            task.assigned_agent = owner
            self.task_repo.save_with_event(
                task,
                "task_dispatched",
                {"attempt_id": task.active_attempt_id, "owner": owner},
                event_key=f"task-dispatched:{task.active_attempt_id}",
            )
            self.dag.replace_task(task)
            dispatched.append(task)
        return dispatched

    def mark_running(self, task: Task) -> None:
        task.status = TaskStatus.RUNNING
        task.started_at = task.started_at or utc_now()
        self.task_repo.save_with_event(
            task,
            "task_running",
            {"attempt_id": task.active_attempt_id},
            event_key=f"task-running:{task.active_attempt_id}",
        )
        self.dag.replace_task(task)

    def mark_succeeded(self, task: Task, outputs: dict) -> None:
        task.status = TaskStatus.SUCCEEDED
        task.completed_at = utc_now()
        task.outputs = outputs
        self.lease_manager.release(task)
        self.task_repo.save_with_event(
            task,
            "task_succeeded",
            {"attempt_id": task.active_attempt_id, "outputs": sorted(outputs)},
            event_key=f"task-succeeded:{task.active_attempt_id}",
        )
        self.dag.replace_task(task)

    def mark_failed(
        self,
        task: Task,
        status: TaskStatus,
        failure_report: Optional[FailureReport] = None,
    ) -> None:
        task.status = status
        task.completed_at = utc_now()
        if failure_report is not None:
            task.failure_report = failure_report
        self.lease_manager.release(task)
        self.task_repo.save_with_event(
            task,
            "task_failed",
            {
                "attempt_id": task.active_attempt_id,
                "status": status.value,
                "failure": failure_report.to_dict() if failure_report else None,
            },
            event_key=f"task-failed:{task.active_attempt_id}:{status.value}",
        )
        self.dag.replace_task(task)

    def summary(self) -> dict[str, int]:
        return self.dag.summary()
