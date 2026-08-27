"""持久化 Job 的断点恢复。

恢复器只恢复可由数据库和沙箱产物证明的状态；它不假装能接管已消失的
Python 线程，也不把 `RUNNING` 自动当作成功。这样的 fail-closed 策略避免
中断时的半写入结果污染后续 DAG。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import TYPE_CHECKING

from repro_agent.domain.enums import JobStatus, TaskStatus

if TYPE_CHECKING:  # pragma: no cover
    from repro_agent.orchestrator.main_agent import MainAgent


_TERMINAL_JOB_STATUSES = {
    JobStatus.USER_REPORT_READY,
    JobStatus.FULLY_REPRODUCED,
    JobStatus.VERIFIED_REPRODUCTION_GAP,
    JobStatus.CANCELLED,
    JobStatus.FAILED,
    JobStatus.BLOCKED_BY_MISSING_RESOURCE,
}
_INTERRUPTED_TASK_STATUSES = {TaskStatus.DISPATCHED, TaskStatus.RUNNING}


@dataclass(frozen=True)
class RecoveryOutcome:
    job_id: str
    recovered_succeeded_task_ids: list[str] = field(default_factory=list)
    requeued_task_ids: list[str] = field(default_factory=list)
    ignored_task_ids: list[str] = field(default_factory=list)


class RecoveryService:
    """把已持久化状态安全还原为一个可继续运行的 MainAgent。"""

    def __init__(self, agent: "MainAgent"):
        self.agent = agent

    def recover(self) -> RecoveryOutcome:
        if self.agent.job.status in _TERMINAL_JOB_STATUSES:
            raise ValueError(
                f"job {self.agent.job.job_id} is terminal ({self.agent.job.status.value}) and cannot resume"
            )

        self._restore_reflection_runtime_state()
        succeeded: list[str] = []
        requeued: list[str] = []
        ignored: list[str] = []

        for task in list(self.agent.scheduler.dag.all_tasks()):
            if task.status not in _INTERRUPTED_TASK_STATUSES:
                ignored.append(task.task_id)
                continue

            previous_status = task.status
            if not self._reconcile_execution_handle(task):
                self.agent.task_repo.record_event(
                    self.agent.job.job_id,
                    task.task_id,
                    "orphaned_execution_termination_unconfirmed",
                    {"attempt_id": task.active_attempt_id},
                    event_key=f"execution-termination-unconfirmed:{task.active_attempt_id}",
                )
                raise RuntimeError(
                    f"cannot resume task {task.task_id}: orphaned container termination is unconfirmed"
                )
            task.status = TaskStatus.RECOVERING
            self.agent.task_repo.save(task)
            self.agent.scheduler.dag.replace_task(task)
            self.agent.task_repo.record_event(
                self.agent.job.job_id,
                task.task_id,
                "task_recovery_started",
                {
                    "previous_status": previous_status.value,
                    "attempt_id": task.active_attempt_id,
                },
            )

            validation = self.agent.validator.validate_recovered_output(
                task, self._output_dir_for(task.task_id, task.active_attempt_id)
            )
            if validation.passed:
                self.agent.scheduler.mark_succeeded(task, validation.outputs)
                self.agent._resume_experiments_after_code_repair(task)
                self.agent._resume_experiments_after_environment_repair(task)
                self.agent._persist_task_evidence(task)
                self.agent._persist_experiment_run(task)
                self.agent._persist_verification_record(task)
                self.agent._on_task_validated_for_reflection(task)
                self.agent.task_repo.record_event(
                    self.agent.job.job_id,
                    task.task_id,
                    "task_recovery_output_accepted",
                    {
                        "attempt_id": task.active_attempt_id,
                        "outputs": sorted(validation.outputs),
                    },
                )
                succeeded.append(task.task_id)
                continue

            self.agent.scheduler.retry(task)
            self.agent.task_repo.record_event(
                self.agent.job.job_id,
                task.task_id,
                "task_recovery_requeued",
                {
                    "attempt_id": task.active_attempt_id,
                    "validation_reasons": validation.reasons,
                },
            )
            requeued.append(task.task_id)

        # Backfill the transition when the previous process committed the
        # repair success immediately before it could requeue the blocked run.
        for task in list(self.agent.scheduler.dag.all_tasks()):
            if task.status == TaskStatus.SUCCEEDED:
                self.agent._resume_experiments_after_code_repair(task)
                self.agent._resume_experiments_after_environment_repair(task)

        self.agent.job_repo.save(self.agent.job)
        return RecoveryOutcome(
            job_id=self.agent.job.job_id,
            recovered_succeeded_task_ids=succeeded,
            requeued_task_ids=requeued,
            ignored_task_ids=ignored,
        )

    def _output_dir_for(self, task_id: str, attempt_id: str) -> Path:
        return (
            self.agent.sandbox_manager.sandbox_root
            / f"task_{task_id}"
            / attempt_id
            / "output"
        )

    def _reconcile_execution_handle(self, task: Task) -> bool:
        state_path = (
            self.agent.sandbox_manager.sandbox_root
            / f"task_{task.task_id}"
            / task.active_attempt_id
            / "logs"
            / f"{task.active_attempt_id}.execution.json"
        )
        if not state_path.is_file():
            # Older versions wrote the state file only after Popen.  Reconcile
            # the deterministic Docker name before requeueing any task that
            # was allowed to execute commands, closing that historical crash
            # window without requiring the missing file.
            if "execute_command" not in task.definition.allowed_tools:
                return True
            backend = self.agent.sandbox_manager.execution_backend
            if backend is None or not hasattr(backend, "cancel"):
                return False
            if hasattr(backend, "container_name_for"):
                container_name = backend.container_name_for(
                    task.task_id, task.active_attempt_id
                )
            else:
                safe_task = "".join(
                    char if char.isalnum() or char in "_.-" else "-"
                    for char in task.task_id
                )
                safe_attempt = "".join(
                    char if char.isalnum() or char in "_.-" else "-"
                    for char in task.active_attempt_id
                )
                container_name = f"repro-{safe_task}-{safe_attempt}"[:120]
            cancelled = bool(backend.cancel(container_name))
            self.agent.task_repo.record_event(
                self.agent.job.job_id,
                task.task_id,
                "orphaned_execution_reconciled_without_state",
                {"container_name": container_name, "cancelled": cancelled},
                event_key=f"execution-reconciled-missing-state:{task.active_attempt_id}",
            )
            return cancelled
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return task.definition.task_type != "experiment_execution"
        if state.get("status") not in {
            "PREPARING",
            "RUNNING",
            "TERMINATION_FAILED",
        }:
            return True
        backend = self.agent.sandbox_manager.execution_backend
        container_name = str(state.get("container_name", ""))
        if (
            backend is not None
            and state.get("runtime") == "conda"
            and hasattr(backend, "reconcile_execution")
        ):
            cancelled = bool(backend.reconcile_execution(state))
        else:
            cancelled = bool(
                backend is not None
                and hasattr(backend, "cancel")
                and backend.cancel(container_name)
            )
        state.update(
            {
                "status": "RECONCILED" if cancelled else "TERMINATION_FAILED",
                "recovery_cancel_succeeded": cancelled,
            }
        )
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        self.agent.task_repo.record_event(
            self.agent.job.job_id,
            task.task_id,
            "orphaned_execution_reconciled",
            {"container_name": container_name, "cancelled": cancelled},
            event_key=f"execution-reconciled:{task.active_attempt_id}",
        )
        return cancelled

    def _restore_reflection_runtime_state(self) -> None:
        """Reconcile durable reflection workflow state after any crash window."""

        reports = self.agent.reflection_repo.list_by_job(self.agent.job.job_id)
        self.agent._reflection_reports = reports

        # Validation and its workflow callback are intentionally separate.  If
        # the process died between them, replay succeeded reflection callbacks;
        # the task-derived report id makes this operation idempotent.
        for task in list(self.agent.scheduler.dag.all_tasks()):
            if (
                task.definition.task_type == "reflection"
                and task.status == TaskStatus.SUCCEEDED
            ):
                self.agent._on_reflection_task_succeeded(task)

        self.agent._pending_audit_task_ids = {}
        for report in list(self.agent._reflection_reports):
            if report.audit_result is not None:
                continue
            if not report.audit_context:
                report.audit_context = self.agent._build_reflection_audit_context()
                self.agent.reflection_repo.save(report)

            # A report may have committed immediately before the process died
            # and before its audit tasks were inserted.  Re-plan by stable
            # creation keys and retain the canonical persisted task ids.
            audit_tasks = self.agent.scheduler.add_tasks(
                self.agent.reflection_controller.plan_audit(report)
            )
            report.audit_task_ids = [task.task_id for task in audit_tasks]
            self.agent.reflection_repo.save(report)
            self.agent._pending_audit_task_ids[report.reflection_id] = {
                task.task_id
                for task in audit_tasks
                if task.status != TaskStatus.SUCCEEDED
            }

            # Likewise, backfill a finding when the audit task was marked
            # succeeded but the post-validation callback had not yet committed.
            for task in audit_tasks:
                if task.status == TaskStatus.SUCCEEDED:
                    self.agent._on_audit_task_succeeded(task)
            # An empty set is meaningful: all audit tasks finished before the
            # process died, but their summary transition may still be pending.
