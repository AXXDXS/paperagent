from __future__ import annotations

from repro_agent.agents.base import AgentRunResult
from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import AgentReport, AgentReportType, Task
from repro_agent.orchestrator.dispatcher import SubAgentHandle
from repro_agent.orchestrator.task_factory import build_task_definition


def _task(main_agent) -> Task:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(objective="race", task_type="paper_analysis"),
    )
    task.status = TaskStatus.RUNNING
    task.active_attempt_id = "attempt-current"
    main_agent.scheduler.add_tasks([task])
    return task


def test_old_attempt_result_cannot_mutate_retried_task(main_agent) -> None:
    task = _task(main_agent)
    stale = SubAgentHandle(
        task,
        lambda: AgentRunResult(succeeded=True),
        attempt_id="attempt-old",
        on_progress_push=lambda *_: None,
    )

    accepted = main_agent._accept_attempt_result(
        task, stale, AgentRunResult(succeeded=True)
    )

    assert accepted is False
    assert task.status == TaskStatus.RUNNING
    assert task.task_id not in main_agent._pending_validation


def test_sandbox_is_attempt_scoped(main_agent) -> None:
    task = _task(main_agent)
    first = main_agent.sandbox_manager.create_sandbox(task)
    task.active_attempt_id = "attempt-next"
    second = main_agent.sandbox_manager.create_sandbox(task)

    assert first.root != second.root
    assert first.attempt_id == "attempt-current"
    assert second.attempt_id == "attempt-next"


def test_old_attempt_report_cannot_renew_current_report_lease(main_agent) -> None:
    task = _task(main_agent)
    before = task.next_report_due_at

    accepted = main_agent.scheduler.report_agent(
        task.task_id,
        AgentReport(
            attempt_id="attempt-old",
            report_type=AgentReportType.PROGRESS,
            progress=0.9,
            eta_seconds=999,
        ),
    )

    assert accepted is False
    assert task.latest_agent_report is None
    assert task.next_report_due_at == before
