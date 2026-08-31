"""Regression tests for dynamic report leases and shared termination handling."""

from __future__ import annotations

import time
from datetime import timedelta

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.agents.registry import SUB_AGENT_REGISTRY
from repro_agent.domain.common import utc_now
from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import Heartbeat, Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.scheduler.agent_reporting import ReportingOutcome, TerminationMode


class _CooperativeLongRunningAgent(BaseSubAgent):
    task_type = "cooperative_long_running"

    def run(self) -> AgentRunResult:
        self.report_progress(0.05, "working", eta_seconds=100.0)
        for _ in range(200):
            self.check_cancellation()
            time.sleep(0.02)
        self.write_json_output("result.json", {"ok": True})
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _UnresponsiveAgent(BaseSubAgent):
    task_type = "unresponsive"

    def run(self) -> AgentRunResult:
        self.report_progress(0.05, "blocked", eta_seconds=100.0)
        time.sleep(5.0)
        return AgentRunResult(succeeded=True, outputs={"ok": True})


def _make_task(main_agent, task_type: str, *, max_overrun_reports: int = 3) -> Task:
    definition = build_task_definition(
        objective=f"test task for {task_type}",
        task_type=task_type,
        extra_allowed_tools=["write_task_output"],
        expected_outputs=["output/result.json"],
        expected_duration_seconds=100,
        max_overrun_reports=max_overrun_reports,
    )
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])
    return task


def _start(main_agent, monkeypatch, task_type, agent_cls, **kwargs):
    monkeypatch.setitem(SUB_AGENT_REGISTRY, task_type, agent_cls)
    task = _make_task(main_agent, task_type, **kwargs)
    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)
    time.sleep(0.05)
    return task


def _settle(main_agent, task_id: str, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while task_id in main_agent._timeout_cancellations and time.monotonic() < deadline:
        main_agent._settle_timeout_terminations()
        time.sleep(0.02)


def test_initial_report_uses_role_estimate_as_dynamic_deadline(main_agent, monkeypatch):
    task = _start(
        main_agent,
        monkeypatch,
        "cooperative_long_running",
        _CooperativeLongRunningAgent,
    )

    decision = main_agent.reporting_policy.evaluate(task)

    assert decision.outcome == ReportingOutcome.WAITING
    assert task.latest_agent_report is not None
    assert task.latest_agent_report.current_step in {"started", "working"}
    assert task.next_report_due_at is not None
    handle = main_agent.dispatcher.get_handle(task.task_id)
    handle.request_graceful_cancel()


def test_activity_signal_does_not_renew_business_report_deadline(main_agent):
    task = _make_task(main_agent, "cooperative_long_running")
    task.status = TaskStatus.RUNNING
    task.active_attempt_id = "attempt-activity"
    task.next_report_due_at = utc_now() + timedelta(seconds=50)
    original_due = task.next_report_due_at

    main_agent._on_subagent_progress_push(
        task, 0.2, "activity:tool_started:execute_command", None
    )

    assert task.next_report_due_at == original_due
    assert task.last_push_heartbeat.current_step.startswith("activity:")


def test_due_report_pulls_status_and_grants_one_bounded_extension(
    main_agent, monkeypatch
):
    task = _start(
        main_agent,
        monkeypatch,
        "cooperative_long_running",
        _CooperativeLongRunningAgent,
    )
    task.next_report_due_at = utc_now() - timedelta(seconds=1)

    main_agent._check_subagent_reporting()

    assert task.status == TaskStatus.RUNNING
    assert task.overrun_report_count == 1
    assert task.latest_agent_report.report_type.value == "extension"
    assert task.last_pull_heartbeat is not None
    assert task.next_report_due_at > utc_now()
    handle = main_agent.dispatcher.get_handle(task.task_id)
    handle.request_graceful_cancel()


def test_pull_probe_does_not_replace_last_push_activity(main_agent, monkeypatch):
    task = _make_task(main_agent, "cooperative_long_running")
    push = Heartbeat(progress=0.4, current_step="training", reported_by="push")
    pull = Heartbeat(progress=0.4, current_step="training (alive)", reported_by="pull")
    task.heartbeat = push
    task.last_push_heartbeat = push
    task.status = TaskStatus.RUNNING

    class _Handle:
        @staticmethod
        def pull_status():
            return pull

    monkeypatch.setattr(main_agent.dispatcher, "get_handle", lambda task_id: _Handle())

    pulled = main_agent.get_subagent_status(task.task_id)

    assert pulled is not None and pulled.reported_by == "pull"
    assert task.heartbeat is push
    assert task.last_push_heartbeat is push
    assert task.last_pull_heartbeat is pulled


def test_third_overdue_report_causes_terminal_cancellation(main_agent, monkeypatch):
    task = _start(
        main_agent,
        monkeypatch,
        "cooperative_long_running",
        _CooperativeLongRunningAgent,
        max_overrun_reports=3,
    )

    for _ in range(3):
        task.next_report_due_at = utc_now() - timedelta(seconds=1)
        main_agent._check_subagent_reporting()

    assert task.reporting_exhausted is True
    assert task.overrun_report_count == 3
    assert task.task_id in main_agent._timeout_cancellations
    _settle(main_agent, task.task_id)
    assert task.status == TaskStatus.TERMINAL_FAILURE
    assert main_agent.dispatcher.get_handle(task.task_id) is None
    main_agent._handle_failed_task(task)
    assert task.status == TaskStatus.TERMINAL_FAILURE


def test_graceful_cancellation_uses_shared_nonblocking_path(main_agent, monkeypatch):
    task = _start(
        main_agent,
        monkeypatch,
        "cooperative_long_running",
        _CooperativeLongRunningAgent,
    )
    handle = main_agent.dispatcher.get_handle(task.task_id)

    main_agent._terminate_subagent(task, handle, reason="test cancellation")
    _settle(main_agent, task.task_id)

    assert task.status == TaskStatus.FAILED_RETRYABLE
    assert main_agent._termination_log[-1].mode == TerminationMode.GRACEFUL


def test_forced_termination_when_agent_ignores_cancel(main_agent, monkeypatch):
    main_agent.config.timeout_cancel_grace_seconds = 0.01
    task = _start(main_agent, monkeypatch, "unresponsive", _UnresponsiveAgent)
    handle = main_agent.dispatcher.get_handle(task.task_id)

    main_agent._terminate_subagent(task, handle, reason="test forced cancellation")
    time.sleep(0.02)
    main_agent._settle_timeout_terminations()

    assert task.status == TaskStatus.TERMINAL_FAILURE
    assert main_agent._termination_log[-1].mode == TerminationMode.FORCED
    assert handle.forced_killed is True


def test_hard_timeout_retains_lease_until_execution_termination(main_agent):
    definition = build_task_definition(
        objective="timeout lease",
        task_type="paper_analysis",
        hard_timeout_seconds=1,
    )
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])
    main_agent.scheduler.dispatch([task])
    main_agent.scheduler.mark_running(task)
    task.started_at = utc_now() - timedelta(seconds=10)

    result = main_agent.scheduler.check_timeouts()

    assert result.hard_timeout_tasks == [task]
    assert task.status == TaskStatus.HARD_TIMEOUT
    assert task.lease_owner is not None
