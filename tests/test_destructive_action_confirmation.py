from __future__ import annotations

from pathlib import Path

import pytest

from repro_agent.context.snapshot import SnapshotStore
from repro_agent.domain.enums import FailureType, InterventionStatus, JobStatus, TaskStatus
from repro_agent.domain.task import FailureReport, Task
from repro_agent.execution.mock import MockExecutionBackend
from repro_agent.orchestrator.main_agent import MainAgent
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.destructive_actions import (
    DestructiveActionConfirmationRequired,
    command_fingerprint,
    inspect_destructive_command,
)
from repro_agent.tools.write_tools import execute_command


@pytest.mark.parametrize(
    "command",
    [
        ["rm", "-rf", "output"],
        ["git", "clean", "-fdx"],
        ["find", ".", "-name", "*.tmp", "-delete"],
        ["bash", "-c", "rm -f result.json"],
        ["python", "-c", "import os; os.unlink('result.json')"],
        ["psql", "-c", "DROP TABLE experiment_runs"],
    ],
)
def test_common_delete_commands_require_confirmation(command: list[str]) -> None:
    finding = inspect_destructive_command(command)

    assert finding is not None
    assert finding.fingerprint == command_fingerprint(command)


def _execution_task(command: list[str], *, attempt: int = 1) -> Task:
    task = Task(
        job_id="job",
        definition=build_task_definition(
            objective="run command",
            task_type="experiment_execution",
            inputs={"command": command},
            restrict_tools=["execute_command"],
        ),
        attempt=attempt,
    )
    task.active_attempt_id = f"attempt_{attempt}"
    return task


def test_execute_command_blocks_until_exact_attempt_approval(tmp_path: Path) -> None:
    command = ["rm", "-f", "artifact.json"]
    manager = SandboxManager(tmp_path / "sandboxes", execution_backend=MockExecutionBackend())
    task = _execution_task(command, attempt=2)
    sandbox = manager.create_sandbox(task)

    with pytest.raises(DestructiveActionConfirmationRequired):
        execute_command(sandbox, command)

    task.definition.inputs["_destructive_action_approvals"] = [
        {
            "fingerprint": command_fingerprint(command),
            "approved_for_attempt": 2,
        }
    ]
    task.active_attempt_id = "attempt_2_approved"
    approved_sandbox = manager.create_sandbox(task)
    assert execute_command(approved_sandbox, command)["exit_code"] == 0

    changed = ["rm", "-f", "different.json"]
    with pytest.raises(DestructiveActionConfirmationRequired):
        execute_command(approved_sandbox, changed)


def test_destructive_failure_creates_bound_human_confirmation(
    main_agent: MainAgent,
) -> None:
    command = ["rm", "-rf", "/output/stale"]
    fingerprint = command_fingerprint(command)
    main_agent.job.status = JobStatus.UNIT_TEST_RUNNING
    main_agent.job_repo.save(main_agent.job)
    task = _execution_task(command, attempt=1)
    task.job_id = main_agent.job.job_id
    task.status = TaskStatus.FAILED_RETRYABLE
    task.failure_report = FailureReport(
        failure_type=FailureType.PERMISSION_ERROR,
        failed_step="destructive_action_confirmation",
        error_message="destructive command requires explicit human confirmation",
        metadata={
            "response_mode": "destructive_action",
            "command": command,
            "command_fingerprint": fingerprint,
            "detection_reasons": ["direct delete executable: rm"],
        },
    )
    main_agent.scheduler.add_tasks([task])

    outcome = main_agent.run_until_finished(max_iterations=10)
    request = main_agent.pending_intervention()

    assert outcome.paused is True
    assert request is not None
    assert request.metadata["response_mode"] == "destructive_action"
    assert request.metadata["command_fingerprint"] == fingerprint
    assert command == request.metadata["command"]

    resolution = main_agent.resolve_intervention(
        request.request_id,
        {"approved": True, "reason": "reviewed exact target"},
        responded_by="owner",
    )

    assert resolution.request.status == InterventionStatus.APPROVED
    approval = resolution.task.definition.inputs["_destructive_action_approvals"][-1]
    assert approval["fingerprint"] == fingerprint
    assert approval["approved_for_attempt"] == 2
    assert resolution.task.status == TaskStatus.PENDING


def test_dispatcher_converts_delete_gate_to_structured_permission_failure(
    main_agent: MainAgent,
) -> None:
    main_agent.sandbox_manager.execution_backend = MockExecutionBackend()
    task = _execution_task(["rm", "-f", "/output/result.json"], attempt=0)
    task.job_id = main_agent.job.job_id
    task = main_agent.scheduler.add_tasks([task])[0]
    main_agent.scheduler.refresh_task_states()
    task = main_agent.scheduler.dispatch([task])[0]

    outcome = main_agent.dispatcher.dispatch_and_run(task)

    assert outcome.result.succeeded is False
    failure = outcome.result.failure_report
    assert failure is not None
    assert failure.failure_type == FailureType.PERMISSION_ERROR
    assert failure.metadata["response_mode"] == "destructive_action"
    assert failure.metadata["command"] == ["rm", "-f", "/output/result.json"]


def test_internal_file_cleanup_is_noop_without_confirmation(tmp_path: Path) -> None:
    task = _execution_task(["python", "-V"])
    manager = SandboxManager(tmp_path / "sandboxes", execution_backend=MockExecutionBackend())
    sandbox = manager.create_sandbox(task)
    marker = sandbox.workspace_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    assert manager.cleanup(task.task_id) is False
    assert marker.exists()
    assert manager.cleanup(task.task_id, human_confirmed=True) is True
    assert not marker.exists()

    snapshots = SnapshotStore(tmp_path / "snapshots", full_snapshot_interval=2)
    for version in range(1, 7):
        snapshots.save(
            job_id="job",
            dag_version=version,
            task_state_version=version,
            memory_version=version,
            active_issues=[],
            evidence_refs=[],
            main_agent_decision=f"decision-{version}",
            reflection_round=0,
            budget_snapshot={},
        )
    before = list((tmp_path / "snapshots" / "job").glob("v*.json"))
    assert snapshots.prune_old_snapshots("job", keep_last_n_full=1) == 0
    assert len(list((tmp_path / "snapshots" / "job").glob("v*.json"))) == len(before)
    assert snapshots.prune_old_snapshots(
        "job", keep_last_n_full=1, human_confirmed=True
    ) > 0
