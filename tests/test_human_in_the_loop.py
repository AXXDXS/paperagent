from __future__ import annotations

import json
from datetime import timedelta

import pytest

from repro_agent.cli.main import main
from repro_agent.domain.common import utc_now
from repro_agent.domain.enums import (
    FailureType,
    InterventionKind,
    InterventionStatus,
    JobStatus,
    TaskStatus,
)
from repro_agent.domain.task import FailureReport, Task
from repro_agent.orchestrator.interventions import InterventionValidationError
from repro_agent.orchestrator.main_agent import MainAgent
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.schemas.results import TaskResultEnvelope
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.resource_tools import check_path_resource


def _failed_task(main_agent: MainAgent, failure_type: FailureType, message: str) -> Task:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="需要人工处理的任务",
            task_type="paper_analysis",
            restrict_tools=["read_file"],
        ),
        status=TaskStatus.FAILED_RETRYABLE,
        attempt=1,
        failure_report=FailureReport(
            failure_type=failure_type,
            failed_step="tool_call",
            error_message=message,
            recommended_action="ask user",
        ),
    )
    main_agent.scheduler.add_tasks([task])
    return task


def test_failure_pauses_without_spinning_and_persists_structured_request(main_agent) -> None:
    main_agent.job.status = JobStatus.CODE_ANALYSIS_RUNNING
    main_agent.job_repo.save(main_agent.job)
    task = _failed_task(main_agent, FailureType.DATA_ERROR, "dataset is unavailable")

    outcome = main_agent.run_until_finished(max_iterations=100)

    assert outcome.completed is False
    assert outcome.paused is True
    assert outcome.reason == "waiting_for_user"
    assert outcome.iterations == 1
    assert main_agent.job.status == JobStatus.WAITING_FOR_USER_DATA
    assert task.status == TaskStatus.WAITING_FOR_USER_DATA
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None
    assert request.task_id == task.task_id
    assert request.kind == InterventionKind.USER_DATA
    assert "dataset_paths" in request.input_schema["properties"]


def test_resource_exhaustion_pauses_for_runtime_resource_confirmation(main_agent) -> None:
    main_agent.job.status = JobStatus.CODE_ANALYSIS_RUNNING
    main_agent.job_repo.save(main_agent.job)
    task = _failed_task(main_agent, FailureType.RESOURCE_EXCEEDED, "timeout_killed")

    outcome = main_agent.run_until_finished(max_iterations=10)

    assert outcome.paused is True
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None
    assert request.task_id == task.task_id
    assert request.kind == InterventionKind.RESOURCE
    assert "max_runtime_seconds" in request.input_schema["properties"]


def test_answer_updates_inputs_and_unblocks_exact_task_across_restart(
    main_agent, mock_provider, tmp_path
) -> None:
    main_agent.job.status = JobStatus.RESOURCE_CHECK_RUNNING
    main_agent.job_repo.save(main_agent.job)
    task = _failed_task(main_agent, FailureType.DATA_ERROR, "dataset is unavailable")
    main_agent.run_until_finished(max_iterations=10)
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None
    replacement = str(tmp_path / "replacement-dataset")

    resolution = main_agent.resolve_intervention(
        request.request_id,
        {"dataset_paths": [replacement]},
        responded_by="tester",
    )

    assert resolution.request.status == InterventionStatus.RESOLVED
    assert resolution.job.status == JobStatus.RESOURCE_CHECK_RUNNING
    assert resolution.job.inputs.dataset_paths == [replacement]
    assert resolution.task is not None
    assert resolution.task.status == TaskStatus.PENDING
    assert resolution.task.definition.inputs["dataset_paths"] == [replacement]
    assert main_agent.job.status == JobStatus.RESOURCE_CHECK_RUNNING
    assert main_agent.scheduler.dag.get(task.task_id).status == TaskStatus.PENDING

    resumed = MainAgent.resume_from_storage(
        main_agent.job.job_id, main_agent.config, mock_provider
    )
    resumed.recover_interrupted_tasks()
    restored = resumed.scheduler.dag.get(task.task_id)
    assert restored is not None
    assert restored.status == TaskStatus.PENDING
    assert resumed.job.inputs.dataset_paths == [replacement]


def test_permission_approval_is_task_scoped_and_cannot_cross_risk_budget(main_agent) -> None:
    main_agent.job.status = JobStatus.CODE_ANALYSIS_RUNNING
    main_agent.job_repo.save(main_agent.job)
    task = _failed_task(
        main_agent,
        FailureType.PERMISSION_ERROR,
        "task is not authorized to call tool 'grep_files'",
    )
    main_agent.run_until_finished(max_iterations=10)
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None

    with pytest.raises(InterventionValidationError, match="security boundary"):
        main_agent.intervention_service.resolve(
            request.request_id,
            {"approved": True, "approved_tools": ["execute_command"]},
        )
    assert main_agent.intervention_repo.get(request.request_id).is_pending

    resolution = main_agent.intervention_service.resolve(
        request.request_id,
        {"approved": True, "approved_tools": ["grep_files"]},
        responded_by="security-reviewer",
    )
    assert resolution.request.status == InterventionStatus.APPROVED
    assert resolution.task is not None
    assert resolution.task.status == TaskStatus.PENDING
    assert "grep_files" in resolution.task.definition.allowed_tools


def test_rejection_is_audited_and_fails_closed(main_agent) -> None:
    task = _failed_task(main_agent, FailureType.MODEL_ERROR, "checkpoint missing")
    main_agent.run_until_finished(max_iterations=10)
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None

    resolution = main_agent.intervention_service.reject(
        request.request_id,
        reason="cannot provide this model",
        responded_by="owner",
    )

    assert resolution.request.status == InterventionStatus.REJECTED
    assert resolution.job.status == JobStatus.FAILED
    assert resolution.task is not None
    assert resolution.task.status == TaskStatus.TERMINAL_FAILURE
    events = main_agent.task_repo.list_events(main_agent.job.job_id)
    assert any(e["event_type"] == "intervention_closed_without_resume" for e in events)


def test_expired_intervention_fails_closed_on_next_run(main_agent) -> None:
    task = _failed_task(main_agent, FailureType.MODEL_ERROR, "checkpoint missing")
    main_agent.run_until_finished(max_iterations=10)
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None
    request.expires_at = utc_now() - timedelta(seconds=1)
    main_agent.intervention_repo.save(request)

    outcome = main_agent.run_until_finished(max_iterations=10)

    assert outcome.completed is True
    assert outcome.terminal_status == JobStatus.FAILED
    expired = main_agent.intervention_repo.get(request.request_id)
    assert expired is not None
    assert expired.status == InterventionStatus.EXPIRED
    assert main_agent.scheduler.dag.get(task.task_id).status == TaskStatus.TERMINAL_FAILURE


def test_blocking_resource_result_becomes_intervention_instead_of_terminal(main_agent, tmp_path) -> None:
    main_agent.job.status = JobStatus.RESOURCE_CHECK_RUNNING
    main_agent.job_repo.save(main_agent.job)
    missing = str(tmp_path / "missing-dataset")
    main_agent.job.inputs.dataset_paths = [missing]
    resource_task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="resource check",
            task_type="resource_check",
            inputs={"dataset_paths": [missing]},
        ),
        status=TaskStatus.SUCCEEDED,
        attempt=1,
        active_attempt_id="attempt_resource",
    )
    output = tmp_path / "resource-result.json"
    output.write_text(
        json.dumps(
            TaskResultEnvelope.succeeded(
                task_id=resource_task.task_id,
                attempt_id=resource_task.active_attempt_id,
                task_type="resource_check",
                payload={"blocking_issues": [f"{missing}: MISSING"]},
            ).to_dict()
        ),
        encoding="utf-8",
    )
    resource_task.outputs = {"result.json": str(output)}
    main_agent.scheduler.add_tasks([resource_task])

    paused = main_agent._advance_phases()

    assert paused is True
    assert main_agent.job.status == JobStatus.WAITING_FOR_USER_DATA
    assert resource_task.status == TaskStatus.WAITING_FOR_USER_DATA
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None
    assert request.metadata["blocking_issues"] == [f"{missing}: MISSING"]


def test_missing_declared_resource_is_preserved_as_safe_missing_probe(tmp_path) -> None:
    missing = str(tmp_path / "secret-host-parent" / "dataset")
    task = Task(
        job_id="job_resource_probe",
        definition=build_task_definition(
            objective="resource check",
            task_type="resource_check",
            inputs={"dataset_paths": [missing]},
        ),
    )

    sandbox = SandboxManager(tmp_path / "sandbox").create_sandbox(task)
    staged = task.definition.inputs["dataset_paths"]

    assert len(staged) == 1
    assert staged[0].startswith("input://_missing/")
    assert str(tmp_path) not in staged[0]
    assert check_path_resource(sandbox, staged[0])["status"] == "MISSING"


def test_cli_lists_and_responds_to_intervention(main_agent, tmp_path, capsys) -> None:
    task = _failed_task(main_agent, FailureType.DATA_ERROR, "dataset unavailable")
    main_agent.run_until_finished(max_iterations=10)
    request = main_agent.intervention_repo.get_pending_for_job(main_agent.job.job_id)
    assert request is not None
    work_dir = str(tmp_path / "unused")
    # The fixture DB path is authoritative; derive its containing work directory.
    work_dir = str(main_agent.db.path.parent)

    assert main(
        [
            "intervention",
            "list",
            "--job-id",
            main_agent.job.job_id,
            "--work-dir",
            work_dir,
        ]
    ) == 0
    assert request.request_id in capsys.readouterr().out

    assert main(
        [
            "intervention",
            "respond",
            "--request-id",
            request.request_id,
            "--work-dir",
            work_dir,
            "--response-json",
            json.dumps({"dataset_paths": [str(tmp_path / "data")]}),
        ]
    ) == 0
    assert "RESOLVED" in capsys.readouterr().out
    assert task.task_id
