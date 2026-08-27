from __future__ import annotations

import pytest

from repro_agent.domain.enums import InterventionKind, InterventionStatus, JobStatus, TaskStatus
from repro_agent.domain.task import Task
from repro_agent.orchestrator.execution_parameters import (
    has_current_execution_parameter_approval,
)
from repro_agent.orchestrator.interventions import InterventionValidationError
from repro_agent.orchestrator.main_agent import MainAgent
from repro_agent.orchestrator.task_factory import build_task_definition


def _ready_experiment_task(main_agent: MainAgent) -> Task:
    main_agent.job.status = JobStatus.SMOKE_TEST_RUNNING
    main_agent.job_repo.save(main_agent.job)
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="run the smoke-test experiment",
            task_type="experiment_execution",
            inputs={
                "tier": "SMOKE_TEST",
                "command": ["python", "train.py", "--smoke"],
                "execution_image": "repro:test",
                "working_dir": "workspace://repository",
                "timeout_seconds": 120,
                "gpu_count": 0,
                "metrics_output_path": "output://metrics.json",
            },
            restrict_tools=["execute_command"],
        ),
    )
    return main_agent.scheduler.add_tasks([task])[0]


def test_experiment_is_paused_before_dispatch_and_exposes_effective_parameters(
    main_agent: MainAgent,
) -> None:
    task = _ready_experiment_task(main_agent)

    main_agent.step()

    request = main_agent.pending_intervention()
    assert request is not None
    assert request.kind == InterventionKind.COMMAND
    assert request.task_id == task.task_id
    assert request.metadata["response_mode"] == "execution_parameters"
    assert request.metadata["proposed_parameters"] == {
        "tier": "SMOKE_TEST",
        "command": ["python", "train.py", "--smoke"],
        "execution_image": "repro:test",
        "working_dir": "workspace://repository",
        "timeout_seconds": 120,
        "cpu_cores": 1.0,
        "memory_mb": 1024,
        "disk_mb": 4096,
        "gpu_count": 0,
        "gpu_memory_gb": 0.0,
        "metrics_output_path": "output://metrics.json",
        "experiment_environment": {},
        "experiment_secret_env_vars": [],
        "network_enabled": False,
        "network_hosts": [],
        "workspace_read_only": True,
    }
    persisted = main_agent.scheduler.dag.get(task.task_id)
    assert persisted is not None
    assert persisted.status == TaskStatus.WAITING_FOR_INPUT
    assert persisted.attempt == 0
    assert main_agent.dispatcher.get_handle(task.task_id) is None


def test_confirmed_edits_are_bound_to_the_next_dispatch_attempt(
    main_agent: MainAgent,
) -> None:
    task = _ready_experiment_task(main_agent)
    main_agent.step()
    request = main_agent.pending_intervention()
    assert request is not None

    resolution = main_agent.resolve_intervention(
        request.request_id,
        {
            "approved": True,
            "command": ["python", "train.py", "--batch-size", "8"],
            "execution_image": "repro:edited",
            "working_dir": "workspace://repository",
            "timeout_seconds": 300,
            "gpu_count": 1,
            "metrics_output_path": "output://smoke-metrics.json",
            "reason": "use the smaller smoke configuration",
        },
    )

    assert resolution.request.status == InterventionStatus.RESOLVED
    assert resolution.task is not None
    assert resolution.task.status == TaskStatus.PENDING
    assert resolution.task.definition.inputs["command"] == [
        "python",
        "train.py",
        "--batch-size",
        "8",
    ]
    assert resolution.task.definition.inputs["execution_image"] == "repro:edited"
    assert resolution.task.definition.inputs["timeout_seconds"] == 300
    assert resolution.task.definition.inputs["gpu_count"] == 1
    assert resolution.task.definition.inputs["metrics_output_path"] == "output://smoke-metrics.json"
    assert resolution.task.definition.inputs["tier_command_verified"] is True
    assert has_current_execution_parameter_approval(
        resolution.task.definition.inputs,
        next_attempt=1,
        default_execution_image=main_agent.config.execution_image,
    )

    dispatched = main_agent.scheduler.dispatch([resolution.task])
    assert dispatched == [resolution.task]
    assert resolution.task.attempt == 1
    assert resolution.task.status == TaskStatus.DISPATCHED
    assert not has_current_execution_parameter_approval(
        resolution.task.definition.inputs,
        next_attempt=2,
        default_execution_image=main_agent.config.execution_image,
    )


def test_retry_requires_a_new_confirmation_and_stale_requests_fail_closed(
    main_agent: MainAgent,
) -> None:
    task = _ready_experiment_task(main_agent)
    main_agent.step()
    first = main_agent.pending_intervention()
    assert first is not None
    first_resolution = main_agent.resolve_intervention(first.request_id, {"approved": True})
    assert first_resolution.task is not None

    dispatched = main_agent.scheduler.dispatch([first_resolution.task])[0]
    main_agent.intervention_service._prepare_task_for_resume(dispatched)
    main_agent.task_repo.save(dispatched)
    main_agent.scheduler.dag.replace_task(dispatched)

    main_agent.step()
    retry_request = main_agent.pending_intervention()
    assert retry_request is not None
    assert retry_request.request_id != first.request_id
    assert retry_request.metadata["proposed_parameters"]["command"] == [
        "python",
        "train.py",
        "--smoke",
    ]

    # A task mutation after the request was shown cannot be silently approved.
    dispatched.definition.inputs["timeout_seconds"] = 301
    main_agent.task_repo.save(dispatched)
    with pytest.raises(InterventionValidationError, match="changed while awaiting confirmation"):
        main_agent.resolve_intervention(retry_request.request_id, {"approved": True})
    assert main_agent.intervention_repo.get(retry_request.request_id).is_pending
