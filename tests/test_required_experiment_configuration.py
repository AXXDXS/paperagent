from __future__ import annotations

import json

import pytest

from repro_agent.agents.code.agent import CodeAnalysisAgent
from repro_agent.domain.enums import InterventionKind, InterventionStatus, JobStatus, TaskStatus
from repro_agent.domain.task import Task
from repro_agent.orchestrator.interventions import InterventionValidationError
from repro_agent.orchestrator.main_agent import MainAgent
from repro_agent.orchestrator.task_factory import build_task_definition


def test_required_configuration_must_reference_validated_code_evidence() -> None:
    requirements = [
        {
            "name": "MODEL_NAME",
            "kind": "model_name",
            "delivery": "command_argument",
            "argument": "--model",
            "required": True,
            "reason": "required parser argument",
            "source_ref": "train.py:12",
        },
        {
            "name": "HALLUCINATED_API_KEY",
            "kind": "credential_env",
            "delivery": "environment",
            "environment_variable": "HALLUCINATED_API_KEY",
            "required": True,
            "reason": "not supported by inspected code",
            "source_ref": "missing.py:99",
        },
    ]

    validated = CodeAnalysisAgent._validate_required_configuration_evidence(
        requirements,
        [{"path": "train.py", "start_line": 10, "end_line": 20}],
    )

    assert [item["name"] for item in validated] == ["MODEL_NAME"]


def _ready_task(main_agent: MainAgent) -> Task:
    main_agent.job.status = JobStatus.SMOKE_TEST_RUNNING
    main_agent.job.inputs.required_experiment_configurations = [
        {
            "name": "MODEL_NAME",
            "kind": "model_name",
            "delivery": "command_argument",
            "argument": "--model",
            "required": True,
            "reason": "the training entry point exits when --model is absent",
            "source_ref": "train.py:12",
        },
        {
            "name": "MODEL_API_BASE",
            "kind": "api_base",
            "delivery": "environment",
            "environment_variable": "MODEL_API_BASE",
            "required": True,
            "reason": "the API client requires an endpoint",
            "source_ref": "client.py:8",
        },
        {
            "name": "OPENAI_API_KEY",
            "kind": "credential_env",
            "delivery": "environment",
            "environment_variable": "OPENAI_API_KEY",
            "required": True,
            "reason": "the API client rejects unauthenticated calls",
            "source_ref": "client.py:9",
        },
    ]
    main_agent.job_repo.save(main_agent.job)
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="run API-backed model experiment",
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


def test_missing_required_configuration_pauses_before_parameter_confirmation(
    main_agent: MainAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task = _ready_task(main_agent)

    main_agent.step()

    request = main_agent.pending_intervention()
    assert request is not None
    assert request.kind == InterventionKind.MODEL
    assert request.metadata["response_mode"] == "required_experiment_configuration"
    assert request.metadata["required_value_names"] == ["MODEL_NAME", "MODEL_API_BASE"]
    assert request.metadata["required_secret_env_vars"] == ["OPENAI_API_KEY"]
    assert request.metadata["credential_values_must_not_be_persisted"] is True
    persisted = main_agent.scheduler.dag.get(task.task_id)
    assert persisted is not None
    assert persisted.status == TaskStatus.WAITING_FOR_USER_DATA
    assert persisted.attempt == 0
    assert main_agent.dispatcher.get_handle(task.task_id) is None


def test_response_requires_live_secret_then_binds_safe_values_to_execution(
    main_agent: MainAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-never-persist-this-value"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task = _ready_task(main_agent)
    main_agent.step()
    request = main_agent.pending_intervention()
    assert request is not None
    payload = {
        "values": {
            "MODEL_NAME": "paper-model-v2",
            "MODEL_API_BASE": "https://models.example.test/v1",
        },
        "confirmed_secret_env_vars": ["OPENAI_API_KEY"],
    }

    with pytest.raises(InterventionValidationError, match="not set"):
        main_agent.resolve_intervention(request.request_id, payload)

    monkeypatch.setenv("OPENAI_API_KEY", secret)
    resolution = main_agent.resolve_intervention(request.request_id, payload)
    assert resolution.request.status == InterventionStatus.RESOLVED
    assert resolution.task is not None
    assert resolution.task.definition.inputs["command"] == [
        "python",
        "train.py",
        "--smoke",
        "--model",
        "paper-model-v2",
    ]
    assert resolution.task.definition.inputs["experiment_environment"] == {
        "MODEL_API_BASE": "https://models.example.test/v1"
    }
    assert resolution.task.definition.inputs["experiment_secret_env_vars"] == [
        "OPENAI_API_KEY"
    ]
    assert resolution.task.definition.inputs["execution_manifest"]["model_identifier"] == (
        "paper-model-v2"
    )
    assert secret not in json.dumps(resolution.request.to_dict(), ensure_ascii=False)
    assert secret not in json.dumps(resolution.job.to_dict(), ensure_ascii=False)

    # The existing exact-parameter confirmation remains the second gate and
    # now exposes safe runtime values plus secret variable names for review.
    main_agent.step()
    confirmation = main_agent.pending_intervention()
    assert confirmation is not None
    assert confirmation.metadata["response_mode"] == "execution_parameters"
    proposed = confirmation.metadata["proposed_parameters"]
    assert proposed["command"][-2:] == ["--model", "paper-model-v2"]
    assert proposed["experiment_environment"] == {
        "MODEL_API_BASE": "https://models.example.test/v1"
    }
    assert proposed["experiment_secret_env_vars"] == ["OPENAI_API_KEY"]
    assert secret not in json.dumps(confirmation.to_dict(), ensure_ascii=False)


def test_preconfigured_values_and_live_secret_skip_required_configuration_gate(
    main_agent: MainAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-present-only-in-process")
    task = _ready_task(main_agent)
    main_agent.job.inputs.experiment_runtime_config.update(
        {
            "MODEL_NAME": "already-selected",
            "MODEL_API_BASE": "https://models.example.test/v1",
        }
    )
    main_agent.job_repo.save(main_agent.job)

    main_agent.step()

    request = main_agent.pending_intervention()
    assert request is not None
    assert request.metadata["response_mode"] == "execution_parameters"
    assert request.task_id == task.task_id
