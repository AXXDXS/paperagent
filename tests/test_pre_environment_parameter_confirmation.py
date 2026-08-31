from __future__ import annotations

from repro_agent.domain.enums import ExperimentTier, InterventionKind, TaskStatus
from repro_agent.domain.task import Task
from repro_agent.orchestrator.main_agent import MainAgent
from repro_agent.orchestrator.task_factory import build_task_definition


def _initial_resource_and_environment(main_agent: MainAgent) -> tuple[Task, Task]:
    main_agent.job.inputs.user_run_commands = [
        "python -m compileall -q .",
        "python -m pytest -q",
        "python train.py --tier smoke --model",
        "python train.py --tier reduced --model",
        "python train.py --tier full --model",
    ]
    main_agent.job_repo.save(main_agent.job)
    resource = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="check final requested resources",
            task_type="resource_check",
            inputs={"creation_key": "test:resource"},
            restrict_tools=["check_disk_space"],
        ),
    )
    resource = main_agent.scheduler.add_tasks([resource])[0]
    environment = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="build only after user confirmation",
            task_type="environment_build",
            dependencies=[resource.task_id],
            inputs={
                "base_image": "python:3.11-slim",
                "repository_path": main_agent.job.inputs.repository_path,
                "creation_key": "test:environment",
            },
            restrict_tools=["build_environment_image"],
        ),
    )
    environment = main_agent.scheduler.add_tasks([environment])[0]
    return resource, environment


def _mark_resource_succeeded(main_agent: MainAgent, resource: Task) -> None:
    main_agent._apply_unconfirmed_resource_defaults(resource)
    main_agent.scheduler.mark_succeeded(resource, {})


def test_run_plan_is_confirmed_after_resource_and_before_environment_dispatch(
    main_agent: MainAgent, monkeypatch
) -> None:
    resource, environment = _initial_resource_and_environment(main_agent)
    dispatched: list[str] = []
    monkeypatch.setattr(
        main_agent.create_subagents_tool,
        "call",
        lambda *, task_ids: dispatched.extend(task_ids),
    )

    main_agent.step()

    assert main_agent.pending_intervention() is None
    assert main_agent.scheduler.dag.get(resource.task_id).status == TaskStatus.DISPATCHED
    assert dispatched == [resource.task_id]
    assert main_agent.dispatcher.get_handle(environment.task_id) is None

    main_agent.scheduler.mark_succeeded(resource, {})
    main_agent.step()

    request = main_agent.pending_intervention()
    assert request is not None
    assert request.kind == InterventionKind.COMMAND
    assert request.task_id == environment.task_id
    assert request.metadata["response_mode"] == "pre_environment_execution_plan"
    assert set(request.metadata["proposed_plan"]["tier_commands"]) == {
        tier.value for tier in ExperimentTier
    }
    assert main_agent.scheduler.dag.get(resource.task_id).status == TaskStatus.SUCCEEDED
    persisted_environment = main_agent.scheduler.dag.get(environment.task_id)
    assert persisted_environment.status == TaskStatus.WAITING_FOR_INPUT
    assert resource.task_id in persisted_environment.dependencies
    assert main_agent.dispatcher.get_handle(environment.task_id) is None


def test_required_model_configuration_is_combined_with_pre_environment_plan(
    main_agent: MainAgent, monkeypatch
) -> None:
    resource, environment = _initial_resource_and_environment(main_agent)
    main_agent.job.inputs.required_experiment_configurations = [
        {
            "name": "MODEL_NAME",
            "kind": "model_name",
            "delivery": "command_argument",
            "argument": "--model",
            "required": True,
            "reason": "training exits when the model is absent",
            "source_ref": "train.py:12",
        },
        {
            "name": "MODEL_API_BASE",
            "kind": "api_base",
            "delivery": "environment",
            "environment_variable": "MODEL_API_BASE",
            "required": True,
            "reason": "remote model client requires an endpoint",
            "source_ref": "client.py:8",
        },
        {
            "name": "OPENAI_API_KEY",
            "kind": "credential_env",
            "delivery": "environment",
            "environment_variable": "OPENAI_API_KEY",
            "required": True,
            "reason": "remote model client requires authentication",
            "source_ref": "client.py:9",
        },
    ]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main_agent.job_repo.save(main_agent.job)
    _mark_resource_succeeded(main_agent, resource)

    main_agent.step()
    plan_request = main_agent.pending_intervention()
    assert plan_request is not None
    assert plan_request.metadata["response_mode"] == "pre_environment_execution_plan"
    assert plan_request.metadata["required_value_names"] == [
        "MODEL_NAME",
        "MODEL_API_BASE",
    ]
    assert plan_request.metadata["required_secret_env_vars"] == ["OPENAI_API_KEY"]
    assert "values" in plan_request.input_schema["required"]
    assert main_agent.dispatcher.get_handle(environment.task_id) is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-only")
    resolution = main_agent.resolve_intervention(
        plan_request.request_id,
        {
            "approved": True,
            "values": {
                "MODEL_NAME": "paper-model-v2",
                "MODEL_API_BASE": "https://models.example.test/v1",
            },
            "confirmed_secret_env_vars": ["OPENAI_API_KEY"],
        },
    )
    assert resolution.task is not None
    tier_commands = main_agent.job.inputs.confirmed_execution_plan["tier_commands"]
    # ``--model`` is a command_argument requirement: it is bound only on
    # commands that already declare the flag (train.py) and must NOT be
    # appended to commands that never accepted it (compileall/pytest),
    # otherwise the run dies with an argument-parser error.
    for tier, command in tier_commands.items():
        if "--model" in command:
            assert command[command.index("--model") + 1] == "paper-model-v2"
        else:
            assert "paper-model-v2" not in command
    assert any(
        "--model" in command for command in tier_commands.values()
    ), "at least the train.py tiers must receive the bound --model value"
    assert main_agent.job.inputs.experiment_runtime_config["MODEL_NAME"] == (
        "paper-model-v2"
    )
    plan = main_agent.job.inputs.confirmed_execution_plan
    assert plan["experiment_environment"] == {
        "MODEL_API_BASE": "https://models.example.test/v1"
    }
    assert plan["experiment_secret_env_vars"] == ["OPENAI_API_KEY"]
    assert plan["network_enabled"] is True
    assert plan["network_hosts"] == ["models.example.test"]


def test_confirmed_plan_updates_resource_and_environment_inputs(
    main_agent: MainAgent, monkeypatch
) -> None:
    resource, environment = _initial_resource_and_environment(main_agent)
    _mark_resource_succeeded(main_agent, resource)
    main_agent.step()
    request = main_agent.pending_intervention()
    assert request is not None

    resolution = main_agent.resolve_intervention(
        request.request_id,
        {
            "approved": True,
            "base_image": "python:3.12-slim",
            "cpu_cores": 3.0,
            "memory_mb": 6144,
            "disk_mb": 12288,
            "gpu_count": 0,
            "timeout_seconds": 900,
        },
    )

    assert resolution.task is not None
    assert resolution.task.definition.inputs["cpu_cores"] == 3.0
    assert resolution.task.definition.inputs["memory_mb"] == 6144
    assert main_agent.job.inputs.confirmed_execution_plan["base_image"] == "python:3.12-slim"
    assert main_agent.job.inputs.max_runtime_seconds == 900
    persisted_job = main_agent.job_repo.get(main_agent.job.job_id)
    assert persisted_job is not None
    assert persisted_job.inputs.confirmed_execution_plan["memory_mb"] == 6144

    persisted_resource = main_agent.scheduler.dag.get(resource.task_id)
    assert persisted_resource is not None
    assert persisted_resource.status == TaskStatus.PENDING
    assert persisted_resource.definition.inputs["requested_cpu_cores"] == 3.0
    assert persisted_resource.definition.inputs["requested_memory_mb"] == 6144
    main_agent.scheduler.mark_succeeded(persisted_resource, {})
    dispatched: list[str] = []
    monkeypatch.setattr(
        main_agent.create_subagents_tool,
        "call",
        lambda *, task_ids: dispatched.extend(task_ids),
    )
    main_agent.step()

    persisted_environment = main_agent.scheduler.dag.get(environment.task_id)
    assert persisted_environment is not None
    assert persisted_environment.status == TaskStatus.DISPATCHED
    assert persisted_environment.definition.inputs["base_image"] == "python:3.12-slim"
    assert persisted_environment.definition.inputs["cpu_cores"] == 3.0
    assert persisted_environment.definition.inputs["memory_mb"] == 6144
    assert persisted_environment.definition.inputs["disk_mb"] == 12288
    assert dispatched == [environment.task_id]


def test_matching_experiment_uses_pre_environment_approval_without_second_prompt(
    main_agent: MainAgent,
) -> None:
    resource, _ = _initial_resource_and_environment(main_agent)
    _mark_resource_succeeded(main_agent, resource)
    main_agent.step()
    request = main_agent.pending_intervention()
    assert request is not None
    main_agent.resolve_intervention(request.request_id, {"approved": True})
    plan = main_agent.job.inputs.confirmed_execution_plan
    tier = ExperimentTier.SMOKE_TEST.value
    experiment = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="run confirmed smoke command",
            task_type="experiment_execution",
            inputs={
                "tier": tier,
                "command": list(plan["tier_commands"][tier]),
                "execution_image": "sha256:" + "a" * 64,
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
            },
            restrict_tools=["execute_command"],
        ),
    )

    assert main_agent._pause_for_execution_parameter_confirmation([experiment]) is False
    assert experiment.definition.inputs["tier_command_verified"] is True
    assert main_agent.pending_intervention() is None
