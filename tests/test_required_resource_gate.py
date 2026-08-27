from __future__ import annotations

from pathlib import Path

from repro_agent.agents.resource.agent import ResourceCheckAgent
from repro_agent.agents.specification.agent import ExperimentSpecificationAgent
from repro_agent.domain.enums import InterventionKind, TaskStatus
from repro_agent.domain.task import Task
from repro_agent.orchestrator.planner import InitialPlanner
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.authorization import ToolAuthorizer


def _run_agent(agent_type, task: Task, tmp_path: Path):
    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    authorization = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        forbidden_actions=task.definition.forbidden_actions,
        sandbox_ctx=sandbox,
        attempt_id="attempt_1",
    )
    return agent_type(
        task,
        authorization,
        MockLLMProvider(),
        attempt_id="attempt_1",
    ).run()


def _locomo_spec() -> dict:
    return {
        "required_resources": [
            {
                "resource_id": "dataset:locomo",
                "name": "LoCoMo",
                "kind": "dataset",
                "required": True,
                "reason": "论文实验使用 LoCoMo 数据集",
                "source_ref": "paper_analysis:page:5",
                "aliases": [],
            }
        ]
    }


def test_specification_extracts_locomo_as_required_runtime_resource(
    tmp_path: Path,
) -> None:
    task = Task(
        job_id="job_spec_resource",
        definition=build_task_definition(
            objective="build experiment specification",
            task_type="specification",
            inputs={
                "experiment_id": "main",
                "target_claim": "reproduce main result",
                "paper_findings": {
                    "extracted_parameters": [
                        {
                            "name": "dataset",
                            "value": "LoCoMo dataset",
                            "page": "5",
                        }
                    ],
                    "effective_parameters": {"dataset": "LoCoMo"},
                    "expected_results": {},
                },
                "code_findings": {"effective_parameters": {}},
            },
        ),
    )

    result = _run_agent(ExperimentSpecificationAgent, task, tmp_path)

    assert result.succeeded is True
    assert result.outputs["required_resources"] == [
        {
            "resource_id": "dataset:locomo",
            "name": "LoCoMo",
            "kind": "dataset",
            "required": True,
            "reason": "实验规格中的 dataset 参数要求该运行资源",
            "source_ref": "paper_analysis:page:5",
            "aliases": [],
        }
    ]
    assert result.outputs["resources"]["required"] == result.outputs[
        "required_resources"
    ]


def test_resource_check_discovers_required_dataset_inside_repository(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = repository / "data" / "locomo"
    dataset.mkdir(parents=True)
    (dataset / "sample.json").write_text("{}\n", encoding="utf-8")
    task = Task(
        job_id="job_resource_found",
        definition=build_task_definition(
            objective="check required resources",
            task_type="resource_check",
            inputs={
                "repository_path": str(repository),
                "experiment_spec": _locomo_spec(),
            },
            restrict_tools=[
                "find_named_resource",
                "check_gpu",
                "check_cuda",
                "check_disk_space",
            ],
        ),
    )

    result = _run_agent(ResourceCheckAgent, task, tmp_path)

    assert result.succeeded is True
    assert result.outputs["blocking_issues"] == []
    locomo = result.outputs["required_resource_status"]["dataset:locomo"]
    assert locomo["status"] == "AVAILABLE_BUT_UNVERIFIED"
    assert locomo["discovery"]["source"] == "repository_search"
    assert locomo["discovery"]["candidates"][0]["relative_path"] == "data/locomo"


def test_missing_required_dataset_is_blocking_and_requests_user_path(
    main_agent,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "train.py").write_text("print('train')\n", encoding="utf-8")
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="check required resources",
            task_type="resource_check",
            inputs={
                "repository_path": str(repository),
                "experiment_spec": _locomo_spec(),
            },
            restrict_tools=[
                "find_named_resource",
                "check_gpu",
                "check_cuda",
                "check_disk_space",
            ],
        ),
    )
    check = _run_agent(ResourceCheckAgent, task, tmp_path / "agent")
    missing = check.outputs["missing_required_resources"]

    assert missing[0]["name"] == "LoCoMo"
    assert check.outputs["blocking_issues"] == [
        "required dataset 'LoCoMo': MISSING"
    ]

    main_agent.scheduler.add_tasks([task])
    request = main_agent.intervention_service.create_for_missing_resources(
        main_agent.job,
        task,
        check.outputs["blocking_issues"],
        missing_required_resources=missing,
    )

    assert request.kind == InterventionKind.USER_DATA
    assert "LoCoMo" in request.question
    assert request.metadata["missing_required_resources"][0]["name"] == "LoCoMo"
    assert task.status == TaskStatus.WAITING_FOR_USER_DATA

    supplied = tmp_path / "provided-locomo"
    supplied.mkdir()
    (supplied / "sample.json").write_text("{}\n", encoding="utf-8")
    resolution = main_agent.intervention_service.resolve(
        request.request_id,
        {"dataset_paths": [str(supplied)]},
    )

    assert resolution.task is not None
    assert resolution.task.status == TaskStatus.PENDING
    assert resolution.task.definition.inputs["dataset_paths"] == [str(supplied)]
    assert "check_path_resource" in resolution.task.definition.allowed_tools


def test_initial_plan_places_resource_gate_between_spec_and_environment(job) -> None:
    tasks = InitialPlanner().plan_initial_tasks(job)
    specification = next(
        task for task in tasks if task.definition.task_type == "specification"
    )
    resource = next(
        task for task in tasks if task.definition.task_type == "resource_check"
    )
    environment = next(
        task for task in tasks if task.definition.task_type == "environment_build"
    )

    assert resource.definition.dependencies == [specification.task_id]
    assert set(environment.definition.dependencies) == {
        specification.task_id,
        resource.task_id,
    }
    assert resource.definition.inputs["repository_path"] == job.inputs.repository_path
    assert "find_named_resource" in resource.definition.allowed_tools
