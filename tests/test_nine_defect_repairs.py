from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from repro_agent.agents.experiment.agent import (
    ExperimentExecutionAgent,
    ExperimentExecutionResult,
)
from repro_agent.domain.enums import (
    ExperimentTier,
    FailureType,
    TaskStatus,
)
from repro_agent.domain.experiment import ExperimentRun
from repro_agent.domain.reflection import ReflectionHypothesis, ReflectionReport
from repro_agent.domain.task import FailureReport, Task
from repro_agent.execution.docker import DockerExecutionBackend
from repro_agent.execution.backend import ExecutionRequest
from repro_agent.orchestrator.phases import PhaseCoordinator
from repro_agent.orchestrator.reflection_controller import ReflectionController
from repro_agent.orchestrator.runtime_configuration import (
    missing_requirements,
    runtime_network_configuration,
)
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.schemas.results import TaskResultEnvelope


def _experiment(main_agent, failure_type: FailureType) -> Task:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="run experiment",
            task_type="experiment_execution",
            inputs={
                "tier": ExperimentTier.SMOKE_TEST.value,
                "command": ["python", "train.py"],
                "repository_path": main_agent.job.inputs.repository_path,
                "execution_image": "repro:old",
                "environment_base_image": "python:3.11-slim",
                "working_dir": "workspace://repository",
                "creation_key": f"test:{failure_type.value}",
            },
        ),
    )
    task.attempt = 1
    task.active_attempt_id = "attempt-failed"
    task.status = TaskStatus.FAILED_RETRYABLE
    task.failure_report = FailureReport(
        failure_type=failure_type,
        failed_step="execute",
        error_message="ModuleNotFoundError: missing dependency",
        metadata={
            "command": ["python", "train.py"],
            "tier": ExperimentTier.SMOKE_TEST.value,
            "stderr_tail": "ModuleNotFoundError: missing dependency",
        },
    )
    main_agent.scheduler.add_tasks([task])
    main_agent.task_repo.save(task)
    return task


def _write_result(task: Task, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            TaskResultEnvelope.succeeded(
                task_id=task.task_id,
                attempt_id=task.active_attempt_id,
                task_type=task.definition.task_type,
                payload=payload,
            ).to_dict()
        ),
        encoding="utf-8",
    )
    task.outputs = {"result.json": str(path)}


def test_cuda_oom_and_exit_137_are_resource_failures() -> None:
    for stderr in (
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "Killed",
    ):
        result = ExperimentExecutionResult(
            tier=ExperimentTier.SMOKE_TEST.value,
            command=["python", "train.py"],
            exit_code=137,
            stderr_tail=stderr,
            termination_reason="completed",
        )
        report = ExperimentExecutionAgent._failure_report_for_unsuccessful_execution(
            result,
            tier=ExperimentTier.SMOKE_TEST,
            metrics_required=False,
        )
        assert report.failure_type == FailureType.RESOURCE_EXCEEDED


def test_task_resource_limits_reach_the_sandbox_policy(tmp_path: Path) -> None:
    task = Task(
        job_id="job",
        definition=build_task_definition(
            objective="bounded run",
            task_type="experiment_execution",
            inputs={
                "cpu_cores": 3.5,
                "memory_mb": 8192,
                "disk_mb": 16384,
                "gpu_count": 2,
                "gpu_memory_gb": 24,
            },
        ),
    )
    sandbox = SandboxManager(tmp_path).create_sandbox(task)
    limits = sandbox.policy.resource_limits
    assert limits.cpu_cores == 3.5
    assert limits.memory_mb == 8192
    assert limits.disk_mb == 16384
    assert limits.gpu_count == 2
    assert limits.gpu_memory_mb == 24 * 1024


def test_confirmed_api_base_is_the_only_runtime_network_switch(tmp_path: Path) -> None:
    requirements = [
        {
            "name": "MODEL_API_BASE",
            "kind": "api_base",
            "delivery": "environment",
            "environment_variable": "MODEL_API_BASE",
            "required": True,
        }
    ]
    assert missing_requirements(requirements, {"MODEL_API_BASE": "not-a-url"})
    enabled, hosts = runtime_network_configuration(
        requirements, {"MODEL_API_BASE": "https://api.example.test/v1"}
    )
    assert enabled is True
    assert hosts == ["api.example.test"]

    input_dir = tmp_path / "input"
    workspace_dir = tmp_path / "workspace"
    output_dir = tmp_path / "output"
    for directory in (input_dir, workspace_dir, output_dir):
        directory.mkdir()
    request = ExecutionRequest(
        task_id="task",
        attempt_id="attempt",
        command=["python", "train.py"],
        image="python:3.11-slim",
        input_dir=input_dir,
        workspace_dir=workspace_dir,
        output_dir=output_dir,
    )
    request = replace(request, network_enabled=True)
    argv = DockerExecutionBackend().build_run_argv(request)
    assert argv[argv.index("--network") + 1] == "bridge"


def test_environment_failure_creates_one_rebuild_prerequisite_and_resumes(
    main_agent, tmp_path: Path
) -> None:
    failed = _experiment(main_agent, FailureType.ENVIRONMENT_ERROR)

    assert main_agent._handle_failed_task(failed) is False
    assert failed.status == TaskStatus.BLOCKED
    environment_id = failed.definition.inputs["environment_repair_task_id"]
    environment = main_agent.scheduler.dag.get(environment_id)
    assert environment is not None
    assert environment.definition.task_type == "environment_build"
    assert environment.definition.inputs["force_rebuild"] is True
    duplicate_experiments = [
        task
        for task in main_agent.scheduler.dag.all_tasks()
        if task.definition.task_type == "experiment_execution"
    ]
    assert duplicate_experiments == [failed]

    environment.active_attempt_id = "attempt-env-repair"
    environment.status = TaskStatus.SUCCEEDED
    _write_result(
        environment,
        tmp_path / "environment-result.json",
            {
                "image_ref": "sha256:" + "b" * 64,
                "environment_fingerprint": "f" * 64,
                "import_test_passed": True,
            },
    )
    main_agent._resume_experiments_after_environment_repair(environment)

    assert failed.status == TaskStatus.PENDING
    assert failed.definition.inputs["execution_image"] == "sha256:" + "b" * 64


def test_decomposition_rewires_children_without_terminal_failure(main_agent) -> None:
    broad = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="analyze broad repository",
            task_type="code_analysis",
            inputs={"repository_path": main_agent.job.inputs.repository_path},
        ),
    )
    child = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="build spec",
            task_type="specification",
            dependencies=[broad.task_id],
        ),
    )
    main_agent.scheduler.add_tasks([broad, child])
    subtasks = main_agent.replanner.decompose(broad)
    main_agent.scheduler.replace_with_subtasks(broad, subtasks)

    assert broad.status == TaskStatus.CANCELLED
    assert broad.task_id not in child.dependencies
    assert set(child.dependencies) == {task.task_id for task in subtasks}
    assert all(
        task.definition.inputs["required_checks"]
        for task in subtasks
    )


def test_repaired_repository_propagates_to_next_tier_and_verification(main_agent) -> None:
    repaired = "/durable/repaired/repository"
    environment = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="environment",
            task_type="environment_build",
        ),
        status=TaskStatus.SUCCEEDED,
    )
    static = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="static",
            task_type="experiment_execution",
            inputs={
                "tier": ExperimentTier.STATIC_CHECK.value,
                "repository_path": repaired,
            },
        ),
        status=TaskStatus.SUCCEEDED,
    )
    runs = [
        ExperimentRun(
            job_id=main_agent.job.job_id,
            experiment_id="main_experiment",
            tier=ExperimentTier.STATIC_CHECK,
            exit_code=0,
        )
    ]
    decision = PhaseCoordinator().advance(
        main_agent.job, [environment, static], runs
    )
    assert decision.tasks_to_create[0].definition.inputs["repository_path"] == repaired

    full = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="full",
            task_type="experiment_execution",
            inputs={
                "tier": ExperimentTier.FULL_EXPERIMENT.value,
                "repository_path": repaired,
            },
        ),
        status=TaskStatus.SUCCEEDED,
    )
    all_runs = [
        ExperimentRun(
            job_id=main_agent.job.job_id,
            experiment_id="main_experiment",
            tier=tier,
            exit_code=0,
        )
        for tier in ExperimentTier
    ]
    verification = PhaseCoordinator().advance(
        main_agent.job, [environment, full], all_runs
    ).tasks_to_create[0]
    assert verification.definition.inputs["repository_path"] == repaired


def test_audit_plan_carries_real_sources_dependencies_and_required_checks() -> None:
    report = ReflectionReport(
        job_id="job",
        round=1,
        hypotheses=[
            ReflectionHypothesis(
                category="B",
                description="verify evaluation path",
                priority=10,
                confidence=0.9,
                required_checks=["metric implementation", "checkpoint selection"],
            ),
            ReflectionHypothesis(
                category="C",
                description="verify batch size",
                priority=9,
                confidence=0.8,
                required_checks=["batch_size"],
            ),
        ],
        audit_context={
            "repository_path": "/repo/repaired",
            "target_experiments": ["main"],
            "experiment_id": "main",
            "upstream_task_ids": {
                "paper_analysis": ["paper-1"],
                "code_analysis": ["code-1"],
            },
        },
    )
    tasks = ReflectionController().plan_audit(report)
    code = next(task for task in tasks if task.definition.task_type == "code_analysis")
    spec = next(task for task in tasks if task.definition.task_type == "specification")
    assert code.definition.inputs["repository_path"] == "/repo/repaired"
    assert code.definition.inputs["required_checks"] == [
        "metric implementation",
        "checkpoint selection",
    ]
    assert spec.dependencies == ["paper-1", "code-1"]


def test_soft_timeout_cannot_requeue_until_old_handle_finishes(main_agent) -> None:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="slow task", task_type="paper_analysis"
        ),
        status=TaskStatus.SOFT_TIMEOUT,
    )
    task.active_attempt_id = "attempt-slow"
    main_agent.scheduler.add_tasks([task])
    main_agent._timeout_cancellations[task.task_id] = time.monotonic()

    assert main_agent._handle_failed_task(task) is False
    assert task.status == TaskStatus.SOFT_TIMEOUT
    assert task.task_id in main_agent._timeout_cancellations


def test_repaired_repository_is_found_after_controller_restart(main_agent) -> None:
    repair = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="repair", task_type="coding"
        ),
        status=TaskStatus.SUCCEEDED,
    )
    repair.active_attempt_id = "attempt-repaired"
    main_agent.scheduler.add_tasks([repair])
    repository = (
        main_agent.sandbox_manager.sandbox_root
        / f"task_{repair.task_id}"
        / repair.active_attempt_id
        / "workspace"
        / "repository"
    )
    repository.mkdir(parents=True)
    report = ReflectionReport(
        job_id=main_agent.job.job_id,
        round=1,
        repair_task_ids=[repair.task_id],
    )

    assert main_agent._repository_after_repairs(report) == str(repository.resolve())


def test_recovery_reconciles_deterministic_container_when_state_is_missing(
    main_agent,
) -> None:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="interrupted execution",
            task_type="experiment_execution",
            inputs={"command": ["python", "train.py"]},
            restrict_tools=["execute_command"],
            expected_outputs=["output/result.json"],
        ),
        status=TaskStatus.RUNNING,
    )
    task.active_attempt_id = "attempt-orphan"
    main_agent.scheduler.add_tasks([task])
    main_agent.task_repo.save(task)
    captured: list[str] = []

    class Backend:
        @staticmethod
        def cancel(container_name: str) -> bool:
            captured.append(container_name)
            return True

        @staticmethod
        def container_name_for(task_id: str, attempt_id: str) -> str:
            return f"stable-{task_id}-{attempt_id}"

    main_agent.sandbox_manager.execution_backend = Backend()
    outcome = main_agent.recover_interrupted_tasks()

    assert captured == [f"stable-{task.task_id}-{task.active_attempt_id}"]
    assert outcome.requeued_task_ids == [task.task_id]


def test_recovered_output_rejects_symlinked_result(main_agent, tmp_path: Path) -> None:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="recover", task_type="paper_analysis",
            expected_outputs=["output/result.json"],
        ),
    )
    task.active_attempt_id = "attempt"
    outside = tmp_path / "outside-result.json"
    _write_result(task, outside, {"ok": True})
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").symlink_to(outside)

    validation = main_agent.validator.validate_recovered_output(task, output)
    assert validation.passed is False
    assert any("缺少预期产物" in reason for reason in validation.reasons)
