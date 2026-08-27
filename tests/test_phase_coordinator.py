from __future__ import annotations

from repro_agent.domain.enums import ExperimentTier, JobStatus, TaskStatus
from repro_agent.domain.experiment import ExperimentRun
from repro_agent.domain.job import JobInputs, ReproductionJob
from repro_agent.domain.task import Task, TaskDefinition
from repro_agent.orchestrator.phases import PhaseCoordinator
from repro_agent.domain.enums import ReproductionStatus
from repro_agent.schemas.results import TaskResultEnvelope


def _task(task_type: str, status: TaskStatus = TaskStatus.SUCCEEDED, **inputs) -> Task:
    return Task(
        job_id="job_1",
        definition=TaskDefinition(objective=task_type, task_type=task_type, inputs=inputs),
        status=status,
    )


def _job() -> ReproductionJob:
    return ReproductionJob(
        job_id="job_1",
        inputs=JobInputs(
            paper_path="paper.txt",
            repository_path="repo",
            target_experiments=["main"],
            user_run_commands=["python train.py"],
        ),
    )


def test_environment_success_creates_static_check_instead_of_finishing() -> None:
    tasks = [
        _task("paper_analysis"),
        _task("code_analysis"),
        _task("resource_check"),
        _task("specification"),
        _task("environment_build"),
    ]

    decision = PhaseCoordinator().advance(_job(), tasks, runs=[])

    assert decision.terminal_status is None
    assert decision.job_status == JobStatus.UNIT_TEST_RUNNING
    assert len(decision.tasks_to_create) == 1
    assert decision.tasks_to_create[0].definition.inputs["tier"] == ExperimentTier.STATIC_CHECK.value


def test_tier_gate_creates_only_immediate_next_experiment_tier() -> None:
    tasks = [_task("environment_build")]
    runs = [
        ExperimentRun(
            job_id="job_1",
            experiment_id="main",
            tier=ExperimentTier.STATIC_CHECK,
            exit_code=0,
        )
    ]

    decision = PhaseCoordinator().advance(_job(), tasks, runs=runs)

    assert len(decision.tasks_to_create) == 1
    assert decision.tasks_to_create[0].definition.inputs["tier"] == ExperimentTier.UNIT_TEST.value


def test_full_run_creates_verification_not_success_terminal() -> None:
    code_task = _task("code_analysis")
    tasks = [_task("environment_build"), code_task]
    runs = [
        ExperimentRun(job_id="job_1", experiment_id="main", tier=tier, exit_code=0)
        for tier in ExperimentTier
    ]

    decision = PhaseCoordinator().advance(_job(), tasks, runs=runs)

    assert decision.terminal_status is None
    assert decision.job_status == JobStatus.RESULT_VERIFICATION_RUNNING
    verification = decision.tasks_to_create[0]
    assert verification.definition.task_type == "verification"
    assert code_task.task_id in verification.dependencies
    assert verification.definition.inputs["repository_path"] == "repo"


def test_active_reflection_is_not_overwritten_by_report_ready() -> None:
    job = _job()
    job.status = JobStatus.REFLECTION_REQUIRED
    tasks = [_task("environment_build"), _task("verification")]
    runs = [
        ExperimentRun(job_id="job_1", experiment_id="main", tier=tier, exit_code=0)
        for tier in ExperimentTier
    ]

    decision = PhaseCoordinator().advance(job, tasks, runs=runs)

    assert decision.job_status == JobStatus.REFLECTION_REQUIRED
    assert decision.terminal_status is None


def _verification_task(tmp_path, payload: dict) -> Task:
    task = _task("verification")
    task.active_attempt_id = "attempt-1"
    path = tmp_path / "verification.json"
    path.write_text(
        __import__("json").dumps(
            TaskResultEnvelope.succeeded(
                task_id=task.task_id,
                attempt_id=task.active_attempt_id,
                task_type="verification",
                payload=payload,
            ).to_dict()
        ),
        encoding="utf-8",
    )
    task.outputs = {"result.json": str(path)}
    return task


def test_mock_verification_only_produces_pipeline_diagnostic(tmp_path) -> None:
    tasks = [
        _task("environment_build"),
        _verification_task(
            tmp_path,
            {
                "comparisons": [],
                "run_actually_executed": True,
                "mock": True,
                "verification_valid": False,
            },
        ),
    ]
    runs = [
        ExperimentRun(
            job_id="job_1",
            experiment_id="main",
            tier=tier,
            exit_code=0,
            run_type="mock",
        )
        for tier in ExperimentTier
    ]

    decision = PhaseCoordinator().advance(_job(), tasks, runs=runs)

    assert decision.job_status == JobStatus.USER_REPORT_READY
    assert decision.reproduction_status == ReproductionStatus.PIPELINE_ONLY
