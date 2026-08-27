from __future__ import annotations

from types import SimpleNamespace

from repro_agent.agents.experiment.agent import (
    ExperimentExecutionAgent,
    ExperimentExecutionResult,
)
from repro_agent.domain.enums import ExperimentTier, FailureType, TaskStatus
from repro_agent.domain.task import FailureReport, Task
from repro_agent.orchestrator.main_agent import MainAgent
from repro_agent.orchestrator.task_factory import build_task_definition


def _failed_experiment(main_agent: MainAgent) -> Task:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="run smoke test",
            task_type="experiment_execution",
            inputs={
                "tier": ExperimentTier.SMOKE_TEST.value,
                "command": ["python", "train.py"],
                "repository_path": main_agent.job.inputs.repository_path,
                "dataset_paths": [],
                "model_paths": [],
                "checkpoint_paths": [],
                "working_dir": "workspace://repository",
                "execution_image": "sha256:" + "a" * 64,
                "creation_key": "test:failed-experiment",
            },
        ),
    )
    task.attempt = 1
    task.active_attempt_id = "attempt-failed-1"
    task.status = TaskStatus.FAILED_RETRYABLE
    task.failure_report = FailureReport(
        failure_type=FailureType.CODE_ERROR,
        failed_step="execute_tier_smoke_test",
        error_message="NameError: missing_name is not defined",
        metadata={
            "tier": ExperimentTier.SMOKE_TEST.value,
            "command": ["python", "train.py"],
            "stderr_tail": (
                'Traceback (most recent call last):\n  File "train.py", line 3\n'
                "NameError: missing_name is not defined"
            ),
        },
    )
    return task


def test_experiment_code_error_creates_repair_and_requeues_with_repaired_repo(
    main_agent: MainAgent,
) -> None:
    canary_secret = "sk-private-canary-must-not-be-persisted"
    # Guard against reintroducing the former provider-to-task credential path.
    # Even if an attribute with the old name is present, repair planning must
    # use environment-variable names only.
    main_agent._raw_llm_provider = SimpleNamespace(
        api_base="https://private-gateway.example/v1",
        api_key=canary_secret,
        default_model="private-model",
    )
    failed = _failed_experiment(main_agent)
    main_agent.scheduler.add_tasks([failed])
    main_agent.task_repo.save(failed)

    waiting_for_user = main_agent._handle_failed_task(failed)

    assert waiting_for_user is False
    assert failed.status == TaskStatus.BLOCKED
    repair_id = failed.definition.inputs["code_repair_task_id"]
    repair = main_agent.scheduler.dag.get(repair_id)
    assert repair is not None
    assert repair.definition.task_type == "coding"
    assert repair.definition.inputs["source_failed_task_id"] == failed.task_id
    assert "NameError" in repair.definition.inputs["fix_instructions"]
    assert "REPRO_AGENT_API_KEY" in repair.definition.inputs["fix_instructions"]
    assert canary_secret not in repr(repair.to_dict())
    persisted_repair = main_agent.task_repo.get(repair.task_id)
    assert persisted_repair is not None
    assert canary_secret not in repr(persisted_repair.to_dict())
    assert repair.task_id in failed.dependencies
    assert failed.task_id in main_agent.scheduler.dag.children_of(repair.task_id)

    repair.attempt = 1
    repair.active_attempt_id = "attempt-repair-1"
    repair.status = TaskStatus.SUCCEEDED
    repair_sandbox = main_agent.sandbox_manager.create_sandbox(repair)

    main_agent._resume_experiments_after_code_repair(repair)

    assert failed.status == TaskStatus.PENDING
    assert failed.failure_report is None
    assert failed.definition.inputs["repository_path"] == str(
        (repair_sandbox.workspace_dir / "repository").resolve()
    )
    assert failed.definition.inputs["command"] == ["python", "train.py"]
    assert failed.definition.inputs["applied_code_repair_task_id"] == repair.task_id
    assert "修复后的隔离仓库" in failed.definition.inputs["retry_guidance"]


def test_stdout_syntax_error_is_classified_as_code_error() -> None:
    result = ExperimentExecutionResult(
        tier=ExperimentTier.STATIC_CHECK.value,
        command=["python", "-m", "compileall", "."],
        exit_code=1,
        stdout_tail="SyntaxError: invalid syntax in train.py",
        stderr_tail="",
        termination_reason="completed",
    )

    report = ExperimentExecutionAgent._failure_report_for_unsuccessful_execution(
        result,
        tier=ExperimentTier.STATIC_CHECK,
        metrics_required=False,
    )

    assert report.failure_type == FailureType.CODE_ERROR
    assert report.metadata["command"] == ["python", "-m", "compileall", "."]
    assert "SyntaxError" in report.metadata["stdout_tail"]


def test_code_error_stops_after_experiment_attempt_budget(
    main_agent: MainAgent,
) -> None:
    failed = _failed_experiment(main_agent)
    failed.attempt = failed.definition.max_attempts
    main_agent.scheduler.add_tasks([failed])
    main_agent.task_repo.save(failed)

    main_agent._handle_failed_task(failed)

    assert failed.status == TaskStatus.TERMINAL_FAILURE
    repairs = [
        task
        for task in main_agent.scheduler.dag.all_tasks()
        if task.definition.inputs.get("source_failed_task_id") == failed.task_id
    ]
    assert repairs == []
