from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from repro_agent.agents.paper.agent import PaperAnalysisAgent
from repro_agent.agents.environment.agent import EnvironmentBuildAgent
from repro_agent.agents.experiment.agent import ExperimentExecutionAgent
from repro_agent.domain.enums import ExperimentTier, TaskStatus
from repro_agent.domain.job import JobBudget, JobInputs, ReproductionJob
from repro_agent.domain.task import Task
from repro_agent.evidence.hashing import sha256_of_directory
from repro_agent.execution.backend import ExecutionRequest
from repro_agent.execution.docker import DockerExecutionBackend
from repro_agent.orchestrator.main_agent import MainAgent, MainAgentConfig
from repro_agent.orchestrator.phases import PhaseCoordinator
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.scheduler.scheduler import TaskScheduler
from repro_agent.sandbox.policy import SandboxPolicy
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.sandbox.workspace import TaskSandbox
from repro_agent.schemas.results import TaskResultEnvelope
from repro_agent.storage.database import Database, SCHEMA_VERSION
from repro_agent.storage.repository import JobRepository, TaskRepository
from repro_agent.providers.base import LLMResponse
from repro_agent.tools.base import ToolExecutionError
from repro_agent.tools.write_tools import execute_command


def test_budget_limits_are_hard_boundaries() -> None:
    job = ReproductionJob(
        JobInputs(paper_path="paper", repository_path="repo"),
        budget=JobBudget(max_gpu_hours=1.0),
    )
    job.gpu_hours_used = 1.0

    assert job.budget_exhausted() == (True, "gpu_budget_limit_reached")


def test_untrusted_analysis_cannot_inject_dockerfile_instructions() -> None:
    dockerfile = EnvironmentBuildAgent._generate_dockerfile(
        None,
        "python:3.11-slim",
        "looks fine\nRUN touch /injected",
        ["wheel house; touch /also-injected"],
    )

    assert "RUN touch /injected" not in dockerfile
    assert "'--find-links=/source/wheel house; touch /also-injected'" in dockerfile


def test_model_inferred_high_cost_commands_are_not_marked_verified() -> None:
    job = ReproductionJob(JobInputs(paper_path="paper", repository_path="repo"))
    coordinator = PhaseCoordinator()

    assert not coordinator._tier_command_verified(
        job, [], ExperimentTier.FULL_EXPERIMENT
    )
    job.inputs.user_run_commands = ["true"] * len(list(ExperimentTier))
    assert coordinator._tier_command_verified(
        job, [], ExperimentTier.FULL_EXPERIMENT
    )


def test_missing_dataset_evidence_does_not_get_a_synthetic_digest() -> None:
    assert ExperimentExecutionAgent._digest_evidence([]) == ""


def test_symlinks_are_not_followed_into_host_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-v1", encoding="utf-8")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "external-link").symlink_to(outside)

    digest_before = sha256_of_directory(repository)
    outside.write_text("secret-v2", encoding="utf-8")
    assert sha256_of_directory(repository) == digest_before

    sandbox = TaskSandbox(
        task_id="task",
        attempt_id="attempt",
        root=tmp_path / "sandbox",
        policy=SandboxPolicy(task_id="task"),
    )
    staged = sandbox.stage_input_file(repository)
    assert (staged / "external-link").is_symlink()

    (sandbox.output_dir / "result.json").symlink_to(outside)
    assert "result.json" not in sandbox.collect_outputs()


def test_schema_v4_is_migrated_without_overwriting_tasks(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '4');
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, status TEXT NOT NULL,
            task_type TEXT NOT NULL, payload TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()

    database = Database(db_path)
    try:
        columns = {
            row["name"] for row in database.query_all("PRAGMA table_info(tasks)")
        }
        assert "creation_key" in columns
        assert database.query_one(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        )["value"] == str(SCHEMA_VERSION)
        assert database.query_one(
            "SELECT name FROM sqlite_master WHERE name = 'task_attempts'"
        ) is not None
    finally:
        database.close()


def test_newer_database_schema_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "future.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        f"INSERT INTO schema_meta VALUES ('schema_version', '{SCHEMA_VERSION + 1}');"
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="newer than supported"):
        Database(db_path)


def test_creation_key_attempt_and_lease_are_persisted(tmp_path: Path) -> None:
    database = Database(tmp_path / "agent.db")
    try:
        job = ReproductionJob(JobInputs(paper_path="paper", repository_path="repo"))
        JobRepository(database).save(job)
        repository = TaskRepository(database)
        scheduler = TaskScheduler(job.job_id, repository)

        def make_task() -> Task:
            return Task(
                job_id=job.job_id,
                definition=build_task_definition(
                    objective="idempotent",
                    task_type="specification",
                    inputs={"creation_key": "same-logical-command"},
                    restrict_tools=[],
                ),
            )

        first = scheduler.add_tasks([make_task()])[0]
        replayed = scheduler.add_tasks([make_task()])[0]
        assert len(repository.list_by_job(job.job_id)) == 1
        assert replayed.task_id == first.task_id

        scheduler.refresh_task_states()
        task = scheduler.get_ready_tasks()[0]
        dispatched = scheduler.dispatch([task])[0]
        assert database.query_one(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (dispatched.active_attempt_id,),
        )["status"] == TaskStatus.DISPATCHED.value
        assert database.query_one(
            "SELECT owner FROM task_leases WHERE task_id = ?", (task.task_id,)
        ) is not None

        scheduler.mark_running(dispatched)
        scheduler.mark_succeeded(dispatched, {"result.json": "/tmp/result.json"})
        assert database.query_one(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (dispatched.active_attempt_id,),
        )["status"] == TaskStatus.SUCCEEDED.value
        assert database.query_one(
            "SELECT 1 FROM task_leases WHERE task_id = ?", (task.task_id,)
        ) is None
    finally:
        database.close()


def test_docker_request_honors_workdir_and_gpu(tmp_path: Path) -> None:
    for name in ("input", "workspace", "output"):
        (tmp_path / name).mkdir()
    (tmp_path / "workspace" / "repository").mkdir()
    request = ExecutionRequest(
        task_id="task",
        attempt_id="attempt",
        command=["python", "train.py"],
        image="sha256:" + "a" * 64,
        input_dir=tmp_path / "input",
        workspace_dir=tmp_path / "workspace",
        output_dir=tmp_path / "output",
        working_dir="repository",
        workspace_read_only=True,
        gpu_count=2,
    )

    argv = DockerExecutionBackend().build_run_argv(request)

    assert argv[argv.index("--workdir") + 1] == "/workspace/repository"
    assert argv[argv.index("--gpus") + 1] == "2"
    workspace_mount = argv[argv.index("--mount", argv.index("--mount") + 1) + 1]
    assert "dst=/workspace" in workspace_mount
    assert workspace_mount.endswith(",readonly")
    with pytest.raises(ValueError, match="working_dir"):
        DockerExecutionBackend().build_run_argv(
            replace(request, working_dir="../escape")
        )


def test_task_gpu_request_becomes_a_finite_sandbox_allowance(tmp_path: Path) -> None:
    task = Task(
        job_id="job",
        definition=build_task_definition(
            objective="bounded gpu execution",
            task_type="experiment_execution",
            inputs={"gpu_count": 2},
        ),
    )
    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    limits = sandbox.policy.resource_limits

    assert limits.cpu_cores == 1.0
    assert limits.memory_mb == 1024
    assert limits.disk_mb == 4096
    assert limits.gpu_count == 2
    with pytest.raises(ToolExecutionError, match="exceeds sandbox allowance"):
        execute_command(sandbox, ["true"], gpu_count=3)


def test_paper_result_contract_normalizes_parameters_and_metrics() -> None:
    payload = {
        "parameters": [
            {
                "name": "seed",
                "value": 7,
                "experiment_scope": "main",
                "provenance": "PAPER_EXPLICIT",
            }
        ],
        "expected_results": {
            "accuracy": {
                "value": 0.9,
                "tolerance_type": "absolute",
                "tolerance": 0.01,
            }
        },
        "notes": "ok",
    }

    parameters, method_summary, notes, expected = PaperAnalysisAgent._parse_llm_output(
        None, json.dumps(payload), scope="body"
    )

    assert parameters[0].name == "seed"
    assert method_summary == ""
    assert notes == "ok"
    assert expected["accuracy"]["value"] == 0.9


def test_mock_pipeline_persists_usage_evidence_and_built_image(
    job, work_dir, mock_provider
) -> None:
    agent = MainAgent(
        job,
        MainAgentConfig(
            memory_root=str(work_dir / "memory"),
            sandbox_root=str(work_dir / "sandboxes"),
            snapshot_root=str(work_dir / "snapshots"),
            db_path=str(work_dir / "agent.db"),
            model="mock-model",
            mock_execution=True,
            require_execution_parameter_confirmation=False,
            main_loop_wait_seconds=0.001,
        ),
        mock_provider,
    )
    agent.bootstrap()
    outcome = agent.run_until_finished(max_iterations=500)

    assert outcome.completed
    assert agent.job.model_calls_made > 0
    assert agent.evidence_repo.list_by_job(job.job_id)
    environment_task = next(
        task
        for task in agent.scheduler.dag.all_tasks()
        if task.definition.task_type == "environment_build"
    )
    environment = TaskResultEnvelope.from_file(
        environment_task.outputs["result.json"],
        expected_task_id=environment_task.task_id,
        expected_attempt_id=environment_task.active_attempt_id,
        expected_task_type="environment_build",
    ).payload
    assert environment["build_succeeded"] is True
    assert environment["image_digest"] == "mock"


def test_spec_conflict_requires_explicit_approval_before_environment(
    job, work_dir
) -> None:
    class RoutingProvider:
        def complete(self, messages, params):
            prompt = messages[-1].content
            if "论文正文" in prompt or "附录" in prompt:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "parameters": [
                                {
                                    "name": "learning_rate",
                                    "value": 0.1,
                                    "experiment_scope": "main",
                                    "provenance": "PAPER_EXPLICIT",
                                }
                            ],
                            "expected_results": {
                                "accuracy": {
                                    "value": 0.9,
                                    "tolerance_type": "absolute",
                                    "tolerance": 0.01,
                                }
                            },
                        }
                    )
                )
            if "仓库根目录" in prompt:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "entry_points": ["train.py"],
                            "effective_parameters": {"learning_rate": 0.2},
                        }
                    )
                )
            return LLMResponse(content="{}")

    agent = MainAgent(
        job,
        MainAgentConfig(
            memory_root=str(work_dir / "memory"),
            sandbox_root=str(work_dir / "sandboxes"),
            snapshot_root=str(work_dir / "snapshots"),
            db_path=str(work_dir / "agent.db"),
            model="mock-model",
            mock_execution=True,
            require_execution_parameter_confirmation=False,
            main_loop_wait_seconds=0.001,
        ),
        RoutingProvider(),
    )
    agent.bootstrap()

    paused = agent.run_until_finished(max_iterations=200)

    assert paused.paused is True
    request = agent.pending_intervention()
    assert request is not None
    assert request.metadata["response_mode"] == "spec_conflict"
    assert not any(
        task.definition.task_type == "environment_build"
        and task.status == TaskStatus.SUCCEEDED
        for task in agent.scheduler.dag.all_tasks()
    )

    agent.resolve_intervention(
        request.request_id, {"approve_primary_values": True}
    )
    completed = agent.run_until_finished(max_iterations=500)

    assert completed.completed is True
