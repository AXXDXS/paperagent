from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from repro_agent.agents.environment.agent import EnvironmentBuildAgent
from repro_agent.cli.main import _build_config, build_parser
from repro_agent.execution.backend import (
    CondaEnvironmentBuildRequest,
    ExecutionRequest,
)
from repro_agent.execution.conda import CondaExecutionBackend
from repro_agent.execution.environment_naming import managed_environment_name
from repro_agent.orchestrator.main_agent import MainAgent, MainAgentConfig
from repro_agent.orchestrator.planner import InitialPlanner


def test_environment_name_defaults_to_readable_repository_directory() -> None:
    assert managed_environment_name("", "/projects/E-mem-main") == "e-mem-main"
    assert managed_environment_name("My E-mem", "/projects/ignored") == "my-e-mem"
    assert managed_environment_name("base", "/projects/ignored") == "repro-base"


def test_new_conda_runs_default_to_the_standard_named_environment_directory(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--paper-path",
            str(tmp_path / "paper.txt"),
            "--repository-path",
            str(tmp_path / "repo"),
            "--environment-backend",
            "conda",
        ]
    )

    config = _build_config(args)
    assert Path(config.conda_env_root) == Path.home() / ".conda" / "envs"
    assert config.mirror_policy == ""


def _fake_conda(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-conda"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import subprocess
import sys

args = sys.argv[1:]
if args and args[0] == "create":
    prefix = pathlib.Path(args[args.index("--prefix") + 1])
    python = prefix / ("python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        python.write_text(
            "#!{sys.executable}\\n"
            "import os, sys\\n"
            "if sys.argv[1:] == ['-m', 'pip', 'freeze']:\\n"
            "    raise SystemExit(0)\\n"
            "if sys.argv[1:3] == ['-m', 'pip'] and any(op in sys.argv for op in ('download', 'install')):\\n"
            "    raise SystemExit(0)\\n"
            "os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\\n"
        )
        python.chmod(0o755)
    raise SystemExit(0)
if args[:1] == ["list"]:
    print(json.dumps([{{"name": "python", "version": "3.11"}}]))
    raise SystemExit(0)
if args[:1] == ["run"]:
    prefix_index = args.index("--prefix")
    command = args[prefix_index + 2:]
    raise SystemExit(subprocess.call(command))
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_conda_backend_builds_reuses_and_executes_opaque_environment(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )
    request = CondaEnvironmentBuildRequest(
        task_id="environment",
        attempt_id="attempt-1",
        requirements_file=requirements,
        environment_name="emem",
        python_version="3.11",
        network_enabled=False,
    )

    built = backend.build_conda_environment(request)
    cached = backend.build_conda_environment(request)

    assert built.exit_code == 0
    assert built.environment_ref.startswith("conda://")
    assert built.environment_ref.endswith("/emem")
    assert str(tmp_path) not in built.environment_ref
    assert len(built.environment_digest) == 64
    assert cached.cache_hit is True
    assert cached.environment_ref == built.environment_ref
    assert built.environment_name == "emem"
    assert built.selected_conda_source == "offline"
    assert (tmp_path / "managed-envs" / "emem").is_dir()
    assert not (tmp_path / "managed-envs" / built.environment_fingerprint).exists()

    workspace = tmp_path / "workspace"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    for path in (workspace, input_dir, output_dir):
        path.mkdir()
    execution = backend.execute(
        ExecutionRequest(
            task_id="experiment",
            attempt_id="attempt-2",
            command=["python", "-c", "print('conda runtime ok')"],
            image=built.environment_ref,
            input_dir=input_dir,
            workspace_dir=workspace,
            output_dir=output_dir,
            state_path=tmp_path / "execution.json",
        )
    )

    assert execution.exit_code == 0
    assert execution.stdout.strip() == "conda runtime ok"
    assert execution.image_digest == built.environment_fingerprint
    state = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert state["runtime"] == "conda"
    assert state["container_name"] == execution.container_name
    assert state["status"] == "COMPLETED"

    invalid = ExecutionRequest(
        task_id="experiment",
        attempt_id="attempt-3",
        command=["python", "-c", "print('must not run')"],
        image=built.environment_ref,
        input_dir=input_dir,
        workspace_dir=workspace,
        output_dir=output_dir,
        environment={"HOME": str(tmp_path)},
    )
    try:
        backend.execute(invalid)
    except ValueError as exc:
        assert "not allowlisted" in str(exc)
    else:
        raise AssertionError("reserved host environment overrides must be rejected")


def test_environment_agent_uses_conda_builder_without_generating_dockerfile() -> None:
    agent = object.__new__(EnvironmentBuildAgent)
    agent.task = SimpleNamespace(
        task_id="environment-task",
        definition=SimpleNamespace(
            inputs={
                "repository_path": ".",
                "dependencies_hint": "",
                "environment_backend": "conda",
                "environment_name": "emem",
                "python_version": "3.12",
            }
        ),
    )
    agent._attempt_id = "attempt-1"
    agent._read_dependency_files = lambda root: ({}, [])
    agent._analyze_dependencies = lambda root, hint, files: "standard library"
    agent._generate_lockfile = lambda files, **kwargs: ""
    agent._generate_import_smoke_test = lambda lockfile: "print('ok')\n"
    writes: list[str] = []
    agent._guarded_write_file = lambda path, content: writes.append(path)
    agent.write_json_output = lambda filename, payload: None
    agent.write_candidate_memory = lambda content: None
    agent._run_import_smoke_test = lambda: True
    calls: list[dict] = []

    def build_conda(**kwargs):
        calls.append(kwargs)
        return {
            "environment_ref": "conda://" + "a" * 64 + "/emem",
            "environment_digest": "b" * 64,
            "exit_code": 0,
            "stdout": "created",
            "stderr": "",
            "cache_hit": False,
            "environment_fingerprint": "a" * 64,
            "cache_ref": "conda://" + "a" * 64 + "/emem",
            "package_manifest_digest": "c" * 64,
        }

    agent._build_conda_environment = build_conda
    agent._build_environment_image = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("Docker builder must not be used in Conda mode")
    )

    result = agent.run()

    assert result.succeeded is True
    assert calls == [
        {
            "python_version": "3.12",
            "environment_name": "emem",
            "wheel_dirs": [],
            "force_rebuild": False,
            "repair_existing": False,
            "base_environment_ref": "",
            "network_enabled": True,
        }
    ]
    assert "workspace://Dockerfile" not in writes
    assert result.outputs["environment_backend"] == "conda"
    assert result.outputs["environment_name"] == "emem"
    assert result.outputs["environment_ref"] == "conda://" + "a" * 64 + "/emem"
    assert result.outputs["import_test_passed"] is True


def test_main_agent_factory_and_planner_select_conda_backend(job, tmp_path: Path) -> None:
    job.inputs.environment_name = "emem"
    config = MainAgentConfig(
        environment_backend="conda",
        conda_env_root=str(tmp_path / "envs"),
        conda_executable="conda",
    )

    backend = MainAgent._create_execution_backend(config)
    tasks = InitialPlanner(
        environment_backend="conda",
        conda_python_version="3.12",
    ).plan_initial_tasks(job)
    environment = next(
        task for task in tasks if task.definition.task_type == "environment_build"
    )

    assert isinstance(backend, CondaExecutionBackend)
    assert environment.definition.inputs["environment_backend"] == "conda"
    assert environment.definition.inputs["environment_name"] == "emem"
    assert environment.definition.inputs["python_version"] == "3.12"
    assert "build_conda_environment" in environment.definition.allowed_tools
    assert "build_environment_image" not in environment.definition.allowed_tools


def test_named_conda_environment_rebuilds_in_place_when_fingerprint_changes(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )

    first = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-1",
            requirements_file=requirements,
            environment_name="E Mem Project",
            network_enabled=False,
        )
    )
    second = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-2",
            requirements_file=requirements,
            environment_name="E Mem Project",
            network_enabled=True,
        )
    )

    assert first.environment_name == "e-mem-project"
    assert second.environment_name == "e-mem-project"
    assert second.exit_code == 0, second.stderr
    assert first.environment_ref != second.environment_ref
    assert second.cache_hit is False
    marker = json.loads(
        (tmp_path / "managed-envs" / "e-mem-project" / ".repro_agent_environment.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["environment_fingerprint"] == second.environment_fingerprint
    assert marker["environment_name"] == "e-mem-project"
    assert not (tmp_path / "managed-envs" / first.environment_fingerprint).exists()


def test_named_conda_environment_migrates_legacy_hash_prefix_without_rebuild(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )
    legacy_request = CondaEnvironmentBuildRequest(
        task_id="environment",
        attempt_id="attempt-1",
        requirements_file=requirements,
        network_enabled=False,
    )
    legacy = backend.build_conda_environment(legacy_request)
    generated_prefix = tmp_path / "managed-envs" / "environment"
    legacy_prefix = tmp_path / "managed-envs" / legacy.environment_fingerprint
    generated_prefix.rename(legacy_prefix)
    assert legacy_prefix.is_dir()

    named = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-2",
            requirements_file=requirements,
            environment_name="emem",
            network_enabled=False,
        )
    )

    assert named.cache_hit is True
    assert named.environment_fingerprint == legacy.environment_fingerprint
    assert named.environment_ref.endswith("/emem")
    assert named.environment_name == "emem"
    assert (tmp_path / "managed-envs" / "emem").is_dir()
    assert not legacy_prefix.exists()


def test_name_bound_reference_never_resolves_to_duplicate_prefix(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )
    built = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-1",
            requirements_file=requirements,
            environment_name="target-env",
            network_enabled=False,
        )
    )
    duplicate = tmp_path / "managed-envs" / "aaa-duplicate"
    shutil.copytree(tmp_path / "managed-envs" / "target-env", duplicate)

    assert backend._prefix_from_ref(built.environment_ref).name == "target-env"


def test_legacy_reference_prefers_the_stable_name_over_attempt_derived_duplicates(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )
    built = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-1",
            requirements_file=requirements,
            environment_name="e-mem-main",
            network_enabled=False,
        )
    )
    shutil.copytree(
        tmp_path / "managed-envs" / "e-mem-main",
        tmp_path / "managed-envs" / "7cdcd1700e145b40-0-repository",
    )
    legacy_ref = "conda://" + built.environment_fingerprint

    assert backend._prefix_from_ref(legacy_ref).name == "e-mem-main"


def test_runtime_dependency_repair_updates_existing_named_prefix_without_conda_create(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )
    commands: list[list[str]] = []
    original_run = backend._run_build_command

    def record_commands(command, **kwargs):
        commands.append(list(command))
        if "pip" in command and any(
            operation in command for operation in ("download", "install")
        ):
            return backend._CommandResult(0, "package operation ok", "", "completed")
        return original_run(command, **kwargs)

    backend._run_build_command = record_commands
    backend._provision_prefix = lambda prefix, request, **kwargs: ("", "", [], "")
    backend._package_manifest = lambda prefix, timeout: json.dumps(
        {"prefix": prefix.name, "requirements": requirements.read_text()}
    )
    initial = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-1",
            requirements_file=requirements,
            environment_name="emem",
            network_enabled=True,
        )
    )
    requirements.write_text("requests\n", encoding="utf-8")
    repaired = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment-repair",
            attempt_id="attempt-2",
            requirements_file=requirements,
            environment_name="emem",
            repair_existing=True,
            base_environment_ref=initial.environment_ref,
            network_enabled=True,
        )
    )

    create_commands = [command for command in commands if command[1:2] == ["create"]]
    assert len(create_commands) == 1
    assert repaired.exit_code == 0
    assert repaired.environment_name == "emem"
    assert repaired.environment_ref.endswith("/emem")
    assert repaired.environment_ref != initial.environment_ref
    assert (tmp_path / "managed-envs" / "emem").is_dir()


def test_named_build_refuses_to_delete_an_unmanaged_conda_environment(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    unmanaged = tmp_path / "managed-envs" / "emem"
    unmanaged.mkdir(parents=True)
    sentinel = unmanaged / "user-data.txt"
    sentinel.write_text("keep", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )

    try:
        backend.build_conda_environment(
            CondaEnvironmentBuildRequest(
                task_id="environment",
                attempt_id="attempt-1",
                requirements_file=requirements,
                environment_name="emem",
                network_enabled=False,
            )
        )
    except RuntimeError as exc:
        assert "unmanaged Conda environment" in str(exc)
    else:
        raise AssertionError("unmanaged environment must not be replaced")
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_failed_rebuild_restores_the_previous_managed_environment(
    tmp_path: Path,
) -> None:
    conda = _fake_conda(tmp_path)
    requirements = tmp_path / "requirements.lock.txt"
    requirements.write_text("", encoding="utf-8")
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "managed-envs",
        conda_binary=str(conda),
    )
    original = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-1",
            requirements_file=requirements,
            environment_name="emem",
            network_enabled=False,
        )
    )
    original_marker = (
        tmp_path / "managed-envs" / "emem" / ".repro_agent_environment.json"
    ).read_text(encoding="utf-8")
    backend._create_prefix_with_failover = lambda **kwargs: (
        backend._CommandResult(1, "", "CondaHTTPError: HTTP 503", "completed"),
        "tuna-main",
        [],
    )
    failed = backend.build_conda_environment(
        CondaEnvironmentBuildRequest(
            task_id="environment",
            attempt_id="attempt-2",
            requirements_file=requirements,
            environment_name="emem",
            force_rebuild=True,
            network_enabled=True,
        )
    )

    assert failed.exit_code == 1
    assert backend._prefix_from_ref(original.environment_ref).name == "emem"
    assert (
        tmp_path / "managed-envs" / "emem" / ".repro_agent_environment.json"
    ).read_text(encoding="utf-8") == original_marker
    assert not list((tmp_path / "managed-envs").glob(".repro-backup-*"))


def test_resume_infers_persisted_conda_backend(job, tmp_path: Path, monkeypatch) -> None:
    from repro_agent.storage.database import Database
    from repro_agent.storage.repository import JobRepository, TaskRepository

    database_path = tmp_path / "state.db"
    database = Database(database_path)
    try:
        JobRepository(database).save(job)
        environment = next(
            task
            for task in InitialPlanner(
                environment_backend="conda", conda_python_version="3.12"
            ).plan_initial_tasks(job)
            if task.definition.task_type == "environment_build"
        )
        TaskRepository(database).save(environment)
    finally:
        database.close()

    captured: dict[str, MainAgentConfig] = {}

    def capture_init(self, resumed_job, config, provider):
        captured["config"] = config

    monkeypatch.setattr(MainAgent, "__init__", capture_init)
    MainAgent.resume_from_storage(
        job.job_id,
        MainAgentConfig(
            db_path=str(database_path),
            environment_backend="",
            conda_env_root=str(tmp_path / "envs"),
        ),
        object(),
    )

    assert captured["config"].environment_backend == "conda"
    assert captured["config"].conda_python_version == "3.12"
