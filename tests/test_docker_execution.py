from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from repro_agent.execution.backend import (
    ExecutionRequest,
    ExecutionResourcePolicy,
    ImageBuildRequest,
)
from repro_agent.execution.docker import DockerExecutionBackend, ExecutionUnavailable
from repro_agent.tools.base import ToolExecutionError
from repro_agent.tools.write_tools import git_worktree_apply


def _request(tmp_path: Path) -> ExecutionRequest:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    for path in (input_dir, workspace, output):
        path.mkdir()
    return ExecutionRequest(
        task_id="task_1",
        attempt_id="attempt_1",
        command=["python", "train.py"],
        image="python:3.11@sha256:" + "a" * 64,
        input_dir=input_dir,
        workspace_dir=workspace,
        output_dir=output,
        timeout_seconds=30,
        resources=ExecutionResourcePolicy(cpu_cores=1.5, memory_mb=512, max_processes=24),
    )


def test_docker_argv_is_offline_read_only_and_bounded(tmp_path: Path) -> None:
    backend = DockerExecutionBackend(docker_binary="docker")

    argv = backend.build_run_argv(_request(tmp_path))

    assert argv[:2] == ["docker", "run"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == "512m"
    assert "--memory-swap" in argv and argv[argv.index("--memory-swap") + 1] == "512m"
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "1.5"
    assert "--pids-limit" in argv and argv[argv.index("--pids-limit") + 1] == "24"
    assert "--ulimit" in argv
    assert argv[argv.index("--ulimit") + 1] == "nofile=256:256"
    assert "no-new-privileges" in argv
    assert "ALL" in argv
    assert any("dst=/input" in value and "readonly" in value for value in argv)
    assert all("docker.sock" not in value for value in argv)
    assert argv[-2:] == ["python", "train.py"]


def test_secret_environment_is_passed_by_name_without_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-value-must-not-appear-in-argv"
    monkeypatch.setenv("MODEL_API_KEY", secret)
    request = replace(
        _request(tmp_path), passthrough_environment=["MODEL_API_KEY"]
    )

    argv = DockerExecutionBackend(docker_binary="docker").build_run_argv(request)

    assert "MODEL_API_KEY" in argv
    assert secret not in argv
    assert all(secret not in part for part in argv)


def test_real_execution_blocks_when_docker_is_missing(tmp_path: Path) -> None:
    backend = DockerExecutionBackend(docker_binary="definitely-missing-docker")

    with pytest.raises(ExecutionUnavailable, match="Docker"):
        backend.execute(_request(tmp_path))


def test_git_worktree_tool_never_spawns_host_git() -> None:
    class Context:
        task_id = "task"

    with pytest.raises(ToolExecutionError, match="disabled"):
        git_worktree_apply(Context(), "input://repo", "work", "branch")


class _CompletedNoisyProcess:
    pid = 4321
    returncode = 0

    def __init__(self, argv, *, stdout, stderr):
        stdout.write(b"x" * 64)
        stderr.write(b"y" * 64)
        stdout.flush()
        stderr.flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def terminate(self):
        self.returncode = -15


def test_fast_log_flood_is_bounded_and_fails_closed(tmp_path: Path, monkeypatch) -> None:
    request = replace(
        _request(tmp_path),
        resources=ExecutionResourcePolicy(max_log_bytes=16),
        state_path=tmp_path / "execution.json",
    )
    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "cancel", lambda name: True)
    monkeypatch.setattr(backend, "_resolve_image_digest", lambda image: "")
    monkeypatch.setattr("repro_agent.execution.docker.subprocess.Popen", _CompletedNoisyProcess)

    result = backend.execute(request)

    assert result.exit_code == 137
    assert result.termination_reason == "log_limit_exceeded"
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 16
    assert json.loads(request.state_path.read_text(encoding="utf-8"))["status"] == "TERMINATED"


def test_unconfirmed_container_termination_is_not_reported_as_cancelled(
    tmp_path: Path, monkeypatch
) -> None:
    cancellation = threading.Event()
    cancellation.set()
    request = replace(
        _request(tmp_path),
        cancellation_event=cancellation,
        state_path=tmp_path / "execution.json",
    )

    class RunningProcess(_CompletedNoisyProcess):
        returncode = None

        def __init__(self, argv, *, stdout, stderr):
            self.returncode = None

        def wait(self, timeout=None):
            self.returncode = -9
            return self.returncode

    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "cancel", lambda name: False)
    monkeypatch.setattr(backend, "_resolve_image_digest", lambda image: "")
    monkeypatch.setattr("repro_agent.execution.docker.subprocess.Popen", RunningProcess)

    result = backend.execute(request)

    state = json.loads(request.state_path.read_text(encoding="utf-8"))
    assert result.exit_code == 125
    assert result.termination_reason == "termination_unconfirmed"
    assert state["status"] == "TERMINATION_FAILED"


def test_execution_identity_is_persisted_before_docker_spawn(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "execution.json"
    request = replace(_request(tmp_path), state_path=state_path)

    class InspectingProcess(_CompletedNoisyProcess):
        def __init__(self, argv, *, stdout, stderr):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert state["status"] == "PREPARING"
            assert state["container_name"].startswith("repro-task_1-attempt_1")
            super().__init__(argv, stdout=stdout, stderr=stderr)

    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_resolve_image_digest", lambda image: "")
    monkeypatch.setattr(
        "repro_agent.execution.docker.subprocess.Popen", InspectingProcess
    )

    result = backend.execute(request)
    assert result.exit_code == 0


def test_image_build_output_is_streamed_and_bounded(tmp_path: Path, monkeypatch) -> None:
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_image_exists", lambda image: False)
    monkeypatch.setattr(backend, "_resolve_image_digest", lambda image: "")
    monkeypatch.setattr("repro_agent.execution.docker.subprocess.Popen", _CompletedNoisyProcess)

    result = backend.build_image(
        ImageBuildRequest(
            task_id="task",
            attempt_id="attempt",
            context_dir=context,
            dockerfile=dockerfile,
            image_tag="repro:test",
            max_log_bytes=16,
            log_dir=tmp_path / "logs",
        )
    )

    assert result.exit_code == 137
    assert result.termination_reason == "log_limit_exceeded"
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 16


def test_identical_environment_reuses_content_addressed_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM scratch\nCOPY requirements.lock.txt .\n", encoding="utf-8")
    (context / "requirements.lock.txt").write_text("demo==1.0\n", encoding="utf-8")
    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(
        backend,
        "_image_exists",
        lambda image: image.startswith("repro-agent/env-cache:"),
    )
    monkeypatch.setattr(
        backend,
        "_resolve_image_digest",
        lambda image: "sha256:" + "a" * 64,
    )

    def unexpected_build(*args, **kwargs):
        raise AssertionError("docker build must not run on a cache hit")

    monkeypatch.setattr("repro_agent.execution.docker.subprocess.Popen", unexpected_build)

    result = backend.build_image(
        ImageBuildRequest(
            task_id="second-job",
            attempt_id="attempt-2",
            context_dir=context,
            dockerfile=dockerfile,
            image_tag="repro:second-job",
        )
    )

    assert result.exit_code == 0
    assert result.cache_hit is True
    assert result.image_digest == "sha256:" + "a" * 64
    assert result.cache_ref.startswith("repro-agent/env-cache:")
    assert len(result.environment_fingerprint) == 64


def test_environment_fingerprint_changes_with_build_context(tmp_path: Path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    lockfile = context / "requirements.lock.txt"
    dockerfile.write_text("FROM scratch\nCOPY requirements.lock.txt .\n", encoding="utf-8")
    lockfile.write_text("demo==1.0\n", encoding="utf-8")

    first = DockerExecutionBackend._environment_fingerprint(
        context, [("scratch", "scratch")]
    )
    lockfile.write_text("demo==2.0\n", encoding="utf-8")
    second = DockerExecutionBackend._environment_fingerprint(
        context, [("scratch", "scratch")]
    )
    different_base = DockerExecutionBackend._environment_fingerprint(
        context, [("python:3.11-slim", "sha256:" + "c" * 64)]
    )

    assert first != second
    assert second != different_base


def test_force_rebuild_bypasses_environment_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_image_exists", lambda image: True)
    monkeypatch.setattr(
        backend,
        "_resolve_image_digest",
        lambda image: "sha256:" + "b" * 64,
    )
    captured: dict[str, list[str]] = {}

    class CapturingProcess(_CompletedNoisyProcess):
        def __init__(self, argv, *, stdout, stderr):
            captured["argv"] = argv
            super().__init__(argv, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(
        "repro_agent.execution.docker.subprocess.Popen", CapturingProcess
    )

    result = backend.build_image(
        ImageBuildRequest(
            task_id="task",
            attempt_id="attempt",
            context_dir=context,
            dockerfile=dockerfile,
            image_tag="repro:rebuilt",
            force_rebuild=True,
        )
    )

    assert result.exit_code == 0
    assert result.cache_hit is False
    assert "--no-cache" in captured["argv"]
    tag_indexes = [
        index + 1
        for index, value in enumerate(captured["argv"])
        if value == "--tag"
    ]
    tags = [captured["argv"][index] for index in tag_indexes]
    assert "repro:rebuilt" in tags
    assert result.cache_ref in tags
