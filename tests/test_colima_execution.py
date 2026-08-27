from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from repro_agent.cli.main import _build_config
from repro_agent.execution import ColimaExecutionBackend, DockerExecutionBackend
from repro_agent.execution.docker import ExecutionUnavailable
from repro_agent.orchestrator.main_agent import MainAgent, MainAgentConfig


class _CompletedProbe:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode


def _patch_binaries(monkeypatch: pytest.MonkeyPatch, *, colima=True, docker=True):
    available = {
        "colima": "/opt/homebrew/bin/colima" if colima else None,
        "docker": "/opt/homebrew/bin/docker" if docker else None,
    }
    monkeypatch.setattr(
        "repro_agent.execution.colima.shutil.which",
        lambda binary: available.get(binary),
    )


def test_colima_backend_requires_running_vm_and_reachable_daemon(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(argv)
        return _CompletedProbe(0)

    monkeypatch.setattr("repro_agent.execution.colima.subprocess.run", run)

    backend = ColimaExecutionBackend()

    assert backend.is_available() is True
    assert calls == [["colima", "status"], ["docker", "info"]]


def test_colima_backend_reports_install_command_when_cli_is_missing(monkeypatch) -> None:
    _patch_binaries(monkeypatch, colima=False)
    backend = ColimaExecutionBackend()

    with pytest.raises(ExecutionUnavailable, match="brew install colima docker"):
        backend.require_available(purpose="test execution")


def test_colima_backend_reports_start_command_when_vm_is_stopped(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    monkeypatch.setattr(
        "repro_agent.execution.colima.subprocess.run",
        lambda *args, **kwargs: _CompletedProbe(1),
    )
    backend = ColimaExecutionBackend()

    with pytest.raises(ExecutionUnavailable, match="colima start"):
        backend.require_available(purpose="test execution")


def test_colima_backend_rejects_unreachable_docker_daemon(monkeypatch) -> None:
    _patch_binaries(monkeypatch)
    results = iter([_CompletedProbe(0), _CompletedProbe(1)] * 2)
    monkeypatch.setattr(
        "repro_agent.execution.colima.subprocess.run",
        lambda *args, **kwargs: next(results),
    )
    backend = ColimaExecutionBackend()

    with pytest.raises(ExecutionUnavailable, match="Docker cannot connect"):
        backend.require_available(purpose="test execution")


def test_main_agent_backend_factory_defaults_to_colima() -> None:
    backend = MainAgent._create_execution_backend(MainAgentConfig())
    assert isinstance(backend, ColimaExecutionBackend)

    docker_backend = MainAgent._create_execution_backend(
        MainAgentConfig(container_runtime="docker")
    )
    assert type(docker_backend) is DockerExecutionBackend


def test_cli_config_defaults_to_colima(tmp_path: Path) -> None:
    args = Namespace(
        work_dir=str(tmp_path),
        model="mock-model",
        mock=False,
    )

    config = _build_config(args)

    assert config.container_runtime == "colima"
