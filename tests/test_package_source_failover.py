from __future__ import annotations

from pathlib import Path

import pytest

from repro_agent.execution.backend import CondaEnvironmentBuildRequest
from repro_agent.execution.conda import CondaExecutionBackend
from repro_agent.execution.package_sources import (
    PackageSource,
    PackageSourcePolicy,
    SourceFailureKind,
    classify_source_failure,
)


def _request(tmp_path: Path) -> CondaEnvironmentBuildRequest:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("requests\n", encoding="utf-8")
    return CondaEnvironmentBuildRequest(
        task_id="environment",
        attempt_id="attempt-1",
        requirements_file=requirements,
        environment_name="test-env",
        network_enabled=True,
        timeout_seconds=60,
    )


def _policy() -> PackageSourcePolicy:
    return PackageSourcePolicy(
        mode="auto",
        pip_sources=(
            PackageSource("pip-one", "https://one.example/simple"),
            PackageSource("pip-two", "https://two.example/simple"),
        ),
        conda_sources=(
            PackageSource("conda-one", "https://one.example/conda"),
            PackageSource("conda-two", "https://two.example/conda"),
        ),
    )


def test_source_failure_classification_is_deterministic() -> None:
    assert classify_source_failure("CondaHTTPError: HTTP 503") == (
        SourceFailureKind.UNAVAILABLE
    )
    assert classify_source_failure("PackagesNotFoundError: numpy") == (
        SourceFailureKind.PACKAGE_NOT_FOUND
    )
    assert classify_source_failure("UnsatisfiableError: conflict") == (
        SourceFailureKind.DEPENDENCY_CONFLICT
    )
    assert classify_source_failure("HTTP 401 Unauthorized") == (
        SourceFailureKind.AUTHENTICATION
    )
    assert classify_source_failure("No space left on device") == (
        SourceFailureKind.LOCAL_FAILURE
    )


def test_private_source_never_falls_back_to_public_automatically() -> None:
    policy = _policy()
    private = PackageSource(
        "private", "https://packages.example.internal/simple", private=True
    )
    assert policy.may_failover(
        source=private, failure_kind=SourceFailureKind.UNAVAILABLE
    ) is False
    assert policy.may_failover(
        source=private, failure_kind=SourceFailureKind.AUTHENTICATION
    ) is False


def test_environment_source_configuration_requires_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPRO_AGENT_PIP_INDEX_URL", "http://unsafe.example/simple")
    with pytest.raises(ValueError, match="HTTPS"):
        PackageSourcePolicy.from_environment(network_enabled=True)


def test_legacy_single_pip_source_remains_the_first_configured_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "REPRO_AGENT_PIP_INDEX_URL", "https://packages.example.internal/simple"
    )
    policy = PackageSourcePolicy.from_environment(network_enabled=True)

    assert policy.pip_sources[0].location == (
        "https://packages.example.internal/simple"
    )
    assert policy.pip_sources[0].private is True


def test_source_policy_is_bounded_and_offline_mode_has_no_remote_sources() -> None:
    policy = PackageSourcePolicy.from_settings(
        network_enabled=True,
        mode="auto",
        pip_index_urls=[f"https://pip-{index}.example/simple" for index in range(8)],
        conda_channels=[f"https://conda-{index}.example/main" for index in range(8)],
    )
    assert len(policy.pip_sources) == 4
    assert len(policy.conda_sources) == 3

    offline = PackageSourcePolicy.from_settings(
        network_enabled=True, mode="offline"
    )
    assert offline.pip_sources == ()
    assert offline.conda_sources == ()


def test_source_urls_cannot_embed_credentials() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        PackageSourcePolicy.from_settings(
            network_enabled=True,
            pip_index_urls=["https://user:token@packages.example/simple"],
        )


def test_conda_create_switches_only_after_a_source_failure(tmp_path: Path) -> None:
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "envs", conda_binary="conda"
    )
    request = _request(tmp_path)
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(list(command))
        channel = command[command.index("--channel") + 1]
        if "one.example" in channel:
            return backend._CommandResult(
                1, "", "CondaHTTPError: HTTP 503", "completed"
            )
        return backend._CommandResult(0, "created", "", "completed")

    backend._run_build_command = run
    result, selected, attempts = backend._create_prefix_with_failover(
        prefix=tmp_path / "envs" / "test-env",
        request=request,
        policy=_policy(),
        deadline=10**12,
    )

    assert result.exit_code == 0
    assert selected == "conda-two"
    assert [attempt["source"] for attempt in attempts] == [
        "conda-one",
        "conda-two",
    ]
    assert len(commands) == 2


def test_conda_dependency_conflict_does_not_switch_source(tmp_path: Path) -> None:
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "envs", conda_binary="conda"
    )
    request = _request(tmp_path)
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(list(command))
        return backend._CommandResult(
            1, "", "UnsatisfiableError: dependency conflict", "completed"
        )

    backend._run_build_command = run
    result, selected, attempts = backend._create_prefix_with_failover(
        prefix=tmp_path / "envs" / "test-env",
        request=request,
        policy=_policy(),
        deadline=10**12,
    )

    assert result.exit_code == 1
    assert selected == "conda-one"
    assert len(attempts) == 1
    assert len(commands) == 1


def test_pip_download_switches_then_installs_offline_once(tmp_path: Path) -> None:
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "envs", conda_binary="conda"
    )
    request = _request(tmp_path)
    prefix = tmp_path / "envs" / "test-env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(list(command))
        if "download" in command and "one.example" in " ".join(command):
            return backend._CommandResult(
                1, "", "ReadTimeout: read timed out", "completed"
            )
        return backend._CommandResult(0, "ok", "", "completed")

    backend._run_build_command = run
    result, selected, attempts = backend._install_requirements_with_failover(
        prefix=prefix,
        request=request,
        policy=_policy(),
        deadline=10**12,
    )

    assert result.exit_code == 0
    assert selected == "pip-two"
    assert [attempt["phase"] for attempt in attempts] == [
        "pip_download",
        "pip_download",
        "pip_install_offline",
    ]
    install_command = commands[-1]
    assert "install" in install_command
    assert "--no-index" in install_command
    assert "--index-url" not in install_command


def test_uv_provisioning_uses_the_same_bounded_failover(tmp_path: Path) -> None:
    backend = CondaExecutionBackend(
        environment_root=tmp_path / "envs", conda_binary="conda"
    )
    request = _request(tmp_path)
    prefix = tmp_path / "envs" / "test-env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("", encoding="utf-8")

    def run(command, **kwargs):
        if "download" in command and "one.example" in " ".join(command):
            return backend._CommandResult(1, "", "HTTP 503", "completed")
        return backend._CommandResult(0, "ok", "", "completed")

    backend._run_build_command = run
    result, selected, attempts = backend._install_uv_with_failover(
        prefix=prefix,
        request=request,
        policy=_policy(),
        deadline=10**12,
        preferred_source_id="pip-one",
    )

    assert result.exit_code == 0
    assert selected == "pip-two"
    assert [attempt["phase"] for attempt in attempts] == [
        "uv_download",
        "uv_download",
        "uv_install_offline",
    ]
