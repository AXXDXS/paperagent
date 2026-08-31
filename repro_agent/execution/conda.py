"""Trusted-local Conda execution backend.

The backend keeps ReproAgent's task workspace/input/output separation, but it
does not claim to provide a container security boundary.  Generated Conda
prefixes live under one controller-owned root and are addressed by opaque,
name-bound ``conda://<fingerprint>/<environment-name>`` references rather than
persisted host paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from repro_agent.execution.backend import (
    CondaEnvironmentBuildRequest,
    CondaEnvironmentBuildResult,
    ExecutionRequest,
    ExecutionResult,
    ImageBuildRequest,
    ImageBuildResult,
)
from repro_agent.execution.environment_naming import managed_environment_name
from repro_agent.execution.package_sources import (
    PackageSource,
    PackageSourcePolicy,
    SourceFailureKind,
    classify_source_failure,
    source_attempt,
)

_CONDA_REF_RE = re.compile(
    r"^conda://([a-f0-9]{64})(?:/([a-z0-9][a-z0-9._-]{0,62}))?$"
)


class CondaExecutionBackend:
    """Create cached Conda prefixes and run commands through ``conda run``.

    This backend is intended for trusted local repositories.  It preserves
    command timeouts, cancellation, bounded logs and per-task workspaces, but
    Conda itself is not a replacement for Docker's network/filesystem/resource
    isolation.
    """

    def __init__(
        self,
        *,
        environment_root: str | Path,
        conda_binary: str = "conda",
        mirror_policy: str = "",
        pip_index_urls: tuple[str, ...] = (),
        conda_channels: tuple[str, ...] = (),
    ) -> None:
        self.environment_root = Path(environment_root).expanduser().resolve()
        self.environment_root.mkdir(parents=True, exist_ok=True)
        self.conda_binary = conda_binary
        self.mirror_policy = mirror_policy
        self.pip_index_urls = pip_index_urls
        self.conda_channels = conda_channels
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._source_unhealthy_until: dict[str, float] = {}

    def is_available(self) -> bool:
        return shutil.which(self.conda_binary) is not None

    def require_available(self, *, purpose: str = "Conda execution") -> None:
        if not self.is_available():
            raise RuntimeError(
                f"Conda executable '{self.conda_binary}' was not found on PATH; "
                f"it is required for {purpose}."
            )

    def build_image(self, request: ImageBuildRequest) -> ImageBuildResult:
        raise RuntimeError("Conda backend does not build Docker images")

    def build_conda_environment(
        self, request: CondaEnvironmentBuildRequest
    ) -> CondaEnvironmentBuildResult:
        self.require_available(purpose="environment creation")
        source_policy = (
            PackageSourcePolicy.from_settings(
                network_enabled=request.network_enabled,
                mode=self.mirror_policy or "auto",
                pip_index_urls=self.pip_index_urls,
                conda_channels=self.conda_channels,
            )
            if self.mirror_policy or self.pip_index_urls or self.conda_channels
            else PackageSourcePolicy.from_environment(
                network_enabled=request.network_enabled
            )
        )
        deadline = time.monotonic() + request.timeout_seconds
        source_attempts: list[dict[str, object]] = []
        requirements = request.requirements_file.read_bytes()
        wheel_fingerprints = self._wheel_fingerprints(request.wheel_dirs)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "format": 1,
                    "python_version": request.python_version,
                    "requirements_sha256": hashlib.sha256(requirements).hexdigest(),
                    "wheels": wheel_fingerprints,
                    "network_enabled": request.network_enabled,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        environment_name = managed_environment_name(
            request.environment_name, request.task_id
        )
        prefix = self._prefix_for_environment_name(environment_name)
        marker = prefix / ".repro_agent_environment.json"
        environment_ref = self._environment_ref(fingerprint, environment_name)

        with self._build_lock:
            if request.repair_existing:
                return self._repair_existing_environment(
                    request=request,
                    prefix=prefix,
                    marker=marker,
                    environment_name=environment_name,
                    fingerprint=fingerprint,
                    environment_ref=environment_ref,
                    source_policy=source_policy,
                    deadline=deadline,
                )
            # One-time compatibility migration: an environment created by the
            # previous hash-directory implementation can be promoted to the
            # new readable project name without reinstalling dependencies.
            if not prefix.exists():
                legacy_prefix = self._prefix_for_fingerprint(fingerprint)
                legacy_marker = legacy_prefix / ".repro_agent_environment.json"
                if self._valid_cache_marker(legacy_marker, fingerprint):
                    try:
                        legacy_prefix.rename(prefix)
                    except OSError:
                        pass
                    else:
                        metadata = json.loads(marker.read_text(encoding="utf-8"))
                        metadata["environment_name"] = environment_name
                        marker.write_text(
                            json.dumps(metadata, sort_keys=True), encoding="utf-8"
                        )
            if not request.force_rebuild and self._valid_cache_marker(marker, fingerprint):
                metadata = json.loads(marker.read_text(encoding="utf-8"))
                selected_pip_source = str(metadata.get("selected_pip_source", ""))
                provision_stdout, provision_stderr, provision_attempts, uv_source = self._provision_prefix(
                    prefix,
                    request,
                    source_policy=source_policy,
                    deadline=deadline,
                    preferred_source_id=selected_pip_source,
                )
                if not selected_pip_source and uv_source:
                    selected_pip_source = uv_source
                    metadata["selected_pip_source"] = uv_source
                    marker.write_text(
                        json.dumps(metadata, sort_keys=True), encoding="utf-8"
                    )
                digest = str(metadata.get("environment_digest", ""))
                return CondaEnvironmentBuildResult(
                    environment_ref=environment_ref,
                    environment_digest=digest,
                    exit_code=0,
                    stdout=(
                        f"reused cached Conda environment "
                        f"{environment_name} ({environment_ref})\n{provision_stdout}"
                    )[-request.max_log_bytes :],
                    stderr=provision_stderr[-request.max_log_bytes :],
                    cache_hit=True,
                    environment_fingerprint=fingerprint,
                    cache_ref=environment_ref,
                    package_manifest_digest=str(
                        metadata.get("package_manifest_digest", digest)
                    ),
                    environment_name=str(
                        metadata.get("environment_name", environment_name)
                    ),
                    selected_conda_source=str(
                        metadata.get("selected_conda_source", "")
                    ),
                    selected_pip_source=selected_pip_source,
                    source_attempts=provision_attempts,
                )

            backup_prefix = self._backup_managed_prefix(
                prefix, attempt_id=request.attempt_id
            )
            prefix.parent.mkdir(parents=True, exist_ok=True)

            create, selected_conda_source, conda_attempts = (
                self._create_prefix_with_failover(
                    prefix=prefix,
                    request=request,
                    policy=source_policy,
                    deadline=deadline,
                )
            )
            source_attempts.extend(conda_attempts)
            stdout = create.stdout
            stderr = create.stderr
            if create.exit_code != 0:
                self._remove_managed_prefix(prefix, allow_unmarked=True)
                self._restore_managed_backup(prefix, backup_prefix)
                return CondaEnvironmentBuildResult(
                    environment_ref=environment_ref,
                    environment_digest="",
                    exit_code=create.exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    termination_reason=create.termination_reason,
                    environment_fingerprint=fingerprint,
                    cache_ref=environment_ref,
                    environment_name=environment_name,
                    selected_conda_source=selected_conda_source,
                    source_attempts=source_attempts,
                )

            selected_pip_source = ""
            if requirements.strip():
                install, selected_pip_source, pip_attempts = (
                    self._install_requirements_with_failover(
                        prefix=prefix,
                        request=request,
                        policy=source_policy,
                        deadline=deadline,
                    )
                )
                source_attempts.extend(pip_attempts)
                stdout = (stdout + "\n" + install.stdout)[-request.max_log_bytes :]
                stderr = (stderr + "\n" + install.stderr)[-request.max_log_bytes :]
                if install.exit_code != 0:
                    self._remove_managed_prefix(prefix, allow_unmarked=True)
                    self._restore_managed_backup(prefix, backup_prefix)
                    return CondaEnvironmentBuildResult(
                        environment_ref=environment_ref,
                        environment_digest="",
                        exit_code=install.exit_code,
                        stdout=stdout,
                        stderr=stderr,
                        termination_reason=install.termination_reason,
                        environment_fingerprint=fingerprint,
                        cache_ref=environment_ref,
                        environment_name=environment_name,
                        selected_conda_source=selected_conda_source,
                        selected_pip_source=selected_pip_source,
                        source_attempts=source_attempts,
                    )

            provision_stdout, provision_stderr, provision_attempts, uv_source = self._provision_prefix(
                prefix,
                request,
                source_policy=source_policy,
                deadline=deadline,
                preferred_source_id=selected_pip_source,
            )
            source_attempts.extend(provision_attempts)
            selected_pip_source = selected_pip_source or uv_source
            stdout = (stdout + "\n" + provision_stdout)[-request.max_log_bytes :]
            stderr = (stderr + "\n" + provision_stderr)[-request.max_log_bytes :]

            try:
                manifest_timeout = int(deadline - time.monotonic())
                if manifest_timeout <= 0:
                    raise subprocess.TimeoutExpired("package manifest", 0)
                manifest = self._package_manifest(prefix, manifest_timeout)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                self._remove_managed_prefix(prefix, allow_unmarked=True)
                self._restore_managed_backup(prefix, backup_prefix)
                return CondaEnvironmentBuildResult(
                    environment_ref=environment_ref,
                    environment_digest="",
                    exit_code=125,
                    stdout=stdout,
                    stderr=(stderr + f"\nfailed to capture package manifest: {exc}")[
                        -request.max_log_bytes :
                    ],
                    termination_reason="manifest_capture_failed",
                    environment_fingerprint=fingerprint,
                    cache_ref=environment_ref,
                    environment_name=environment_name,
                    selected_conda_source=selected_conda_source,
                    selected_pip_source=selected_pip_source,
                    source_attempts=source_attempts,
                )
            manifest_digest = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
            environment_digest = hashlib.sha256(
                f"{fingerprint}:{manifest_digest}".encode("utf-8")
            ).hexdigest()
            marker.write_text(
                json.dumps(
                    {
                        "environment_fingerprint": fingerprint,
                        "environment_digest": environment_digest,
                        "package_manifest_digest": manifest_digest,
                        "environment_name": environment_name,
                        "selected_conda_source": selected_conda_source,
                        "selected_pip_source": selected_pip_source,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._discard_managed_backup(prefix, backup_prefix)
            return CondaEnvironmentBuildResult(
                environment_ref=environment_ref,
                environment_digest=environment_digest,
                exit_code=0,
                stdout=stdout,
                stderr=stderr,
                environment_fingerprint=fingerprint,
                cache_ref=environment_ref,
                package_manifest_digest=manifest_digest,
                environment_name=environment_name,
                selected_conda_source=selected_conda_source,
                selected_pip_source=selected_pip_source,
                source_attempts=source_attempts,
            )

    def _create_prefix_with_failover(
        self,
        *,
        prefix: Path,
        request: CondaEnvironmentBuildRequest,
        policy: PackageSourcePolicy,
        deadline: float,
    ) -> tuple[_CommandResult, str, list[dict[str, object]]]:
        if not request.network_enabled:
            command = self._conda_create_command(prefix, request)
            command.append("--offline")
            result = self._run_with_deadline(command, request, deadline)
            return result, "offline", []

        attempts: list[dict[str, object]] = []
        logs: list[str] = []
        last_result = self._CommandResult(
            125, "", "no approved Conda package source", "source_exhausted"
        )
        selected_source = ""
        for source in self._ordered_sources(policy.conda_sources):
            if prefix.exists():
                self._remove_managed_prefix(prefix, allow_unmarked=True)
            command = self._conda_create_command(prefix, request)
            command[2:2] = ["--override-channels", "--channel", source.location]
            result = self._run_with_deadline(
                command, request, deadline, max_seconds=300
            )
            failure_kind = (
                None
                if result.exit_code == 0
                else self._classify_attempt_failure(result, deadline)
            )
            attempts.append(
                source_attempt(
                    source,
                    phase="conda_create",
                    exit_code=result.exit_code,
                    failure_kind=failure_kind,
                )
            )
            logs.append(self._source_log(source, result, failure_kind))
            last_result = result
            selected_source = source.source_id
            if result.exit_code == 0:
                self._source_unhealthy_until.pop(source.source_id, None)
                return (
                    self._with_combined_logs(result, logs, request.max_log_bytes),
                    selected_source,
                    attempts,
                )
            may_failover = policy.may_failover(
                source=source,
                failure_kind=failure_kind or SourceFailureKind.UNKNOWN,
            )
            if may_failover:
                self._source_unhealthy_until[source.source_id] = (
                    time.monotonic() + 600
                )
            if not may_failover:
                break
        if prefix.exists():
            self._remove_managed_prefix(prefix, allow_unmarked=True)
        return (
            self._with_combined_logs(last_result, logs, request.max_log_bytes),
            selected_source,
            attempts,
        )

    def _install_requirements_with_failover(
        self,
        *,
        prefix: Path,
        request: CondaEnvironmentBuildRequest,
        policy: PackageSourcePolicy,
        deadline: float,
    ) -> tuple[_CommandResult, str, list[dict[str, object]]]:
        if not request.network_enabled:
            command = self._offline_pip_install_command(
                prefix, request, download_dir=None
            )
            result = self._run_with_deadline(command, request, deadline)
            return result, "offline", []

        attempts: list[dict[str, object]] = []
        logs: list[str] = []
        last_result = self._CommandResult(
            125, "", "no approved pip package source", "source_exhausted"
        )
        selected_source = ""
        for source in self._ordered_sources(policy.pip_sources):
            with tempfile.TemporaryDirectory(
                prefix=".repro-download-", dir=self.environment_root
            ) as download_root:
                download_dir = Path(download_root)
                command = [
                    str(self._python_executable(prefix)),
                    "-m",
                    "pip",
                    "--isolated",
                    "download",
                    "--dest",
                    str(download_dir),
                    "--retries",
                    "2",
                    "--timeout",
                    "30",
                    "--index-url",
                    source.location,
                ]
                for wheel_dir in request.wheel_dirs:
                    command.extend(["--find-links", str(wheel_dir)])
                command.extend(["-r", str(request.requirements_file)])
                downloaded = self._run_with_deadline(
                    command, request, deadline, max_seconds=600
                )
                failure_kind = (
                    None
                    if downloaded.exit_code == 0
                    else self._classify_attempt_failure(downloaded, deadline)
                )
                attempts.append(
                    source_attempt(
                        source,
                        phase="pip_download",
                        exit_code=downloaded.exit_code,
                        failure_kind=failure_kind,
                    )
                )
                logs.append(self._source_log(source, downloaded, failure_kind))
                last_result = downloaded
                selected_source = source.source_id
                if downloaded.exit_code == 0:
                    self._source_unhealthy_until.pop(source.source_id, None)
                    install_command = self._offline_pip_install_command(
                        prefix, request, download_dir=download_dir
                    )
                    installed = self._run_with_deadline(
                        install_command, request, deadline
                    )
                    install_failure = (
                        None
                        if installed.exit_code == 0
                        else classify_source_failure(
                            installed.stderr, installed.stdout
                        )
                    )
                    attempts.append(
                        source_attempt(
                            source,
                            phase="pip_install_offline",
                            exit_code=installed.exit_code,
                            failure_kind=install_failure,
                        )
                    )
                    logs.append(
                        self._source_log(source, installed, install_failure)
                    )
                    return (
                        self._with_combined_logs(
                            installed, logs, request.max_log_bytes
                        ),
                        selected_source,
                        attempts,
                    )
                may_failover = policy.may_failover(
                    source=source,
                    failure_kind=failure_kind or SourceFailureKind.UNKNOWN,
                )
                if may_failover:
                    self._source_unhealthy_until[source.source_id] = (
                        time.monotonic() + 600
                    )
                if not may_failover:
                    break
        return (
            self._with_combined_logs(last_result, logs, request.max_log_bytes),
            selected_source,
            attempts,
        )

    def _conda_create_command(
        self, prefix: Path, request: CondaEnvironmentBuildRequest
    ) -> list[str]:
        return [
            self.conda_binary,
            "create",
            "--yes",
            "--prefix",
            str(prefix),
            f"python={request.python_version}",
            "pip",
        ]

    def _offline_pip_install_command(
        self,
        prefix: Path,
        request: CondaEnvironmentBuildRequest,
        *,
        download_dir: Path | None,
    ) -> list[str]:
        command = [
            str(self._python_executable(prefix)),
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-cache-dir",
            "--no-index",
        ]
        if download_dir is not None:
            command.extend(["--find-links", str(download_dir)])
        for wheel_dir in request.wheel_dirs:
            command.extend(["--find-links", str(wheel_dir)])
        command.extend(["-r", str(request.requirements_file)])
        return command

    def _run_with_deadline(
        self,
        command: list[str],
        request: CondaEnvironmentBuildRequest,
        deadline: float,
        max_seconds: int | None = None,
    ) -> _CommandResult:
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            return self._CommandResult(
                124, "", "environment build deadline exhausted", "timeout_killed"
            )
        timeout_seconds = max(1, remaining)
        if max_seconds is not None:
            timeout_seconds = min(timeout_seconds, max_seconds)
        return self._run_build_command(
            command,
            timeout_seconds=timeout_seconds,
            cancellation_event=request.cancellation_event,
            max_log_bytes=request.max_log_bytes,
        )

    @staticmethod
    def _classify_attempt_failure(
        result: _CommandResult, deadline: float
    ) -> SourceFailureKind:
        if result.termination_reason in {
            "cancelled_by_controller",
            "log_limit_exceeded",
        }:
            return SourceFailureKind.LOCAL_FAILURE
        if result.termination_reason == "timeout_killed":
            return (
                SourceFailureKind.UNAVAILABLE
                if time.monotonic() < deadline
                else SourceFailureKind.LOCAL_FAILURE
            )
        return classify_source_failure(result.stderr, result.stdout)

    @staticmethod
    def _source_location(
        sources: tuple[PackageSource, ...], source_id: str
    ) -> str:
        selected = next(
            (source.location for source in sources if source.source_id == source_id),
            "",
        )
        return selected or (sources[0].location if sources else "")

    def _ordered_sources(
        self, sources: tuple[PackageSource, ...]
    ) -> tuple[PackageSource, ...]:
        now = time.monotonic()
        return tuple(
            sorted(
                sources,
                key=lambda source: (
                    self._source_unhealthy_until.get(source.source_id, 0) > now,
                    sources.index(source),
                ),
            )
        )

    @staticmethod
    def _source_log(
        source: PackageSource,
        result: _CommandResult,
        failure_kind: SourceFailureKind | None,
    ) -> str:
        status = "success" if result.exit_code == 0 else (
            failure_kind.value if failure_kind else "failed"
        )
        return f"package source {source.source_id} ({source.display_location}): {status}"

    @classmethod
    def _with_combined_logs(
        cls,
        result: _CommandResult,
        logs: list[str],
        max_log_bytes: int,
    ) -> _CommandResult:
        summary = "\n".join(logs)
        return cls._CommandResult(
            result.exit_code,
            (summary + "\n" + result.stdout)[-max_log_bytes:],
            (summary + "\n" + result.stderr)[-max_log_bytes:],
            result.termination_reason,
        )

    def _repair_existing_environment(
        self,
        *,
        request: CondaEnvironmentBuildRequest,
        prefix: Path,
        marker: Path,
        environment_name: str,
        fingerprint: str,
        environment_ref: str,
        source_policy: PackageSourcePolicy,
        deadline: float,
    ) -> CondaEnvironmentBuildResult:
        """Install a dependency delta into one existing named environment.

        The old marker is left untouched until installation, provisioning and
        manifest capture all succeed.  Most importantly, this path never calls
        ``conda create`` and never allocates an attempt-derived prefix.
        """

        base_match = _CONDA_REF_RE.fullmatch(request.base_environment_ref)
        if base_match is None:
            raise ValueError(
                "repair_existing requires a valid base_environment_ref"
            )
        base_fingerprint, referenced_name = base_match.groups()
        if referenced_name and referenced_name != environment_name:
            raise ValueError(
                "base environment reference does not match environment_name"
            )
        if not self._valid_cache_marker(marker, base_fingerprint):
            raise RuntimeError(
                f"existing Conda environment is unavailable or stale: {environment_name}"
            )

        previous_metadata = json.loads(marker.read_text(encoding="utf-8"))
        selected_conda_source = str(
            previous_metadata.get("selected_conda_source", "")
        )
        selected_pip_source = str(previous_metadata.get("selected_pip_source", ""))
        source_attempts: list[dict[str, object]] = []
        stdout = f"repairing existing Conda environment {environment_name}"
        stderr = ""
        requirements = request.requirements_file.read_bytes()
        if requirements.strip():
            install, selected_pip_source, pip_attempts = (
                self._install_requirements_with_failover(
                    prefix=prefix,
                    request=request,
                    policy=source_policy,
                    deadline=deadline,
                )
            )
            source_attempts.extend(pip_attempts)
            stdout = (stdout + "\n" + install.stdout)[-request.max_log_bytes :]
            stderr = install.stderr[-request.max_log_bytes :]
            if install.exit_code != 0:
                return CondaEnvironmentBuildResult(
                    environment_ref=request.base_environment_ref,
                    environment_digest="",
                    exit_code=install.exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    termination_reason=install.termination_reason,
                    environment_fingerprint=base_fingerprint,
                    cache_ref=request.base_environment_ref,
                    environment_name=environment_name,
                    selected_conda_source=selected_conda_source,
                    selected_pip_source=selected_pip_source,
                    source_attempts=source_attempts,
                )

        provision_stdout, provision_stderr, provision_attempts, uv_source = self._provision_prefix(
            prefix,
            request,
            source_policy=source_policy,
            deadline=deadline,
            preferred_source_id=selected_pip_source,
        )
        source_attempts.extend(provision_attempts)
        selected_pip_source = selected_pip_source or uv_source
        stdout = (stdout + "\n" + provision_stdout)[-request.max_log_bytes :]
        stderr = (stderr + "\n" + provision_stderr)[-request.max_log_bytes :]
        try:
            manifest_timeout = int(deadline - time.monotonic())
            if manifest_timeout <= 0:
                raise subprocess.TimeoutExpired("package manifest", 0)
            manifest = self._package_manifest(prefix, manifest_timeout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            return CondaEnvironmentBuildResult(
                environment_ref=request.base_environment_ref,
                environment_digest="",
                exit_code=125,
                stdout=stdout,
                stderr=(stderr + f"\nfailed to capture package manifest: {exc}")[
                    -request.max_log_bytes :
                ],
                termination_reason="manifest_capture_failed",
                environment_fingerprint=base_fingerprint,
                cache_ref=request.base_environment_ref,
                environment_name=environment_name,
                selected_conda_source=selected_conda_source,
                selected_pip_source=selected_pip_source,
                source_attempts=source_attempts,
            )
        manifest_digest = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
        environment_digest = hashlib.sha256(
            f"{fingerprint}:{manifest_digest}".encode("utf-8")
        ).hexdigest()
        marker.write_text(
            json.dumps(
                {
                    "environment_fingerprint": fingerprint,
                    "environment_digest": environment_digest,
                    "package_manifest_digest": manifest_digest,
                    "environment_name": environment_name,
                    "selected_conda_source": selected_conda_source,
                    "selected_pip_source": selected_pip_source,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return CondaEnvironmentBuildResult(
            environment_ref=environment_ref,
            environment_digest=environment_digest,
            exit_code=0,
            stdout=stdout,
            stderr=stderr,
            environment_fingerprint=fingerprint,
            cache_ref=environment_ref,
            package_manifest_digest=manifest_digest,
            environment_name=environment_name,
            selected_conda_source=selected_conda_source,
            selected_pip_source=selected_pip_source,
            source_attempts=source_attempts,
        )

    def _install_uv_with_failover(
        self,
        *,
        prefix: Path,
        request: CondaEnvironmentBuildRequest,
        policy: PackageSourcePolicy,
        deadline: float,
        preferred_source_id: str,
    ) -> tuple[_CommandResult, str, list[dict[str, object]]]:
        if not request.network_enabled:
            command = [
                str(self._python_executable(prefix)),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-cache-dir",
                "--no-index",
            ]
            for wheel_dir in request.wheel_dirs:
                command.extend(["--find-links", str(wheel_dir)])
            command.append("uv")
            return self._run_with_deadline(command, request, deadline), "offline", []

        attempts: list[dict[str, object]] = []
        logs: list[str] = []
        last_result = self._CommandResult(
            125, "", "no approved pip source for uv", "source_exhausted"
        )
        selected_source = ""
        sources = self._prefer_source(
            self._ordered_sources(policy.pip_sources), preferred_source_id
        )
        for source in sources:
            with tempfile.TemporaryDirectory(
                prefix=".repro-uv-download-", dir=self.environment_root
            ) as download_root:
                download_dir = Path(download_root)
                download_command = [
                    str(self._python_executable(prefix)),
                    "-m",
                    "pip",
                    "--isolated",
                    "download",
                    "--dest",
                    str(download_dir),
                    "--retries",
                    "2",
                    "--timeout",
                    "30",
                    "--index-url",
                    source.location,
                    "uv",
                ]
                downloaded = self._run_with_deadline(
                    download_command, request, deadline, max_seconds=180
                )
                failure_kind = (
                    None
                    if downloaded.exit_code == 0
                    else self._classify_attempt_failure(downloaded, deadline)
                )
                attempts.append(
                    source_attempt(
                        source,
                        phase="uv_download",
                        exit_code=downloaded.exit_code,
                        failure_kind=failure_kind,
                    )
                )
                logs.append(self._source_log(source, downloaded, failure_kind))
                last_result = downloaded
                selected_source = source.source_id
                if downloaded.exit_code == 0:
                    install_command = [
                        str(self._python_executable(prefix)),
                        "-m",
                        "pip",
                        "--isolated",
                        "install",
                        "--no-cache-dir",
                        "--no-index",
                        "--find-links",
                        str(download_dir),
                        "uv",
                    ]
                    installed = self._run_with_deadline(
                        install_command, request, deadline
                    )
                    install_failure = (
                        None
                        if installed.exit_code == 0
                        else classify_source_failure(
                            installed.stderr, installed.stdout
                        )
                    )
                    attempts.append(
                        source_attempt(
                            source,
                            phase="uv_install_offline",
                            exit_code=installed.exit_code,
                            failure_kind=install_failure,
                        )
                    )
                    logs.append(
                        self._source_log(source, installed, install_failure)
                    )
                    return (
                        self._with_combined_logs(
                            installed, logs, request.max_log_bytes
                        ),
                        selected_source,
                        attempts,
                    )
                may_failover = policy.may_failover(
                    source=source,
                    failure_kind=failure_kind or SourceFailureKind.UNKNOWN,
                )
                if may_failover:
                    self._source_unhealthy_until[source.source_id] = (
                        time.monotonic() + 600
                    )
                if not may_failover:
                    break
        return (
            self._with_combined_logs(last_result, logs, request.max_log_bytes),
            selected_source,
            attempts,
        )

    @staticmethod
    def _prefer_source(
        sources: tuple[PackageSource, ...], preferred_source_id: str
    ) -> tuple[PackageSource, ...]:
        if not preferred_source_id:
            return sources
        preferred = tuple(
            source for source in sources if source.source_id == preferred_source_id
        )
        others = tuple(
            source for source in sources if source.source_id != preferred_source_id
        )
        return preferred + others

    def _provision_prefix(
        self,
        prefix: Path,
        request: CondaEnvironmentBuildRequest,
        *,
        source_policy: PackageSourcePolicy,
        deadline: float,
        preferred_source_id: str = "",
    ) -> tuple[str, str, list[dict[str, object]], str]:
        """Best-effort post-install provisioning shared by every environment.

        评测脚本普遍依赖 ``uv run`` 启动，而 nltk 语料不在任何 pip 包里；
        缺这两样时实验智能体只能反复试错。此处统一补齐：

        1. 安装 ``uv``（幂等，已安装时秒回）；
        2. 若宿主机存在 ``~/nltk_data``，镜像到环境 prefix 下。

        所有失败均不致命，仅记录到构建日志。
        """
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        source_attempts: list[dict[str, object]] = []
        selected_source = ""
        try:
            result, selected_source, source_attempts = (
                self._install_uv_with_failover(
                    prefix=prefix,
                    request=request,
                    policy=source_policy,
                    deadline=deadline,
                    preferred_source_id=preferred_source_id,
                )
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stderr_parts.append(f"warning: uv provisioning failed: {exc}")
        else:
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            if result.exit_code != 0:
                stderr_parts.append("warning: uv provisioning failed (non-fatal)")
                selected_source = ""
        host_nltk = Path.home() / "nltk_data"
        if host_nltk.is_dir():
            try:
                shutil.copytree(host_nltk, prefix / "nltk_data", dirs_exist_ok=True)
                stdout_parts.append(f"provisioned nltk_data into {prefix / 'nltk_data'}")
            except OSError as exc:
                stderr_parts.append(f"warning: nltk_data copy failed: {exc}")
        return (
            "\n".join(stdout_parts),
            "\n".join(stderr_parts),
            source_attempts,
            selected_source,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.require_available(purpose="experiment execution")
        prefix = self._prefix_from_ref(request.image)
        if not self._python_executable(prefix).is_file():
            raise RuntimeError(f"Conda environment is not ready: {request.image}")
        working_dir = (request.workspace_dir / request.working_dir).resolve()
        workspace = request.workspace_dir.resolve()
        if working_dir != workspace and workspace not in working_dir.parents:
            raise ValueError("working_dir must stay inside the task workspace")

        identifier = f"conda-{request.attempt_id}"
        command = [
            self.conda_binary,
            "run",
            "--no-capture-output",
            "--prefix",
            str(prefix),
            *request.command,
        ]
        runtime_environment = self._runtime_environment(request, prefix=prefix)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        state_path = request.state_path
        self._write_execution_state(
            state_path,
            {
                "status": "PREPARING",
                "runtime": "conda",
                "environment_ref": request.image,
                "container_name": identifier,
                "started_at": started_at,
            },
        )
        stdout_path = request.workspace_dir / f".{identifier}.stdout"
        stderr_path = request.workspace_dir / f".{identifier}.stderr"
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        try:
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                env=runtime_environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            stdout_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)
            self._write_execution_state(
                state_path,
                {
                    "status": "TERMINATION_FAILED",
                    "runtime": "conda",
                    "environment_ref": request.image,
                    "container_name": identifier,
                    "started_at": started_at,
                },
            )
            raise
        with self._lock:
            self._processes[identifier] = process
        self._write_execution_state(
            state_path,
            {
                "status": "RUNNING",
                "runtime": "conda",
                "environment_ref": request.image,
                "environment_prefix": str(prefix),
                "container_name": identifier,
                "pid": process.pid,
                "started_at": started_at,
            },
        )

        termination_reason = "completed"
        exit_code = -1
        try:
            while True:
                return_code = process.poll()
                elapsed = time.monotonic() - started
                cancelled = bool(
                    request.cancellation_event is not None
                    and request.cancellation_event.is_set()
                )
                timed_out = elapsed >= request.timeout_seconds
                log_limited = (
                    self._file_size(stdout_path) + self._file_size(stderr_path)
                    > request.resources.max_log_bytes
                )
                if return_code is not None and not log_limited:
                    exit_code = return_code
                    break
                if not (cancelled or timed_out or log_limited):
                    time.sleep(min(0.25, request.timeout_seconds))
                    continue
                termination_reason = (
                    "cancelled_by_controller"
                    if cancelled
                    else "timeout_killed"
                    if timed_out
                    else "log_limit_exceeded"
                )
                exit_code = 130 if cancelled else 124 if timed_out else 137
                self.cancel(identifier)
                break
        finally:
            stdout_handle.close()
            stderr_handle.close()
            with self._lock:
                self._processes.pop(identifier, None)

        stdout = self._read_bounded(stdout_path, request.resources.max_log_bytes)
        stderr = self._read_bounded(stderr_path, request.resources.max_log_bytes)
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        completed_at = datetime.now(timezone.utc).isoformat()
        self._write_execution_state(
            state_path,
            {
                "status": (
                    "COMPLETED" if termination_reason == "completed" else "TERMINATED"
                ),
                "runtime": "conda",
                "environment_ref": request.image,
                "environment_prefix": str(prefix),
                "container_name": identifier,
                "pid": process.pid,
                "exit_code": exit_code,
                "termination_reason": termination_reason,
                "started_at": started_at,
                "completed_at": completed_at,
            },
        )
        return ExecutionResult(
            command=request.command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            container_name=identifier,
            image_digest=self._fingerprint_from_ref(request.image),
            termination_reason=termination_reason,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=time.monotonic() - started,
            execution_state_path=str(state_path or ""),
        )

    def cancel(self, identifier: str) -> bool:
        with self._lock:
            process = self._processes.get(identifier)
        if process is None:
            # 本控制器进程没有注册过该标识符：没有可信号的子进程。
            # 这与 Docker 后端“容器不存在即视为已对账”的语义一致；
            # 跨进程孤儿进程由 reconcile_execution 从持久化 PID 校验终止。
            return True
        if process.poll() is not None:
            return True
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except (OSError, ProcessLookupError):
            return process.poll() is not None
        return process.poll() is not None

    def reconcile_execution(self, state: dict[str, object]) -> bool:
        """Terminate a persisted Conda process group after controller restart.

        The PID is used only after its current command line is verified to be
        this backend's ``conda run --prefix <managed-prefix>`` invocation.  A
        disappeared PID is already reconciled; a reused/unrelated PID is never
        signalled.
        """

        try:
            pid = int(state.get("pid", 0))
        except (TypeError, ValueError):
            return False
        if pid <= 1:
            return False
        try:
            command_line = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return False
        if not command_line:
            return True
        prefix = str(state.get("environment_prefix", ""))
        try:
            expected_prefix = self._prefix_from_ref(
                str(state.get("environment_ref", ""))
            )
        except (RuntimeError, ValueError):
            return False
        if prefix != str(expected_prefix):
            return False
        if (
            str(expected_prefix) not in command_line
            or Path(self.conda_binary).name not in command_line
        ):
            return False
        try:
            os.killpg(pid, signal.SIGTERM)
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
                time.sleep(0.25)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return True

    def _prefix_from_ref(self, environment_ref: str) -> Path:
        match = _CONDA_REF_RE.fullmatch(str(environment_ref))
        if match is None:
            raise ValueError(
                "Conda execution requires a valid conda:// environment_ref"
            )
        fingerprint, environment_name = match.groups()
        if environment_name:
            named_prefix = self._prefix_for_environment_name(environment_name)
            if self._valid_cache_marker(
                named_prefix / ".repro_agent_environment.json", fingerprint
            ):
                return named_prefix
            return self._prefix_for_fingerprint(fingerprint)
        legacy_prefix = self._prefix_for_fingerprint(fingerprint)
        if self._valid_cache_marker(
            legacy_prefix / ".repro_agent_environment.json", fingerprint
        ):
            return legacy_prefix

        # Named prefixes keep the opaque fingerprint reference stable.  Resolve
        # it by validated marker so a renamed/rebuilt environment can never be
        # mistaken for the older dependency set.
        matches: list[Path] = []
        for candidate in sorted(
            self.environment_root.iterdir(), key=lambda item: item.name
        ):
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            prefix = candidate.resolve()
            if prefix.parent != self.environment_root:
                continue
            if self._valid_cache_marker(
                prefix / ".repro_agent_environment.json", fingerprint
            ):
                matches.append(prefix)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Compatibility for jobs created before references carried the
            # stable environment name.  Old repair agents derived names from
            # staged inputs such as ``<16-hex>-0-repository``.  If exactly one
            # matching prefix is not attempt-derived, it is the original
            # project environment and can be upgraded safely on the next build.
            stable_matches = [
                prefix
                for prefix in matches
                if re.fullmatch(
                    r"[a-f0-9]{16}-\d+-repository", prefix.name
                )
                is None
            ]
            if len(stable_matches) == 1:
                return stable_matches[0]
            raise RuntimeError(
                "legacy Conda environment reference is ambiguous; rebuild or "
                "repair it once to obtain a name-bound reference"
            )
        # Preserve the historical failure path: execute() will report that the
        # requested immutable environment is no longer available.
        return legacy_prefix

    @staticmethod
    def _environment_ref(fingerprint: str, environment_name: str) -> str:
        return f"conda://{fingerprint}/{environment_name}"

    @staticmethod
    def _fingerprint_from_ref(environment_ref: str) -> str:
        match = _CONDA_REF_RE.fullmatch(str(environment_ref))
        if match is None:
            raise ValueError("invalid Conda environment reference")
        return match.group(1)

    def _prefix_for_fingerprint(self, fingerprint: str) -> Path:
        prefix = (self.environment_root / fingerprint).resolve()
        if prefix.parent != self.environment_root:
            raise ValueError("invalid managed Conda environment fingerprint")
        return prefix

    def _prefix_for_environment_name(self, environment_name: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,62}", environment_name):
            raise ValueError("invalid managed Conda environment name")
        prefix = (self.environment_root / environment_name).resolve()
        if prefix.parent != self.environment_root:
            raise ValueError("invalid managed Conda environment name")
        return prefix

    def _remove_managed_prefix(
        self, prefix: Path, *, allow_unmarked: bool = False
    ) -> None:
        if prefix.parent != self.environment_root:
            raise ValueError("refusing to remove unmanaged Conda prefix")
        if prefix.exists():
            marker = prefix / ".repro_agent_environment.json"
            if not allow_unmarked and not self._is_managed_marker(marker):
                raise RuntimeError(
                    f"refusing to replace an unmanaged Conda environment: {prefix.name}"
                )
            shutil.rmtree(prefix)

    def _backup_managed_prefix(
        self, prefix: Path, *, attempt_id: str
    ) -> Path | None:
        if not prefix.exists():
            return None
        marker = prefix / ".repro_agent_environment.json"
        if not self._is_managed_marker(marker, expected_name=prefix.name):
            raise RuntimeError(
                f"refusing to replace an unmanaged Conda environment: {prefix.name}"
            )
        safe_attempt = re.sub(r"[^A-Za-z0-9_.-]", "-", attempt_id)[-32:]
        backup = self.environment_root / f".repro-backup-{prefix.name}-{safe_attempt}"
        if backup.exists():
            raise RuntimeError(f"stale Conda environment backup exists: {backup.name}")
        prefix.rename(backup)
        return backup

    def _restore_managed_backup(
        self, prefix: Path, backup: Path | None
    ) -> None:
        if backup is None:
            return
        if prefix.exists():
            self._remove_managed_prefix(prefix, allow_unmarked=True)
        if not self._is_managed_marker(
            backup / ".repro_agent_environment.json", expected_name=prefix.name
        ):
            raise RuntimeError("refusing to restore an invalid Conda environment backup")
        backup.rename(prefix)

    def _discard_managed_backup(
        self, prefix: Path, backup: Path | None
    ) -> None:
        if backup is None:
            return
        if not self._is_managed_marker(
            backup / ".repro_agent_environment.json", expected_name=prefix.name
        ):
            raise RuntimeError("refusing to discard an invalid Conda environment backup")
        shutil.rmtree(backup)

    @staticmethod
    def _is_managed_marker(
        marker: Path, *, expected_name: str | None = None
    ) -> bool:
        if not marker.is_file():
            return False
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return all(
            re.fullmatch(r"[a-f0-9]{64}", str(value.get(field, ""))) is not None
            for field in (
                "environment_fingerprint",
                "environment_digest",
                "package_manifest_digest",
            )
        ) and str(value.get("environment_name", "")) == (
            expected_name or marker.parent.name
        )

    def _valid_cache_marker(self, marker: Path, fingerprint: str) -> bool:
        if not marker.is_file() or not self._python_executable(marker.parent).is_file():
            return False
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        digest = str(value.get("environment_digest", ""))
        manifest_digest = str(value.get("package_manifest_digest", ""))
        return (
            value.get("environment_fingerprint") == fingerprint
            and re.fullmatch(r"[a-f0-9]{64}", digest) is not None
            and re.fullmatch(r"[a-f0-9]{64}", manifest_digest) is not None
        )

    @staticmethod
    def _wheel_fingerprints(wheel_dirs: tuple[Path, ...]) -> list[dict[str, str]]:
        """Hash vendored wheels so cache reuse follows dependency content."""

        wheels = sorted(
            {
                path.resolve()
                for directory in wheel_dirs
                for path in directory.rglob("*.whl")
            },
            key=lambda path: str(path),
        )
        if len(wheels) > 512:
            raise ValueError("refusing to fingerprint more than 512 vendored wheels")
        result: list[dict[str, str]] = []
        for wheel in wheels:
            digest = hashlib.sha256()
            with wheel.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result.append({"name": wheel.name, "sha256": digest.hexdigest()})
        return result

    @staticmethod
    def _python_executable(prefix: Path) -> Path:
        return prefix / ("python.exe" if os.name == "nt" else "bin/python")

    def _package_manifest(self, prefix: Path, timeout_seconds: int) -> str:
        conda_list = subprocess.run(
            [self.conda_binary, "list", "--prefix", str(prefix), "--json"],
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout_seconds, 300)),
            check=True,
        ).stdout
        pip_freeze = subprocess.run(
            [str(self._python_executable(prefix)), "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout_seconds, 300)),
            check=True,
        ).stdout
        return json.dumps(
            {
                "conda": json.loads(conda_list),
                "pip": sorted(line for line in pip_freeze.splitlines() if line),
            },
            sort_keys=True,
        )

    def _runtime_environment(
        self, request: ExecutionRequest, *, prefix: Path | None = None
    ) -> dict[str, str]:
        allowed_host = {
            name: os.environ[name]
            for name in (
                "PATH",
                "LANG",
                "LC_ALL",
                "TMPDIR",
                "SYSTEMROOT",
                # HuggingFace 镜像/缓存位置：国内访问 huggingface.co 受阻，
                # 允许宿主机预配置 HF_ENDPOINT（镜像）与共享 HF_HOME（离线
                # 缓存）透传进任务进程，避免运行时 tokenizer 下载卡死。
                "HF_ENDPOINT",
                "HF_HOME",
            )
            if os.environ.get(name)
        }
        task_home = request.workspace_dir / ".repro_home"
        task_home.mkdir(parents=True, exist_ok=True)
        allowed_host["HOME"] = str(task_home)
        # 安全敏感的宿主机路径变量：任务无权改写，必须显式拒绝。
        reserved = {
            "HOME",
            "PATH",
            "TMP",
            "TEMP",
            "TMPDIR",
        }
        # 后端托管的路径变量：不同后端的挂载点不同（conda 用宿主路径，
        # docker 用容器路径）。旧任务里可能残留 agent 提供的同名值，
        # 静默丢弃而非拒绝，保证后端无关的调用方不会被卡死。
        backend_managed = {
            "REPRO_AGENT_INPUT_DIR",
            "REPRO_AGENT_OUTPUT_DIR",
            "REPRO_AGENT_METRICS_PATH",
        }
        for key, value in request.environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", key) or any(
                marker in key.lower()
                for marker in ("secret", "token", "password", "key")
            ):
                raise ValueError(f"environment variable is not allowlisted: {key}")
            if key in backend_managed:
                # Backend-managed locations: input/output/metrics paths differ
                # per backend (host paths here, container mounts in Docker).
                # Request-supplied duplicates are dropped, not rejected, so a
                # backend-agnostic caller cannot wedge every execution.
                continue
            if key in reserved:
                raise ValueError(f"environment variable is not allowlisted: {key}")
            allowed_host[key] = str(value)
        for name in request.passthrough_environment:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name):
                raise ValueError(f"invalid passthrough environment variable: {name}")
            if not os.environ.get(name, ""):
                raise ValueError(f"passthrough environment variable is not set: {name}")
            allowed_host[name] = os.environ[name]
        allowed_host["REPRO_AGENT_INPUT_DIR"] = str(request.input_dir)
        allowed_host["REPRO_AGENT_OUTPUT_DIR"] = str(request.output_dir)
        allowed_host["REPRO_AGENT_METRICS_PATH"] = str(request.output_dir / "metrics.json")
        # 项目自带的包装脚本常用 ``uv run`` 自建 venv：默认走 pypi.org 在国内
        # 直连极慢（torch 级别依赖十分钟超时不够用），且每个任务的隔离 HOME
        # 会让 uv 缓存无法跨 attempt 复用。这里统一注入镜像与共享缓存；
        # 同时把预置好的 nltk_data 挂进 NLTK_DATA，避免评测脚本在隔离
        # HOME 里重新联网下载语料。
        pip_index_url = os.environ.get("REPRO_AGENT_PIP_INDEX_URL", "")
        if pip_index_url:
            allowed_host["UV_DEFAULT_INDEX"] = pip_index_url
            allowed_host["UV_INDEX_URL"] = pip_index_url
        shared_uv_cache = self.environment_root / ".uv_cache"
        try:
            shared_uv_cache.mkdir(parents=True, exist_ok=True)
            allowed_host["UV_CACHE_DIR"] = str(shared_uv_cache)
        except OSError:
            pass
        if prefix is not None and (prefix / "nltk_data").is_dir():
            allowed_host["NLTK_DATA"] = str(prefix / "nltk_data")
        return allowed_host

    @dataclass
    class _CommandResult:
        exit_code: int
        stdout: str
        stderr: str
        termination_reason: str

    def _run_build_command(
        self,
        command: list[str],
        *,
        timeout_seconds: int,
        cancellation_event: threading.Event | None,
        max_log_bytes: int,
    ) -> _CommandResult:
        started = time.monotonic()
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            termination_reason = "completed"
            requested_exit: int | None = None
            while process.poll() is None:
                cancelled = cancellation_event is not None and cancellation_event.is_set()
                timed_out = time.monotonic() - started >= timeout_seconds
                log_limited = (
                    os.fstat(stdout_file.fileno()).st_size
                    + os.fstat(stderr_file.fileno()).st_size
                    > max_log_bytes
                )
                if cancelled or timed_out or log_limited:
                    requested_exit = 130 if cancelled else 124 if timed_out else 137
                    termination_reason = (
                        "cancelled_by_controller"
                        if cancelled
                        else "timeout_killed"
                        if timed_out
                        else "log_limit_exceeded"
                    )
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    break
                time.sleep(0.25)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read().decode("utf-8", errors="replace")[-max_log_bytes:]
            stderr = stderr_file.read().decode("utf-8", errors="replace")[-max_log_bytes:]
            return self._CommandResult(
                requested_exit if requested_exit is not None else int(process.returncode or 0),
                stdout,
                stderr,
                termination_reason,
            )

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _read_bounded(path: Path, limit: int) -> str:
        try:
            with path.open("rb") as handle:
                if path.stat().st_size > limit:
                    handle.seek(-limit, os.SEEK_END)
                return handle.read(limit).decode("utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _write_execution_state(
        state_path: Path | None, payload: dict[str, object]
    ) -> None:
        if state_path is None:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
