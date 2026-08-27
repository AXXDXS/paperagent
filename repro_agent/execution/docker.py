"""Fail-closed Docker execution for untrusted experiment commands."""

from __future__ import annotations

import shutil
import subprocess
import time
import json
import os
import re
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath

from repro_agent.execution.backend import (
    ExecutionRequest,
    ExecutionResult,
    ImageBuildRequest,
    ImageBuildResult,
)


class ExecutionUnavailable(RuntimeError):
    pass


class DockerExecutionBackend:
    def __init__(self, *, docker_binary: str = "docker"):
        self.docker_binary = docker_binary

    def is_available(self) -> bool:
        return shutil.which(self.docker_binary) is not None

    def unavailable_reason(self, *, purpose: str = "container execution") -> str:
        """Return an actionable diagnostic for a missing Docker control plane."""

        if shutil.which(self.docker_binary) is None:
            return (
                f"Docker CLI '{self.docker_binary}' is required for {purpose} but "
                "was not found on PATH"
            )
        return f"Docker is unavailable for {purpose}"

    def require_available(self, *, purpose: str = "container execution") -> None:
        """Fail closed instead of falling back to an unisolated host process."""

        if not self.is_available():
            raise ExecutionUnavailable(self.unavailable_reason(purpose=purpose))

    @staticmethod
    def _container_name(request: ExecutionRequest) -> str:
        return DockerExecutionBackend.container_name_for(
            request.task_id, request.attempt_id
        )

    @staticmethod
    def container_name_for(task_id: str, attempt_id: str) -> str:
        """Return the stable name used both by launch and crash recovery."""

        safe_task = "".join(c if c.isalnum() or c in "_.-" else "-" for c in task_id)
        safe_attempt = "".join(c if c.isalnum() or c in "_.-" else "-" for c in attempt_id)
        return f"repro-{safe_task}-{safe_attempt}"[:120]

    def build_run_argv(self, request: ExecutionRequest) -> list[str]:
        if not request.command:
            raise ValueError("execution command must not be empty")
        for directory in (request.input_dir, request.workspace_dir, request.output_dir):
            if not directory.is_dir():
                raise ValueError(f"execution mount does not exist: {directory}")
        limits = request.resources
        if limits.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        for field_name in (
            "memory_mb",
            "disk_mb",
            "max_processes",
            "max_open_files",
            "max_log_bytes",
            "tmpfs_mb",
        ):
            if int(getattr(limits, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if request.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        name = self._container_name(request)
        working_dir = PurePosixPath(request.working_dir)
        if working_dir.is_absolute() or ".." in working_dir.parts:
            raise ValueError("working_dir must stay within /workspace")
        container_workdir = str(PurePosixPath("/workspace") / working_dir)
        workspace_mount = (
            f"type=bind,src={request.workspace_dir.resolve()},dst=/workspace"
        )
        if request.workspace_read_only:
            workspace_mount += ",readonly"
        argv = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "bridge" if request.network_enabled else "none",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--memory",
            f"{limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            "--cpus",
            str(limits.cpu_cores),
            "--pids-limit",
            str(limits.max_processes),
            "--ulimit",
            f"nofile={limits.max_open_files}:{limits.max_open_files}",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={limits.tmpfs_mb}m",
            "--mount",
            f"type=bind,src={request.input_dir.resolve()},dst=/input,readonly",
            "--mount",
            workspace_mount,
            "--mount",
            f"type=bind,src={request.output_dir.resolve()},dst=/output",
            "--workdir",
            container_workdir,
        ]
        if request.gpu_count:
            if request.gpu_count < 0:
                raise ValueError("gpu_count must be non-negative")
            argv.extend(["--gpus", str(request.gpu_count)])
        managed_environment = {
            # Backend-managed locations mirror what the Conda backend exports;
            # request-supplied duplicates are dropped, not rejected.
            "REPRO_AGENT_INPUT_DIR": "/input",
            "REPRO_AGENT_OUTPUT_DIR": "/output",
            "REPRO_AGENT_METRICS_PATH": "/output/metrics.json",
        }
        for key, value in sorted(request.environment.items()):
            if not key or any(marker in key.lower() for marker in ("secret", "token", "password", "key")):
                raise ValueError(f"environment variable is not allowlisted: {key}")
            if key in managed_environment:
                continue
            argv.extend(["--env", f"{key}={value}"])
        for key, value in sorted(managed_environment.items()):
            argv.extend(["--env", f"{key}={value}"])
        for key in sorted(set(request.passthrough_environment)):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", key):
                raise ValueError(f"invalid passthrough environment variable: {key}")
            if not os.environ.get(key, ""):
                raise ValueError(f"passthrough environment variable is not set: {key}")
            # Docker's name-only form copies the value from its own process
            # environment without placing it in argv or persisted state.
            argv.extend(["--env", key])
        return [*argv, request.image, *request.command]

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.require_available(purpose="real experiment execution")
        argv = self.build_run_argv(request)
        name = self._container_name(request)
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        log_root = request.state_path.parent if request.state_path else request.output_dir
        log_root.mkdir(parents=True, exist_ok=True)
        stdout_capture = log_root / f".{name}.controller.stdout.tmp"
        stderr_capture = log_root / f".{name}.controller.stderr.tmp"
        stdout_handle = stdout_capture.open("wb")
        stderr_handle = stderr_capture.open("wb")
        # Persist the deterministic container identity before spawning Docker.
        # If the controller dies immediately after Popen, recovery can still
        # locate and terminate the named container instead of requeueing a
        # second attempt alongside an orphan.
        self._write_execution_state(
            request,
            {
                "status": "PREPARING",
                "container_name": name,
                "controller_pid": os.getpid(),
                "started_at": started_at,
                "command": request.command,
                "image": request.image,
                "network_enabled": request.network_enabled,
                "resource_policy": {
                    "cpu_cores": request.resources.cpu_cores,
                    "memory_mb": request.resources.memory_mb,
                    "disk_mb": request.resources.disk_mb,
                    "gpu_count": request.gpu_count,
                    "gpu_memory_mb": request.resources.gpu_memory_mb,
                },
            },
        )
        try:
            process = subprocess.Popen(
                argv,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        except Exception as exc:
            stdout_handle.close()
            stderr_handle.close()
            self._remove_capture_files(stdout_capture, stderr_capture)
            self._write_execution_state(
                request,
                {
                    "status": "LAUNCH_FAILED",
                    "container_name": name,
                    "started_at": started_at,
                    "error": str(exc)[:2000],
                },
            )
            raise
        self._write_execution_state(
            request,
            {
                "status": "RUNNING",
                "container_name": name,
                "controller_pid": process.pid,
                "started_at": started_at,
                "command": request.command,
                "image": request.image,
            },
        )
        termination_reason = "completed"
        exit_code = -1
        state_status = "COMPLETED"
        next_disk_check = 0.0
        try:
            while True:
                return_code = process.poll()
                elapsed = time.monotonic() - started_monotonic
                cancelled = bool(
                    request.cancellation_event is not None
                    and request.cancellation_event.is_set()
                )
                timed_out = elapsed >= request.timeout_seconds
                log_bytes = self._file_size(stdout_capture) + self._file_size(stderr_capture)
                log_limited = log_bytes > request.resources.max_log_bytes
                disk_limited = False
                if not (cancelled or timed_out) and (
                    return_code is not None or elapsed >= next_disk_check
                ):
                    disk_limited = self._directory_size_exceeds(
                        (request.workspace_dir, request.output_dir),
                        request.resources.disk_mb * 1024 * 1024,
                    )
                    next_disk_check = elapsed + 1.0
                if return_code is not None and not (log_limited or disk_limited):
                    exit_code = return_code
                    break
                if not (cancelled or timed_out or log_limited or disk_limited):
                    time.sleep(min(0.25, request.timeout_seconds))
                    continue

                if cancelled:
                    requested_reason, requested_exit = "cancelled_by_controller", 130
                elif timed_out:
                    requested_reason, requested_exit = "timeout_killed", 124
                elif log_limited:
                    requested_reason, requested_exit = "log_limit_exceeded", 137
                else:
                    requested_reason, requested_exit = "disk_limit_exceeded", 137

                # First stop/kill request handles an already-running container.
                # Then quiesce the local Docker CLI and inspect once more; this
                # closes the race where cancellation arrives before ``docker
                # run`` has finished creating the named container.
                self.cancel(name)
                controller_quiesced = True
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # This only terminates the local Docker CLI.  Container
                    # termination is represented solely by ``cancel`` above.
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        controller_quiesced = False
                termination_confirmed = controller_quiesced and self.cancel(name)
                if termination_confirmed:
                    termination_reason = requested_reason
                    exit_code = requested_exit
                    state_status = "TERMINATED"
                else:
                    termination_reason = "termination_unconfirmed"
                    exit_code = 125
                    state_status = "TERMINATION_FAILED"
                break
        finally:
            stdout_handle.close()
            stderr_handle.close()

        stdout, stderr = self._read_bounded_captures(
            stdout_capture,
            stderr_capture,
            request.resources.max_log_bytes,
        )
        self._remove_capture_files(stdout_capture, stderr_capture)

        completed_at = datetime.now(timezone.utc).isoformat()
        state = {
            "status": state_status,
            "container_name": name,
            "controller_pid": process.pid,
            "started_at": started_at,
            "completed_at": completed_at,
            "exit_code": exit_code,
            "termination_reason": termination_reason,
        }
        self._write_execution_state(request, state)
        return ExecutionResult(
            command=request.command,
            exit_code=exit_code,
            stdout=stdout or "",
            stderr=stderr or "",
            container_name=name,
            image_digest=self._resolve_image_digest(request.image),
            termination_reason=termination_reason,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=time.monotonic() - started_monotonic,
            execution_state_path=str(request.state_path or ""),
        )

    def cancel(self, container_name: str) -> bool:
        if not container_name or not self.is_available():
            return False
        if not self._container_exists(container_name):
            return True
        commands = (
            [self.docker_binary, "stop", "--time", "5", container_name],
            [self.docker_binary, "kill", container_name],
            [self.docker_binary, "rm", "--force", container_name],
        )
        for command in commands:
            try:
                subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                pass
            if not self._container_exists(container_name):
                return True
        return False

    def _container_exists(self, container_name: str) -> bool:
        try:
            inspected = subprocess.run(
                [self.docker_binary, "container", "inspect", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            # An unavailable control plane is not proof that a container died.
            return True
        return inspected.returncode == 0

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @classmethod
    def _directory_size_exceeds(cls, roots: tuple[Path, ...], limit_bytes: int) -> bool:
        total = 0
        for root in roots:
            try:
                entries = root.rglob("*")
                for entry in entries:
                    try:
                        if entry.is_file() and not entry.is_symlink():
                            total += entry.stat().st_size
                            if total > limit_bytes:
                                return True
                    except OSError:
                        continue
            except OSError:
                continue
        return False

    @classmethod
    def _read_bounded_captures(
        cls, stdout_path: Path, stderr_path: Path, limit_bytes: int
    ) -> tuple[str, str]:
        stdout_size = cls._file_size(stdout_path)
        stderr_size = cls._file_size(stderr_path)
        stdout_budget = min(stdout_size, limit_bytes // 2)
        stderr_budget = min(stderr_size, limit_bytes - stdout_budget)
        remaining = limit_bytes - stdout_budget - stderr_budget
        extra_stdout = min(max(0, stdout_size - stdout_budget), remaining)
        stdout_budget += extra_stdout
        remaining -= extra_stdout
        stderr_budget += min(max(0, stderr_size - stderr_budget), remaining)
        return (
            cls._read_file_tail(stdout_path, stdout_budget),
            cls._read_file_tail(stderr_path, stderr_budget),
        )

    @staticmethod
    def _read_file_tail(path: Path, max_bytes: int) -> str:
        if max_bytes <= 0:
            return ""
        try:
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size - max_bytes))
                return handle.read(max_bytes).decode("utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _remove_capture_files(*paths: Path) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _write_execution_state(request: ExecutionRequest, payload: dict) -> None:
        if request.state_path is None:
            return
        request.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = request.state_path.with_suffix(request.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary.replace(request.state_path)

    def _resolve_image_digest(self, image: str) -> str:
        """Return the immutable image identifier actually used by Docker."""

        if "@" in image:
            return image.split("@", 1)[1]
        try:
            completed = subprocess.run(
                [self.docker_binary, "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip() if completed.returncode == 0 else ""

    @staticmethod
    def _environment_fingerprint(
        context: Path,
        base_image_digests: list[tuple[str, str]],
        network_enabled: bool = False,
    ) -> str:
        """Hash every input that can affect an environment build.

        Timestamps and ownership are intentionally excluded because Docker's
        resulting environment does not depend on them.  Paths, file types,
        permission bits, symlink targets, file contents, host platform and the
        immutable IDs of all base images are included.  Hashing the complete
        context is conservative: even files not copied by the current
        Dockerfile invalidate the cache instead of risking a stale image.
        The build-time network mode is also hashed: an online build (pip
        resolving from PyPI) must never be served from an offline cache entry
        and vice versa.
        """

        digest = hashlib.sha256()

        def add(value: str | bytes) -> None:
            encoded = (
                value.encode("utf-8", errors="surrogateescape")
                if isinstance(value, str)
                else value
            )
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

        add("repro-agent-environment-cache-v1")
        add(platform.system())
        add(platform.machine())
        add("build-network=" + ("online" if network_enabled else "offline"))
        for image, image_digest in base_image_digests:
            add("base-image")
            add(image)
            add(image_digest)

        try:
            entries = sorted(
                context.rglob("*"),
                key=lambda path: path.relative_to(context).as_posix(),
            )
        except OSError as exc:
            raise ValueError(f"cannot enumerate image build context: {exc}") from exc
        for path in entries:
            relative = path.relative_to(context).as_posix()
            try:
                stat_result = path.lstat()
                mode = stat_result.st_mode & 0o7777
                if path.is_symlink():
                    kind = "symlink"
                    content = os.readlink(path)
                elif path.is_dir():
                    kind = "directory"
                    content = ""
                elif path.is_file():
                    kind = "file"
                    content = None
                else:
                    kind = "special"
                    content = ""
            except OSError as exc:
                raise ValueError(
                    f"cannot inspect image build context entry {relative}: {exc}"
                ) from exc
            add(kind)
            add(relative)
            add(str(mode))
            if content is not None:
                add(content)
                continue
            add(str(stat_result.st_size))
            try:
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
            except OSError as exc:
                raise ValueError(
                    f"cannot read image build context entry {relative}: {exc}"
                ) from exc
        return digest.hexdigest()

    def build_image(self, request: ImageBuildRequest) -> ImageBuildResult:
        """Build an auditable image without exposing Docker to agents.

        Builds default to ``--network none`` (fully offline).  When the request
        sets ``network_enabled`` ("Route A" build-time networking), the build
        runs on the default bridge network so pip can resolve dependencies
        online and missing base images may be pulled from a registry.
        """

        self.require_available(purpose="environment image builds")
        context = request.context_dir.resolve()
        dockerfile = request.dockerfile.resolve()
        if request.timeout_seconds <= 0:
            raise ValueError("image build timeout_seconds must be positive")
        if request.max_log_bytes <= 0:
            raise ValueError("image build max_log_bytes must be positive")
        if not context.is_dir() or context not in dockerfile.parents:
            raise ValueError("Dockerfile must be inside the declared build context")
        try:
            dockerfile_text = dockerfile.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read Dockerfile: {exc}") from exc
        base_images = []
        for line in dockerfile_text.splitlines():
            parts = line.strip().split()
            if not parts or parts[0].upper() != "FROM":
                continue
            image_index = 2 if len(parts) > 1 and parts[1].startswith("--platform=") else 1
            if len(parts) <= image_index:
                raise ValueError("invalid FROM instruction in Dockerfile")
            base_images.append(parts[image_index])
        base_image_digests: list[tuple[str, str]] = []
        for base_image in base_images:
            if base_image.lower() == "scratch":
                base_image_digests.append((base_image, "scratch"))
                continue
            if not self._image_exists(base_image):
                if not request.network_enabled:
                    return ImageBuildResult(
                        image_ref=request.image_tag,
                        image_digest="",
                        exit_code=125,
                        stderr=(
                            f"base image is not present locally: {base_image}; "
                            "offline builds never pull from a registry"
                        ),
                    )
                pull_failure = self._pull_base_image(base_image, request)
                if pull_failure is not None:
                    return pull_failure
            base_digest = self._resolve_image_digest(base_image)
            if not base_digest:
                return ImageBuildResult(
                    image_ref=request.image_tag,
                    image_digest="",
                    exit_code=125,
                    stderr=f"could not resolve immutable digest for base image: {base_image}",
                )
            base_image_digests.append((base_image, base_digest))
        fingerprint = self._environment_fingerprint(
            context, base_image_digests, network_enabled=request.network_enabled
        )
        cache_ref = f"repro-agent/env-cache:{fingerprint}"
        if not request.force_rebuild and self._image_exists(cache_ref):
            cached_digest = self._resolve_image_digest(cache_ref)
            if cached_digest:
                return ImageBuildResult(
                    image_ref=cached_digest,
                    image_digest=cached_digest,
                    exit_code=0,
                    stdout=f"reused cached environment image {cache_ref}",
                    cache_hit=True,
                    environment_fingerprint=fingerprint,
                    cache_ref=cache_ref,
                )
        argv = [
            self.docker_binary, "build",
            "--network", "bridge" if request.network_enabled else "none",
            "--pull=false",
        ]
        if request.force_rebuild:
            argv.append("--no-cache")
        argv.extend([
            "--file", str(dockerfile),
            "--tag", request.image_tag,
            "--tag", cache_ref,
            str(context),
        ])
        safe_attempt = "".join(
            char if char.isalnum() or char in "_.-" else "-"
            for char in request.attempt_id
        )[:80]
        log_root = (request.log_dir or context.parent).resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        stdout_capture = log_root / f".build-{safe_attempt}.controller.stdout.tmp"
        stderr_capture = log_root / f".build-{safe_attempt}.controller.stderr.tmp"
        stdout_handle = stdout_capture.open("wb")
        stderr_handle = stderr_capture.open("wb")
        try:
            process = subprocess.Popen(
                argv,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            self._remove_capture_files(stdout_capture, stderr_capture)
            raise

        started = time.monotonic()
        exit_code = -1
        termination_reason = "completed"
        try:
            while True:
                return_code = process.poll()
                log_limited = (
                    self._file_size(stdout_capture) + self._file_size(stderr_capture)
                    > request.max_log_bytes
                )
                cancelled = bool(
                    request.cancellation_event is not None
                    and request.cancellation_event.is_set()
                )
                timed_out = time.monotonic() - started >= request.timeout_seconds
                if return_code is not None and not log_limited:
                    exit_code = return_code
                    break
                if not (cancelled or timed_out or log_limited):
                    time.sleep(min(0.25, request.timeout_seconds))
                    continue
                if cancelled:
                    exit_code, termination_reason = 130, "cancelled_by_controller"
                elif timed_out:
                    exit_code, termination_reason = 124, "timeout_killed"
                else:
                    exit_code, termination_reason = 137, "log_limit_exceeded"
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        exit_code, termination_reason = 125, "termination_unconfirmed"
                break
        finally:
            stdout_handle.close()
            stderr_handle.close()

        stdout, stderr = self._read_bounded_captures(
            stdout_capture, stderr_capture, request.max_log_bytes
        )
        self._remove_capture_files(stdout_capture, stderr_capture)
        digest = self._resolve_image_digest(cache_ref) if exit_code == 0 else ""
        return ImageBuildResult(
            image_ref=digest or request.image_tag,
            image_digest=digest,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            termination_reason=termination_reason,
            cache_hit=False,
            environment_fingerprint=fingerprint,
            cache_ref=cache_ref,
        )

    def _pull_base_image(
        self, image: str, request: ImageBuildRequest
    ) -> ImageBuildResult | None:
        """Pull a missing base image; only reachable in online build mode.

        Returns ``None`` on success (or when the pull was cancelled and the
        caller should retry the whole build later).  A non-``None`` result is
        the failure payload ``build_image`` should return to the agent.
        """

        cancelled = bool(
            request.cancellation_event is not None
            and request.cancellation_event.is_set()
        )
        if cancelled:
            return ImageBuildResult(
                image_ref=request.image_tag,
                image_digest="",
                exit_code=130,
                stderr=f"base image pull cancelled before start: {image}",
                termination_reason="cancelled_by_controller",
            )
        try:
            completed = subprocess.run(
                [self.docker_binary, "pull", image],
                capture_output=True,
                text=True,
                timeout=max(60, min(request.timeout_seconds, 1800)),
            )
        except subprocess.TimeoutExpired:
            return ImageBuildResult(
                image_ref=request.image_tag,
                image_digest="",
                exit_code=124,
                stderr=f"base image pull timed out: {image}",
                termination_reason="timeout_killed",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ImageBuildResult(
                image_ref=request.image_tag,
                image_digest="",
                exit_code=125,
                stderr=f"base image pull failed: {image}: {exc}",
            )
        if completed.returncode != 0 or not self._image_exists(image):
            return ImageBuildResult(
                image_ref=request.image_tag,
                image_digest="",
                exit_code=125,
                stderr=(
                    f"base image pull failed: {image}: "
                    + (completed.stderr or completed.stdout)[-1000:]
                ),
            )
        return None

    def _image_exists(self, image: str) -> bool:
        try:
            result = subprocess.run(
                [self.docker_binary, "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0
