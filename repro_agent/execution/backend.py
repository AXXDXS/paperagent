"""Command execution contracts shared by real and mock backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import threading


@dataclass(frozen=True)
class ExecutionResourcePolicy:
    cpu_cores: float = 1.0
    memory_mb: int = 1024
    disk_mb: int = 4096
    max_processes: int = 32
    max_open_files: int = 256
    max_log_bytes: int = 20_000_000
    tmpfs_mb: int = 256
    gpu_memory_mb: int = 0


@dataclass(frozen=True)
class ExecutionRequest:
    task_id: str
    attempt_id: str
    command: list[str]
    image: str
    input_dir: Path
    workspace_dir: Path
    output_dir: Path
    timeout_seconds: int = 600
    resources: ExecutionResourcePolicy = field(default_factory=ExecutionResourcePolicy)
    environment: dict[str, str] = field(default_factory=dict)
    # Names only: the backend inherits values from the controller process so
    # credential values never enter task payloads, SQLite or command argv.
    passthrough_environment: list[str] = field(default_factory=list)
    working_dir: str = "."
    workspace_read_only: bool = False
    gpu_count: int = 0
    network_enabled: bool = False
    cancellation_event: threading.Event | None = None
    state_path: Path | None = None


@dataclass
class ExecutionResult:
    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    container_name: str = ""
    image_digest: str = ""
    termination_reason: str = "completed"
    mock: bool = False
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    execution_state_path: str = ""


@dataclass(frozen=True)
class ImageBuildRequest:
    task_id: str
    attempt_id: str
    context_dir: Path
    dockerfile: Path
    image_tag: str
    timeout_seconds: int = 900
    max_log_bytes: int = 20_000_000
    cancellation_event: threading.Event | None = None
    log_dir: Path | None = None
    # Bypass the content-addressed environment cache.  This is used only when
    # a cached image was found but failed the import smoke test.
    force_rebuild: bool = False
    # "Route A" build-time networking: allow the build itself to reach package
    # indexes (pip online resolution) and, if needed, pull a missing base
    # image.  Experiment *runtime* isolation is unaffected: ExecutionRequest
    # keeps its own independent network_enabled flag (default False).
    network_enabled: bool = False


@dataclass
class ImageBuildResult:
    image_ref: str
    image_digest: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    mock: bool = False
    termination_reason: str = "completed"
    cache_hit: bool = False
    environment_fingerprint: str = ""
    cache_ref: str = ""


@dataclass(frozen=True)
class CondaEnvironmentBuildRequest:
    """Create one fingerprint-validated, controller-managed Conda prefix."""

    task_id: str
    attempt_id: str
    requirements_file: Path
    environment_name: str = ""
    python_version: str = "3.11"
    timeout_seconds: int = 1800
    max_log_bytes: int = 20_000_000
    force_rebuild: bool = False
    network_enabled: bool = False
    wheel_dirs: tuple[Path, ...] = ()
    cancellation_event: threading.Event | None = None


@dataclass
class CondaEnvironmentBuildResult:
    environment_ref: str
    environment_digest: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    mock: bool = False
    termination_reason: str = "completed"
    cache_hit: bool = False
    environment_fingerprint: str = ""
    cache_ref: str = ""
    package_manifest_digest: str = ""
    environment_name: str = ""


class ExecutionBackend(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        ...

    def build_image(self, request: ImageBuildRequest) -> ImageBuildResult:
        ...

    def build_conda_environment(
        self, request: CondaEnvironmentBuildRequest
    ) -> CondaEnvironmentBuildResult:
        ...

    def cancel(self, container_name: str) -> bool:
        ...
