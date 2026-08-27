"""Colima-backed execution using the Docker CLI compatibility surface.

Colima owns the Linux VM and Docker daemon on macOS.  ReproAgent continues to
use the audited Docker argv builder, but refuses to execute until all three
control-plane components are ready: the ``colima`` CLI, a running Colima VM,
and a Docker client connected to that VM.
"""

from __future__ import annotations

import shutil
import subprocess

from repro_agent.execution.docker import DockerExecutionBackend


class ColimaExecutionBackend(DockerExecutionBackend):
    """Run Docker-compatible workloads against a user-managed Colima VM.

    The backend deliberately does not install or start Colima.  Starting a VM
    is an external lifecycle decision, so failures return an actionable message
    instead of mutating host state or silently executing on the host.
    """

    def __init__(
        self,
        *,
        colima_binary: str = "colima",
        docker_binary: str = "docker",
        probe_timeout_seconds: float = 10.0,
    ):
        super().__init__(docker_binary=docker_binary)
        self.colima_binary = colima_binary
        self.probe_timeout_seconds = probe_timeout_seconds

    def is_available(self) -> bool:
        return self._availability_error() is None

    def unavailable_reason(self, *, purpose: str = "container execution") -> str:
        reason = self._availability_error()
        if reason is None:
            return f"Colima is unavailable for {purpose}"
        return f"{reason} Required for {purpose}."

    def _availability_error(self) -> str | None:
        if shutil.which(self.colima_binary) is None:
            return (
                f"Colima CLI '{self.colima_binary}' was not found on PATH. "
                "Install it with 'brew install colima docker', then run 'colima start'."
            )
        if shutil.which(self.docker_binary) is None:
            return (
                f"Docker CLI '{self.docker_binary}' was not found on PATH. "
                "Install the client with 'brew install docker'."
            )
        if not self._probe([self.colima_binary, "status"]):
            return "Colima VM is not running. Start it with 'colima start'."
        if not self._probe([self.docker_binary, "info"]):
            return (
                "Docker cannot connect to the Colima daemon. Run 'colima status' and "
                "select the Colima Docker context before retrying."
            )
        return None

    def _probe(self, argv: list[str]) -> bool:
        try:
            result = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.probe_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0
