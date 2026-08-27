"""Deterministic command backend for explicitly mock jobs."""

from __future__ import annotations

from datetime import datetime, timezone

from repro_agent.execution.backend import (
    CondaEnvironmentBuildRequest,
    CondaEnvironmentBuildResult,
    ExecutionRequest,
    ExecutionResult,
    ImageBuildRequest,
    ImageBuildResult,
)


class MockExecutionBackend:
    def cancel(self, container_name: str) -> bool:
        return True

    def build_image(self, request: ImageBuildRequest) -> ImageBuildResult:
        return ImageBuildResult(
            image_ref=f"{request.image_tag}@mock",
            image_digest="mock",
            exit_code=0,
            stdout="mock image build completed",
            mock=True,
            termination_reason="completed",
        )

    def build_conda_environment(
        self, request: CondaEnvironmentBuildRequest
    ) -> CondaEnvironmentBuildResult:
        fingerprint = "0" * 64
        return CondaEnvironmentBuildResult(
            environment_ref=f"conda://{fingerprint}",
            environment_digest=fingerprint,
            exit_code=0,
            stdout="mock Conda environment build completed",
            mock=True,
            cache_hit=False,
            environment_fingerprint=fingerprint,
            cache_ref=f"conda://{fingerprint}",
            package_manifest_digest=fingerprint,
            environment_name=request.environment_name,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        now = datetime.now(timezone.utc).isoformat()
        return ExecutionResult(
            command=request.command,
            exit_code=0,
            stdout="mock execution completed",
            container_name=f"mock-{request.attempt_id}",
            image_digest="mock",
            mock=True,
            started_at=now,
            completed_at=now,
            execution_state_path=str(request.state_path or ""),
        )
