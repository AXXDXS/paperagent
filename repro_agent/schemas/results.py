"""Versioned, attempt-bound task result contracts."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ResultValidationError(ValueError):
    """A task result cannot be trusted or consumed."""


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


_REQUIRED_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    "paper_analysis": ("extracted_parameters",),
    "code_analysis": ("entry_points",),
    "resource_check": ("blocking_issues",),
    "environment_build": ("import_test_passed",),
    "experiment_execution": ("tier", "exit_code", "run_id"),
    "verification": ("comparisons", "run_actually_executed"),
    "reflection": ("hypotheses",),
}


@dataclass
class TaskResultEnvelope:
    schema_version: int
    task_id: str
    attempt_id: str
    task_type: str
    outcome: str
    payload: dict[str, Any]
    artifacts: list[ArtifactReference] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def succeeded(
        cls,
        *,
        task_id: str,
        attempt_id: str,
        task_type: str,
        payload: dict[str, Any],
    ) -> "TaskResultEnvelope":
        return cls(
            schema_version=1,
            task_id=task_id,
            attempt_id=attempt_id,
            task_type=task_type,
            outcome="succeeded",
            payload=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "task_type": self.task_type,
            "outcome": self.outcome,
            "payload": self.payload,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "evidence_refs": self.evidence_refs,
            "warnings": self.warnings,
        }

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_task_id: str,
        expected_attempt_id: str,
        expected_task_type: str,
    ) -> "TaskResultEnvelope":
        result_path = Path(path)
        if not result_path.is_file() or result_path.stat().st_size == 0:
            raise ResultValidationError("result file is missing or empty")
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultValidationError(f"result is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ResultValidationError("result envelope must be a JSON object")
        if data.get("schema_version") != 1:
            raise ResultValidationError("unsupported result schema_version")
        for key, expected in (
            ("task_id", expected_task_id),
            ("attempt_id", expected_attempt_id),
            ("task_type", expected_task_type),
        ):
            if data.get(key) != expected:
                raise ResultValidationError(f"result {key} does not match active task")
        if data.get("outcome") != "succeeded":
            raise ResultValidationError("result outcome is not succeeded")
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise ResultValidationError("result payload must be an object")
        for required in _REQUIRED_PAYLOAD_FIELDS.get(expected_task_type, ()):
            if required not in payload:
                raise ResultValidationError(f"payload missing required field: {required}")
        if expected_task_type == "specification":
            is_full_spec = "experiment_id" in payload and "target_claim" in payload
            is_audit = "unresolved_conflicts" in payload and "fields" in payload
            if not (is_full_spec or is_audit):
                raise ResultValidationError("payload missing required field: experiment_id")
        artifacts = []
        for item in data.get("artifacts", []):
            try:
                reference = ArtifactReference(
                    path=str(item["path"]),
                    size_bytes=int(item["size_bytes"]),
                    sha256=str(item["sha256"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ResultValidationError("malformed artifact reference") from exc
            relative = Path(reference.path)
            output_root = result_path.parent.resolve()
            artifact_path = (output_root / relative).resolve()
            if relative.is_absolute() or not artifact_path.is_relative_to(output_root):
                raise ResultValidationError("artifact path escapes task output directory")
            if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                raise ResultValidationError(f"artifact is missing or empty: {reference.path}")
            actual_size = artifact_path.stat().st_size
            if actual_size != reference.size_bytes:
                raise ResultValidationError(
                    f"artifact size mismatch for {reference.path}"
                )
            digest = hashlib.sha256()
            with artifact_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != reference.sha256.lower():
                raise ResultValidationError(
                    f"artifact digest mismatch for {reference.path}"
                )
            artifacts.append(reference)
        return cls(
            schema_version=1,
            task_id=expected_task_id,
            attempt_id=expected_attempt_id,
            task_type=expected_task_type,
            outcome="succeeded",
            payload=payload,
            artifacts=artifacts,
            evidence_refs=[str(value) for value in data.get("evidence_refs", [])],
            warnings=[str(value) for value in data.get("warnings", [])],
        )
