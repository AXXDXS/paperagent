from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from repro_agent.schemas.results import ResultValidationError, TaskResultEnvelope


def test_result_envelope_rejects_stale_attempt(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task_1",
                "attempt_id": "attempt_old",
                "task_type": "paper_analysis",
                "outcome": "succeeded",
                "payload": {"extracted_parameters": []},
                "artifacts": [],
                "evidence_refs": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResultValidationError, match="attempt_id"):
        TaskResultEnvelope.from_file(
            path,
            expected_task_id="task_1",
            expected_attempt_id="attempt_current",
            expected_task_type="paper_analysis",
        )


def test_result_envelope_rejects_empty_specification_payload(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task_1",
                "attempt_id": "attempt_1",
                "task_type": "specification",
                "outcome": "succeeded",
                "payload": {},
                "artifacts": [],
                "evidence_refs": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResultValidationError, match="experiment_id"):
        TaskResultEnvelope.from_file(
            path,
            expected_task_id="task_1",
            expected_attempt_id="attempt_1",
            expected_task_type="specification",
        )


def test_result_envelope_verifies_artifact_digest_and_size(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_bytes(b'{"accuracy": 0.9}')
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task_1",
                "attempt_id": "attempt_1",
                "task_type": "paper_analysis",
                "outcome": "succeeded",
                "payload": {"extracted_parameters": []},
                "artifacts": [
                    {
                        "path": "metrics.json",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResultValidationError, match="digest"):
        TaskResultEnvelope.from_file(
            path,
            expected_task_id="task_1",
            expected_attempt_id="attempt_1",
            expected_task_type="paper_analysis",
        )


def test_result_envelope_accepts_verified_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_bytes(b'{"accuracy": 0.9}')
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task_1",
                "attempt_id": "attempt_1",
                "task_type": "paper_analysis",
                "outcome": "succeeded",
                "payload": {"extracted_parameters": []},
                "artifacts": [
                    {
                        "path": "metrics.json",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    envelope = TaskResultEnvelope.from_file(
        path,
        expected_task_id="task_1",
        expected_attempt_id="attempt_1",
        expected_task_type="paper_analysis",
    )

    assert envelope.artifacts[0].path == "metrics.json"
