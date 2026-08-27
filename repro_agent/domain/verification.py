"""Persisted, independently auditable verification verdicts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.domain.experiment import MetricComparison


@dataclass
class VerificationRecord:
    job_id: str
    task_id: str
    run_id: str
    expected_metric_names: list[str]
    observed_metric_names: list[str]
    missing_metrics: list[str]
    comparisons: list[MetricComparison]
    run_actually_executed: bool
    provenance_verified: bool
    anti_cheat_passed: bool
    is_fully_traceable: bool
    mock: bool = False
    verification_valid: bool = False
    verification_id: str = field(default_factory=lambda: new_id("verification"))
    completed_at: datetime = field(default_factory=utc_now)

    @property
    def gap_fingerprint(self) -> str:
        data = [item.to_dict() for item in self.comparisons if not item.within_tolerance]
        encoded = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "expected_metric_names": self.expected_metric_names,
            "observed_metric_names": self.observed_metric_names,
            "missing_metrics": self.missing_metrics,
            "comparisons": [item.to_dict() for item in self.comparisons],
            "run_actually_executed": self.run_actually_executed,
            "provenance_verified": self.provenance_verified,
            "anti_cheat_passed": self.anti_cheat_passed,
            "is_fully_traceable": self.is_fully_traceable,
            "mock": self.mock,
            "verification_valid": self.verification_valid,
            "gap_fingerprint": self.gap_fingerprint,
            "completed_at": iso(self.completed_at),
        }
