from __future__ import annotations

from repro_agent.domain.enums import ToleranceType
from repro_agent.domain.experiment import MetricComparison
from repro_agent.domain.verification import VerificationRecord
from repro_agent.storage.database import Database
from repro_agent.storage.repository import VerificationRepository


def test_verification_record_roundtrip(tmp_path) -> None:
    repository = VerificationRepository(Database(tmp_path / "records.db"))
    record = VerificationRecord(
        job_id="job",
        task_id="task",
        run_id="run",
        expected_metric_names=["accuracy", "f1"],
        observed_metric_names=["accuracy"],
        missing_metrics=["f1"],
        comparisons=[
            MetricComparison(
                metric="accuracy",
                paper_value=0.9,
                reproduced_value=0.89,
                tolerance_type=ToleranceType.ABSOLUTE,
                tolerance=0.02,
                within_tolerance=True,
            )
        ],
        run_actually_executed=True,
        provenance_verified=False,
        anti_cheat_passed=True,
        is_fully_traceable=False,
    )

    repository.save(record)
    loaded = repository.list_by_job("job")

    assert len(loaded) == 1
    assert loaded[0].missing_metrics == ["f1"]
    assert loaded[0].comparisons[0].metric == "accuracy"
