from __future__ import annotations

from repro_agent.domain.enums import ExperimentTier, ToleranceType
from repro_agent.domain.experiment import ExperimentRun, MetricComparison
from repro_agent.domain.verification import VerificationRecord
from repro_agent.observability.assembler import ReportAssembler
from repro_agent.storage.database import Database, dumps
from repro_agent.storage.repository import (
    ExperimentRunRepository,
    JobRepository,
    VerificationRepository,
)


def test_report_assembler_includes_runs_metrics_and_evidence(tmp_path, job) -> None:
    database = Database(tmp_path / "report.db")
    JobRepository(database).save(job)
    ExperimentRunRepository(database).save(
        ExperimentRun(
            job_id=job.job_id,
            experiment_id="main",
            tier=ExperimentTier.FULL_EXPERIMENT,
            exit_code=0,
            run_type="mock",
        )
    )
    comparison = MetricComparison(
        metric="accuracy",
        paper_value=0.9,
        reproduced_value=0.88,
        tolerance_type=ToleranceType.ABSOLUTE,
        tolerance=0.01,
        within_tolerance=False,
    )
    VerificationRepository(database).save(
        VerificationRecord(
            job_id=job.job_id,
            task_id="verification-task",
            run_id="run",
            expected_metric_names=["accuracy"],
            observed_metric_names=["accuracy"],
            missing_metrics=[],
            comparisons=[comparison],
            run_actually_executed=True,
            provenance_verified=False,
            anti_cheat_passed=True,
            is_fully_traceable=False,
            mock=True,
        )
    )
    with database.transaction() as cursor:
        cursor.execute(
            "INSERT INTO evidence_records (evidence_id, job_id, task_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("e1", job.job_id, "verification-task", "test", dumps({"sha256": "abc"}), "now"),
        )

    inputs = ReportAssembler(database).build(job.job_id)

    assert inputs.experiment_runs
    assert inputs.final_comparisons[0].metric == "accuracy"
    assert inputs.evidence_records[0]["payload"]["sha256"] == "abc"
    assert inputs.is_mock is True
