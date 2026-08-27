from __future__ import annotations

from pathlib import Path

from repro_agent.domain.enums import (
    AuditResultType,
    JobStatus,
    ReproductionStatus,
    RerunScope,
)
from repro_agent.domain.job import JobInputs, ReproductionJob
from repro_agent.domain.reflection import ReflectionReport
from repro_agent.domain.task import Heartbeat, Task, TaskDefinition
from repro_agent.storage.database import Database
from repro_agent.storage.repository import JobRepository, ReflectionRepository, TaskRepository


def test_task_roundtrip_preserves_behavior_affecting_fields(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    JobRepository(database).save(
        ReproductionJob(
            job_id="job_1",
            inputs=JobInputs(paper_path="paper.txt", repository_path="repo"),
        )
    )
    task = Task(
        job_id="job_1",
        definition=TaskDefinition(
            objective="persist all scheduling inputs",
            task_type="paper_analysis",
            liveness_grace_seconds=7,
            priority=9,
        ),
    )
    task.active_attempt_id = "attempt_current"
    task.last_activity_signature = "log:42"
    task.last_push_heartbeat = Heartbeat(progress=0.4, reported_by="push", eta_seconds=12)
    task.last_pull_heartbeat = Heartbeat(progress=0.4, reported_by="pull", eta_seconds=10)
    task.heartbeat = task.last_push_heartbeat

    repo = TaskRepository(database)
    repo.save(task)
    loaded = repo.get(task.task_id)

    assert loaded is not None
    assert loaded.definition.liveness_grace_seconds == 7
    assert loaded.definition.priority == 9
    assert loaded.active_attempt_id == "attempt_current"
    assert loaded.last_activity_signature == "log:42"
    assert loaded.last_push_heartbeat.reported_by == "push"
    assert loaded.last_push_heartbeat.eta_seconds == 12
    assert loaded.last_pull_heartbeat.reported_by == "pull"
    assert loaded.last_pull_heartbeat.eta_seconds == 10


def test_job_roundtrip_preserves_final_reproduction_status(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    job = ReproductionJob(
        inputs=JobInputs(paper_path="paper.txt", repository_path="repo"),
        status=JobStatus.FULLY_REPRODUCED,
        final_reproduction_status=ReproductionStatus.FULLY_REPRODUCED,
    )

    repo = JobRepository(database)
    repo.save(job)
    loaded = repo.get(job.job_id)

    assert loaded is not None
    assert loaded.final_reproduction_status == ReproductionStatus.FULLY_REPRODUCED


def test_reflection_roundtrip_returns_domain_object_with_pipeline_state(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    report = ReflectionReport(
        job_id="job_1",
        round=2,
        likely_source="configuration",
        audit_result=AuditResultType.CONFIG_ERROR_CONFIRMED,
        recommended_rerun_scope=RerunScope.EVALUATION_ONLY,
        repair_task_ids=["repair_1"],
        rerun_triggered=True,
    )

    repo = ReflectionRepository(database)
    repo.save(report)
    loaded = repo.list_by_job("job_1")

    assert len(loaded) == 1
    restored = loaded[0]
    assert isinstance(restored, ReflectionReport)
    assert restored.likely_source == "configuration"
    assert restored.repair_task_ids == ["repair_1"]
    assert restored.rerun_triggered is True
