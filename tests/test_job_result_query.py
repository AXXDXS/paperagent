from __future__ import annotations

import hashlib
import json

import pytest

from repro_agent.domain.enums import JobStatus, ReproductionStatus, TaskStatus
from repro_agent.domain.task import Task, TaskDefinition
from repro_agent.observability.result_query import JobResultIntegrityError, JobResultService
from repro_agent.orchestrator.result_tools import GetJobResultTool
from repro_agent.schemas.results import TaskResultEnvelope
from repro_agent.storage.database import Database
from repro_agent.storage.repository import EvidenceRepository, JobRepository, TaskRepository
from repro_agent.tools.registry import default_registry


def _persist_completed_job(tmp_path, job):
    work_dir = tmp_path / "work"
    database = Database(work_dir / "repro_agent.db")
    job.status = JobStatus.FULLY_REPRODUCED
    job.final_reproduction_status = ReproductionStatus.FULLY_REPRODUCED
    JobRepository(database).save(job)

    task = Task(
        job_id=job.job_id,
        definition=TaskDefinition(
            objective="run full experiment",
            task_type="experiment_execution",
            expected_outputs=["output/result.json"],
        ),
        status=TaskStatus.SUCCEEDED,
        attempt=1,
        active_attempt_id="attempt-current",
    )
    output_dir = (
        work_dir
        / "sandbox"
        / job.job_id
        / "sandbox"
        / f"task_{task.task_id}"
        / task.active_attempt_id
        / "output"
    )
    output_dir.mkdir(parents=True)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(
            TaskResultEnvelope.succeeded(
                task_id=task.task_id,
                attempt_id=task.active_attempt_id,
                task_type=task.definition.task_type,
                payload={"tier": "full_experiment", "exit_code": 0, "run_id": "run-1"},
            ).to_dict(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    task.outputs = {"result.json": str(result_path)}
    TaskRepository(database).save(task)
    content = result_path.read_bytes()
    EvidenceRepository(database).record(
        job_id=job.job_id,
        task_id=task.task_id,
        kind="task_artifact",
        payload={
            "attempt_id": task.active_attempt_id,
            "task_type": task.definition.task_type,
            "path": str(result_path),
            "relative_path": "result.json",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        },
    )
    return work_dir, database, task, result_path


def test_job_result_service_writes_job_scoped_reports_and_verifies_current_artifacts(
    tmp_path, job
) -> None:
    work_dir, database, task, result_path = _persist_completed_job(tmp_path, job)
    try:
        result = JobResultService(database, work_dir).get(job.job_id)
    finally:
        database.close()

    assert result.job_id == job.job_id
    assert result.job_status == "FULLY_REPRODUCED"
    assert result.integrity_ok is True
    assert result.report_paths == {
        "markdown": str(work_dir / "reports" / job.job_id / "final_report.md"),
        "json": str(work_dir / "reports" / job.job_id / "final_report.json"),
    }
    assert (work_dir / "reports" / job.job_id / "final_report.md").is_file()
    assert (work_dir / "reports" / job.job_id / "final_report.json").is_file()
    assert result.artifacts[0].task_id == task.task_id
    assert result.artifacts[0].attempt_id == "attempt-current"
    assert result.artifacts[0].path == str(result_path)
    assert result.artifacts[0].integrity == "verified"


def test_job_result_service_fails_closed_when_artifact_digest_changed(tmp_path, job) -> None:
    work_dir, database, _task, result_path = _persist_completed_job(tmp_path, job)
    tampered = bytearray(result_path.read_bytes())
    tampered[-1] = ord(" ") if tampered[-1] != ord(" ") else ord("\n")
    result_path.write_bytes(tampered)
    try:
        with pytest.raises(JobResultIntegrityError) as exc_info:
            JobResultService(database, work_dir).get(job.job_id)
    finally:
        database.close()

    assert "sha256 mismatch" in str(exc_info.value)


def test_job_result_service_rejects_symlinked_artifact_inside_work_dir(tmp_path, job) -> None:
    work_dir, database, task, result_path = _persist_completed_job(tmp_path, job)
    link_path = result_path.parent / "result-link.json"
    link_path.symlink_to(result_path.name)
    task.outputs["result-link.json"] = str(link_path)
    TaskRepository(database).save(task)
    content = result_path.read_bytes()
    EvidenceRepository(database).record(
        job_id=job.job_id,
        task_id=task.task_id,
        kind="task_artifact",
        payload={
            "attempt_id": task.active_attempt_id,
            "task_type": task.definition.task_type,
            "path": str(link_path),
            "relative_path": "result-link.json",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        },
    )
    try:
        with pytest.raises(JobResultIntegrityError) as exc_info:
            JobResultService(database, work_dir).get(job.job_id)
    finally:
        database.close()

    assert "missing or unsafe" in str(exc_info.value)


def test_get_job_result_tool_is_main_only_and_cannot_query_another_job(tmp_path, job) -> None:
    work_dir, database, _task, _result_path = _persist_completed_job(tmp_path, job)
    try:
        tool = GetJobResultTool(
            JobResultService(database, work_dir), current_job_id=job.job_id
        )
        payload = tool.call(job_id=job.job_id)
        assert payload["job_id"] == job.job_id
        with pytest.raises(PermissionError):
            tool.call(job_id="job-other")
    finally:
        database.close()

    assert tool.to_openai_tool()["function"]["name"] == "get_job_result"
    assert default_registry().get("get_job_result") is None
