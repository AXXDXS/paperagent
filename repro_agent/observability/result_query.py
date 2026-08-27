"""Accurate, job-scoped lookup and delivery of persisted experiment results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from repro_agent.domain.enums import TaskStatus
from repro_agent.observability.assembler import ReportAssembler
from repro_agent.observability.report import FinalReportGenerator
from repro_agent.schemas.results import ResultValidationError, TaskResultEnvelope
from repro_agent.storage.repository import EvidenceRepository, TaskRepository


class JobResultNotFoundError(KeyError):
    """The requested job is not present in the selected work directory."""


class JobResultIntegrityError(RuntimeError):
    """A persisted artifact cannot be safely attributed to the active attempt."""


@dataclass(frozen=True)
class VerifiedArtifact:
    task_id: str
    task_type: str
    attempt_id: str
    relative_path: str
    path: str
    sha256: str
    size_bytes: int
    integrity: str = "verified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobResult:
    job_id: str
    job_status: str
    reproduction_status: str | None
    final_conclusion: str
    is_mock: bool
    metric_comparisons: list[dict[str, Any]]
    experiment_runs: list[dict[str, Any]]
    artifacts: list[VerifiedArtifact]
    report_paths: dict[str, str]
    integrity_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_status": self.job_status,
            "reproduction_status": self.reproduction_status,
            "final_conclusion": self.final_conclusion,
            "mock": self.is_mock,
            "metric_comparisons": list(self.metric_comparisons),
            "experiment_runs": list(self.experiment_runs),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "report_paths": dict(self.report_paths),
            "integrity_ok": self.integrity_ok,
        }


class JobResultService:
    """Resolve one job through SQLite and verify every current task artifact."""

    def __init__(self, database, work_dir: str | Path):
        self.database = database
        self.work_dir = Path(work_dir)
        self.tasks = TaskRepository(database)
        self.evidence = EvidenceRepository(database)

    def get(self, job_id: str, *, write_legacy_reports: bool = False) -> JobResult:
        assembler = ReportAssembler(self.database)
        try:
            inputs = assembler.build(job_id)
        except KeyError as exc:
            raise JobResultNotFoundError(str(exc)) from exc

        artifacts = self._verify_current_artifacts(job_id)
        report_dir = self.work_dir / "reports" / job_id
        report_paths = {
            "markdown": str(report_dir / "final_report.md"),
            "json": str(report_dir / "final_report.json"),
        }
        result = JobResult(
            job_id=job_id,
            job_status=inputs.job.status.value,
            reproduction_status=(
                inputs.job.final_reproduction_status.value
                if inputs.job.final_reproduction_status is not None
                else None
            ),
            final_conclusion=inputs.final_conclusion,
            is_mock=inputs.is_mock,
            metric_comparisons=[item.to_dict() for item in inputs.final_comparisons],
            experiment_runs=[run.to_dict() for run in inputs.experiment_runs],
            artifacts=artifacts,
            report_paths=report_paths,
        )
        self._write_reports(inputs, result, write_legacy_reports=write_legacy_reports)
        return result

    def _verify_current_artifacts(self, job_id: str) -> list[VerifiedArtifact]:
        evidence_index: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in self.evidence.list_by_job(job_id):
            payload = record.get("payload") or {}
            if record.get("kind") != "task_artifact" or not isinstance(payload, dict):
                continue
            key = (
                str(record.get("task_id") or ""),
                str(payload.get("attempt_id") or ""),
                str(payload.get("relative_path") or ""),
            )
            evidence_index[key] = payload

        verified: list[VerifiedArtifact] = []
        allowed_root = self.work_dir.resolve()
        for task in self.tasks.list_by_job(job_id):
            if task.status != TaskStatus.SUCCEEDED:
                continue
            for relative_path, stored_path in sorted(task.outputs.items()):
                stored = Path(str(stored_path))
                if stored.is_symlink():
                    raise JobResultIntegrityError(
                        f"artifact is missing or unsafe: task={task.task_id} path={stored}"
                    )
                path = stored.resolve()
                if not path.is_relative_to(allowed_root):
                    raise JobResultIntegrityError(
                        f"artifact escapes work directory: task={task.task_id} path={path}"
                    )
                if not path.is_file():
                    raise JobResultIntegrityError(
                        f"artifact is missing or unsafe: task={task.task_id} path={path}"
                    )
                evidence = evidence_index.get(
                    (task.task_id, task.active_attempt_id, relative_path)
                )
                if evidence is None:
                    raise JobResultIntegrityError(
                        "artifact has no evidence for active attempt: "
                        f"task={task.task_id} attempt={task.active_attempt_id} "
                        f"path={relative_path}"
                    )
                evidence_path = Path(str(evidence.get("path", ""))).resolve()
                if evidence_path != path:
                    raise JobResultIntegrityError(
                        f"artifact path mismatch: task={task.task_id} path={relative_path}"
                    )
                size_bytes = path.stat().st_size
                expected_size = int(evidence.get("size_bytes", -1))
                if size_bytes != expected_size:
                    raise JobResultIntegrityError(
                        f"artifact size mismatch: task={task.task_id} path={relative_path}"
                    )
                digest = self._sha256(path)
                expected_digest = str(evidence.get("sha256", "")).lower()
                if digest != expected_digest:
                    raise JobResultIntegrityError(
                        f"artifact sha256 mismatch: task={task.task_id} path={relative_path}"
                    )
                if relative_path == "result.json":
                    try:
                        TaskResultEnvelope.from_file(
                            path,
                            expected_task_id=task.task_id,
                            expected_attempt_id=task.active_attempt_id,
                            expected_task_type=task.definition.task_type,
                        )
                    except ResultValidationError as exc:
                        raise JobResultIntegrityError(
                            f"result contract mismatch for task {task.task_id}: {exc}"
                        ) from exc
                verified.append(
                    VerifiedArtifact(
                        task_id=task.task_id,
                        task_type=task.definition.task_type,
                        attempt_id=task.active_attempt_id,
                        relative_path=relative_path,
                        path=str(path),
                        sha256=digest,
                        size_bytes=size_bytes,
                    )
                )
        return verified

    def _write_reports(self, inputs, result: JobResult, *, write_legacy_reports: bool) -> None:
        report_dir = self.work_dir / "reports" / result.job_id
        report_dir.mkdir(parents=True, exist_ok=True)
        markdown = FinalReportGenerator().generate(inputs)
        report_json = ReportAssembler.to_json(inputs)
        report_json["result_index"] = result.to_dict()
        markdown_path = report_dir / "final_report.md"
        json_path = report_dir / "final_report.json"
        markdown_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(
            json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if write_legacy_reports:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "final_report.md").write_text(markdown, encoding="utf-8")
            (self.work_dir / "final_report.json").write_text(
                json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
