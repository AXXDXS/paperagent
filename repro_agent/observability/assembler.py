"""Build reports only from persisted, validated repository records."""

from __future__ import annotations

from typing import Any

from repro_agent.domain.enums import TaskStatus
from repro_agent.observability.conclusion import render_conclusion_text
from repro_agent.observability.report import ReportInputs
from repro_agent.schemas.results import ResultValidationError, TaskResultEnvelope
from repro_agent.storage.database import Database, loads
from repro_agent.storage.repository import (
    ExperimentRunRepository,
    InterventionRepository,
    JobRepository,
    ReflectionRepository,
    TaskRepository,
    VerificationRepository,
)


class ReportAssembler:
    def __init__(self, database: Database):
        self.database = database
        self.jobs = JobRepository(database)
        self.tasks = TaskRepository(database)
        self.runs = ExperimentRunRepository(database)
        self.verifications = VerificationRepository(database)
        self.reflections = ReflectionRepository(database)
        self.interventions = InterventionRepository(database)

    def build(self, job_id: str) -> ReportInputs:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        tasks = self.tasks.list_by_job(job_id)
        runs = self.runs.list_by_job(job_id)
        verifications = self.verifications.list_by_job(job_id)
        latest_verification = verifications[-1] if verifications else None
        comparisons = latest_verification.comparisons if latest_verification else []
        payloads = {
            "paper_analysis": self._merged_paper_payload(tasks),
            "code_analysis": self._latest_payload(tasks, "code_analysis"),
            "specification": self._latest_payload(tasks, "specification"),
            "resource_check": self._latest_payload(tasks, "resource_check"),
            "environment_build": self._latest_payload(tasks, "environment_build"),
        }
        evidence = [
            {
                "evidence_id": row["evidence_id"],
                "task_id": row["task_id"],
                "kind": row["kind"],
                "payload": loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in self.database.query_all(
                "SELECT evidence_id, task_id, kind, payload, created_at "
                "FROM evidence_records WHERE job_id = ? ORDER BY created_at",
                (job_id,),
            )
        ]
        is_mock = any(run.run_type == "mock" for run in runs) or any(
            record.mock for record in verifications
        )
        final_conclusion = (
            render_conclusion_text(job.final_reproduction_status, comparisons)
            if job.final_reproduction_status is not None
            else "任务尚未形成可验证的最终结论。"
        )
        return ReportInputs(
            job=job,
            paper_analysis_output=payloads["paper_analysis"],
            code_analysis_output=payloads["code_analysis"],
            experiment_spec_output=payloads["specification"],
            resource_check_output=payloads["resource_check"],
            environment_build_output=payloads["environment_build"],
            experiment_runs=runs,
            final_comparisons=comparisons,
            failed_tasks=[task for task in tasks if task.status.is_failure],
            reflection_reports=self.reflections.list_by_job(job_id),
            final_conclusion=final_conclusion,
            verification_records=verifications,
            evidence_records=evidence,
            events=self.tasks.list_events(job_id),
            interventions=[
                {
                    "request_id": request.request_id,
                    "task_id": request.task_id,
                    "kind": request.kind.value,
                    "status": request.status.value,
                    "question": request.question,
                    "reason": request.reason,
                    "created_at": request.to_dict()["created_at"],
                    "expires_at": request.to_dict()["expires_at"],
                    "responded_at": request.to_dict()["responded_at"],
                    "responded_by": request.responded_by,
                    # 报告只暴露字段名；原始回答保留在受控 SQLite 中，
                    # 避免命令或路径里意外出现凭据后扩散到报告文件。
                    "response_fields": sorted(request.response),
                }
                for request in self.interventions.list_by_job(job_id)
            ],
            is_mock=is_mock,
        )

    @staticmethod
    def _merged_paper_payload(tasks) -> dict[str, Any]:
        """正文 + 附录（可能多片）paper_analysis 产物的合并视图。

        论文分析拆成并行子任务后，报告不能再只取“最新完成的一个”
        paper_analysis 任务（那几乎必然是附录任务，会丢掉正文的
        method_summary/expected_results）。这里复用与 ArtifactResolver
        同一份 scope-aware 合并逻辑，保证报告与下游规格看到的是同一份
        合并结果。"""

        from repro_agent.orchestrator.artifacts import merge_paper_findings

        payloads: list[dict[str, Any]] = []
        for task in tasks:
            if task.definition.task_type != "paper_analysis":
                continue
            if task.status != TaskStatus.SUCCEEDED:
                continue
            path = task.outputs.get("result.json")
            if not path:
                continue
            try:
                payloads.append(
                    TaskResultEnvelope.from_file(
                        path,
                        expected_task_id=task.task_id,
                        expected_attempt_id=task.active_attempt_id,
                        expected_task_type=task.definition.task_type,
                    ).payload
                )
            except ResultValidationError:
                continue
        return merge_paper_findings(payloads)

    @staticmethod
    def _latest_payload(tasks, task_type: str) -> dict[str, Any]:
        candidates = [
            task
            for task in tasks
            if task.definition.task_type == task_type
            and task.status == TaskStatus.SUCCEEDED
        ]
        for task in reversed(candidates):
            path = task.outputs.get("result.json")
            if not path:
                continue
            try:
                return TaskResultEnvelope.from_file(
                    path,
                    expected_task_id=task.task_id,
                    expected_attempt_id=task.active_attempt_id,
                    expected_task_type=task.definition.task_type,
                ).payload
            except ResultValidationError:
                continue
        return {}

    @staticmethod
    def to_json(inputs: ReportInputs) -> dict[str, Any]:
        return {
            "job": inputs.job.to_dict(),
            "mock": inputs.is_mock,
            "paper_analysis": inputs.paper_analysis_output,
            "code_analysis": inputs.code_analysis_output,
            "experiment_spec": inputs.experiment_spec_output,
            "resource_check": inputs.resource_check_output,
            "environment_build": inputs.environment_build_output,
            "experiment_runs": [run.to_dict() for run in inputs.experiment_runs],
            "metric_comparisons": [item.to_dict() for item in inputs.final_comparisons],
            "verification_records": [item.to_dict() for item in inputs.verification_records],
            "reflection_reports": [item.to_dict() for item in inputs.reflection_reports],
            "evidence_records": inputs.evidence_records,
            "events": inputs.events,
            "interventions": inputs.interventions,
            "final_conclusion": inputs.final_conclusion,
        }
