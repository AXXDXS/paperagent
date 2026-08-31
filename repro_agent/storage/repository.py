"""Job/Task/ExperimentRun/ReflectionReport 的仓储层。

所有对 domain 对象的读写都必须经过这里，保证"数据库是任务状态的
唯一事实来源"（设计文档 §3 原则 16）——上层（调度器、主智能体）
不应该绕过 Repository 直接操作内存对象后忘记落库。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.domain.enums import (
    AuditResultType,
    ExperimentTier,
    FailureType,
    InterventionKind,
    InterventionStatus,
    JobStatus,
    ReproductionStatus,
    RerunScope,
    TaskStatus,
    ToleranceType,
)
from repro_agent.domain.experiment import ExperimentRun, MetricComparison
from repro_agent.domain.intervention import InterventionRequest
from repro_agent.domain.job import JobBudget, JobInputs, ReproductionJob
from repro_agent.domain.reflection import (
    AuditFinding,
    ReflectionHypothesis,
    ReflectionReport,
)
from repro_agent.domain.task import (
    AgentReport,
    AgentReportType,
    FailureReport,
    Heartbeat,
    Task,
    TaskDefinition,
)
from repro_agent.domain.verification import VerificationRecord
from repro_agent.storage.database import Database, dumps, loads
from repro_agent.tools.base import ToolInvocationLog


def _parse_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


class JobRepository:
    """Job 的持久化读写。"""

    def __init__(self, db: Database):
        self.db = db

    def save(self, job: ReproductionJob) -> None:
        job.touch()
        payload = dumps(job.to_dict())
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO jobs (job_id, status, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    job.job_id,
                    job.status.value,
                    payload,
                    iso(job.created_at),
                    iso(job.updated_at),
                ),
            )

    def get(self, job_id: str) -> Optional[ReproductionJob]:
        row = self.db.query_one(
            "SELECT payload FROM jobs WHERE job_id = ?", (job_id,)
        )
        if row is None:
            return None
        return self._deserialize(loads(row["payload"]))

    def list_all(self) -> list[ReproductionJob]:
        rows = self.db.query_all("SELECT payload FROM jobs ORDER BY created_at")
        return [self._deserialize(loads(r["payload"])) for r in rows]

    @staticmethod
    def _deserialize(data: dict) -> ReproductionJob:
        inputs_data = data["inputs"]
        inputs = JobInputs(**inputs_data)
        budget = JobBudget(**data["budget"])
        job = ReproductionJob(
            inputs=inputs,
            job_id=data["job_id"],
            budget=budget,
            status=JobStatus(data["status"]),
            reflection_round=data.get("reflection_round", 0),
            full_experiment_rerun_count=data.get("full_experiment_rerun_count", 0),
            created_at=_parse_dt(data["created_at"]) or utc_now(),
            updated_at=_parse_dt(data["updated_at"]) or utc_now(),
            final_reproduction_status=ReproductionStatus(data["final_reproduction_status"])
            if data.get("final_reproduction_status")
            else None,
            gpu_hours_used=data.get("gpu_hours_used", 0.0),
            model_call_cost_usd=data.get("model_call_cost_usd", 0.0),
            model_input_tokens_used=data.get("model_input_tokens_used", 0),
            model_output_tokens_used=data.get("model_output_tokens_used", 0),
            model_calls_made=data.get("model_calls_made", 0),
        )
        return job


class TaskRepository:
    """Task 的持久化读写，是调度器判断 READY/超时/重试的唯一数据来源。"""

    def __init__(self, db: Database):
        self.db = db

    def save(self, task: Task) -> None:
        task.touch()
        with self.db.transaction() as cur:
            self._upsert(cur, task)

    def save_with_event(
        self,
        task: Task,
        event_type: str,
        event_payload: dict,
        *,
        event_key: str = "",
    ) -> None:
        """Atomically persist a state transition, attempt snapshot and event."""

        task.touch()
        with self.db.transaction() as cur:
            self._upsert(cur, task)
            self._insert_event(
                cur,
                task.job_id,
                task.task_id,
                event_type,
                event_payload,
                event_key=event_key,
            )

    @staticmethod
    def _upsert(cur, task: Task) -> None:
        creation_key = str(task.definition.inputs.get("creation_key", "") or "")
        cur.execute(
            """
            INSERT INTO tasks
                (task_id, job_id, status, task_type, creation_key, payload,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                status=excluded.status,
                task_type=excluded.task_type,
                creation_key=excluded.creation_key,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                task.task_id,
                task.job_id,
                task.status.value,
                task.definition.task_type,
                creation_key,
                dumps(task.to_dict()),
                iso(task.created_at),
                iso(task.updated_at),
            ),
        )
        # Persist ownership rather than keeping the lease solely in memory.
        if task.lease_owner and task.lease_expires_at:
            cur.execute(
                """
                INSERT INTO task_leases
                    (task_id, job_id, owner, expires_at, heartbeat_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    owner=excluded.owner,
                    expires_at=excluded.expires_at,
                    heartbeat_at=excluded.heartbeat_at
                """,
                (
                    task.task_id,
                    task.job_id,
                    task.lease_owner,
                    iso(task.lease_expires_at),
                    iso((task.heartbeat or task.last_push_heartbeat).updated_at)
                    if (task.heartbeat or task.last_push_heartbeat)
                    else iso(utc_now()),
                ),
            )
        else:
            cur.execute("DELETE FROM task_leases WHERE task_id = ?", (task.task_id,))

        if task.active_attempt_id:
            cur.execute(
                """
                INSERT INTO task_attempts
                    (attempt_id, job_id, task_id, attempt_number, status, payload,
                     started_at, completed_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    status=excluded.status,
                    payload=excluded.payload,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    updated_at=excluded.updated_at
                """,
                (
                    task.active_attempt_id,
                    task.job_id,
                    task.task_id,
                    task.attempt,
                    task.status.value,
                    dumps(task.to_dict()),
                    iso(task.started_at),
                    iso(task.completed_at),
                    iso(task.updated_at),
                ),
            )

    def get_by_creation_key(self, job_id: str, creation_key: str) -> Optional[Task]:
        if not creation_key:
            return None
        row = self.db.query_one(
            "SELECT payload FROM tasks WHERE job_id = ? AND creation_key = ?",
            (job_id, creation_key),
        )
        return self._deserialize(loads(row["payload"])) if row is not None else None

    def get(self, task_id: str) -> Optional[Task]:
        row = self.db.query_one(
            "SELECT payload FROM tasks WHERE task_id = ?", (task_id,)
        )
        if row is None:
            return None
        return self._deserialize(loads(row["payload"]))

    def list_by_job(self, job_id: str) -> list[Task]:
        rows = self.db.query_all(
            "SELECT payload FROM tasks WHERE job_id = ? ORDER BY created_at",
            (job_id,),
        )
        return [self._deserialize(loads(r["payload"])) for r in rows]

    def list_by_status(self, job_id: str, status: TaskStatus) -> list[Task]:
        rows = self.db.query_all(
            "SELECT payload FROM tasks WHERE job_id = ? AND status = ? ORDER BY created_at",
            (job_id, status.value),
        )
        return [self._deserialize(loads(r["payload"])) for r in rows]

    def record_event(
        self,
        job_id: str,
        task_id: Optional[str],
        event_type: str,
        payload: dict,
        *,
        event_key: str = "",
    ) -> None:
        """审计事件流水账（供最终报告 §20 第八/九部分回溯"错误和修复"）。"""

        with self.db.transaction() as cur:
            self._insert_event(
                cur, job_id, task_id, event_type, payload, event_key=event_key
            )

    @staticmethod
    def _insert_event(
        cur,
        job_id: str,
        task_id: Optional[str],
        event_type: str,
        payload: dict,
        *,
        event_key: str = "",
    ) -> None:
        cur.execute(
            """
            INSERT INTO task_events
                (job_id, task_id, event_type, event_key, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, event_key) WHERE event_key <> '' DO NOTHING
            """,
            (
                job_id,
                task_id,
                event_type,
                event_key,
                dumps(payload),
                iso(utc_now()),
            ),
        )

    def list_events(self, job_id: str) -> list[dict]:
        rows = self.db.query_all(
            "SELECT task_id, event_type, payload, created_at FROM task_events "
            "WHERE job_id = ? ORDER BY event_id",
            (job_id,),
        )
        return [
            {
                "task_id": r["task_id"],
                "event_type": r["event_type"],
                "payload": loads(r["payload"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    @staticmethod
    def _deserialize(data: dict) -> Task:
        definition = TaskDefinition(
            objective=data["objective"],
            task_type=data["task_type"],
            dependencies=data.get("dependencies", []),
            inputs=data.get("inputs", {}),
            allowed_tools=data.get("allowed_tools", []),
            forbidden_actions=data.get("forbidden_actions", []),
            expected_outputs=data.get("expected_outputs", []),
            completion_criteria=data.get("completion_criteria", []),
            expected_duration_seconds=data.get("expected_duration_seconds", 300),
            soft_timeout_seconds=data.get("soft_timeout_seconds", 600),
            hard_timeout_seconds=data.get("hard_timeout_seconds", 1200),
            heartbeat_interval_seconds=data.get("heartbeat_interval_seconds", 30),
            liveness_grace_seconds=data.get("liveness_grace_seconds", 120),
            max_overrun_reports=data.get("max_overrun_reports", 3),
            failure_report_required=data.get("failure_report_required", True),
            priority=data.get("priority", 0),
            max_attempts=data.get("max_attempts", 3),
            parent_task_id=data.get("parent_task_id"),
        )
        def parse_heartbeat(hb: dict | None) -> Heartbeat | None:
            if not hb:
                return None
            return Heartbeat(
                progress=hb.get("progress", 0.0),
                current_step=hb.get("current_step", ""),
                last_completed_step=hb.get("last_completed_step", ""),
                last_log_position=hb.get("last_log_position", 0),
                updated_at=_parse_dt(hb.get("updated_at")) or utc_now(),
                eta_seconds=hb.get("eta_seconds"),
                reported_by=hb.get("reported_by", "push"),
            )
        heartbeat = parse_heartbeat(data.get("heartbeat"))
        latest_agent_report = None
        if data.get("latest_agent_report"):
            report = data["latest_agent_report"]
            latest_agent_report = AgentReport(
                report_id=report.get("report_id") or new_id("report"),
                attempt_id=report.get("attempt_id", ""),
                sequence=report.get("sequence", 0),
                report_type=AgentReportType(
                    report.get("report_type", AgentReportType.PROGRESS.value)
                ),
                progress=report.get("progress", 0.0),
                current_step=report.get("current_step", ""),
                eta_seconds=report.get("eta_seconds"),
                next_report_after_seconds=report.get(
                    "next_report_after_seconds"
                ),
                reason=report.get("reason", ""),
                evidence=report.get("evidence", {}),
                reported_at=_parse_dt(report.get("reported_at")) or utc_now(),
                reported_by=report.get("reported_by", "push"),
            )
        failure_report = None
        if data.get("failure_report"):
            fr = data["failure_report"]
            failure_report = FailureReport(
                failure_type=FailureType(fr["failure_type"]),
                failed_step=fr.get("failed_step", ""),
                last_successful_step=fr.get("last_successful_step", ""),
                error_message=fr.get("error_message", ""),
                partial_outputs=fr.get("partial_outputs", []),
                likely_causes=fr.get("likely_causes", []),
                recommended_action=fr.get("recommended_action", ""),
                metadata=fr.get("metadata", {}),
            )
        task = Task(
            job_id=data["job_id"],
            definition=definition,
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            assigned_agent=data.get("assigned_agent"),
            attempt=data.get("attempt", 0),
            created_at=_parse_dt(data["created_at"]) or utc_now(),
            updated_at=_parse_dt(data["updated_at"]) or utc_now(),
            dispatched_at=_parse_dt(data.get("dispatched_at")),
            started_at=_parse_dt(data.get("started_at")),
            completed_at=_parse_dt(data.get("completed_at")),
            heartbeat=heartbeat,
            last_push_heartbeat=parse_heartbeat(data.get("last_push_heartbeat")),
            last_pull_heartbeat=parse_heartbeat(data.get("last_pull_heartbeat")),
            failure_report=failure_report,
            outputs=data.get("outputs", {}),
            last_activity_signature=data.get("last_activity_signature", ""),
            active_attempt_id=data.get("active_attempt_id", ""),
            lease_owner=data.get("lease_owner"),
            lease_expires_at=_parse_dt(data.get("lease_expires_at")),
            latest_agent_report=latest_agent_report,
            next_report_due_at=_parse_dt(data.get("next_report_due_at")),
            report_sequence=data.get("report_sequence", 0),
            overrun_report_count=data.get("overrun_report_count", 0),
            reporting_exhausted=data.get("reporting_exhausted", False),
        )
        return task


class EvidenceRepository:
    """Append-only evidence index for task artifacts and execution manifests."""

    def __init__(self, db: Database):
        self.db = db

    def record(
        self,
        *,
        job_id: str,
        task_id: str | None,
        kind: str,
        payload: dict,
        evidence_id: str | None = None,
    ) -> str:
        record_id = evidence_id or new_id("evidence")
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO evidence_records
                    (evidence_id, job_id, task_id, kind, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO NOTHING
                """,
                (
                    record_id,
                    job_id,
                    task_id,
                    kind,
                    dumps(payload),
                    iso(utc_now()),
                ),
            )
        return record_id

    def list_by_job(self, job_id: str) -> list[dict]:
        rows = self.db.query_all(
            "SELECT evidence_id, task_id, kind, payload, created_at "
            "FROM evidence_records WHERE job_id = ? ORDER BY created_at",
            (job_id,),
        )
        return [
            {
                "evidence_id": row["evidence_id"],
                "task_id": row["task_id"],
                "kind": row["kind"],
                "payload": loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


class InterventionRepository:
    """人工介入请求仓储；请求、Job 和任务状态可在同一事务内切换。"""

    def __init__(self, db: Database):
        self.db = db

    def save(self, request: InterventionRequest) -> None:
        now = iso(utc_now())
        with self.db.transaction() as cur:
            self._upsert(cur, request, now=now)

    def create_and_pause(
        self,
        request: InterventionRequest,
        job: ReproductionJob,
        task: Task | None,
    ) -> None:
        """原子创建请求，同时把 Job/Task 切换到等待状态。"""

        job.touch()
        if task is not None:
            task.touch()
        now = iso(utc_now())
        with self.db.transaction() as cur:
            pending = cur.execute(
                "SELECT request_id FROM intervention_requests "
                "WHERE job_id = ? AND status = ? LIMIT 1",
                (request.job_id, InterventionStatus.PENDING.value),
            ).fetchone()
            if pending is not None:
                raise ValueError(
                    f"job {request.job_id} already has pending intervention "
                    f"{pending['request_id']}"
                )
            self._upsert_job(cur, job)
            if task is not None:
                self._upsert_task(cur, task)
            self._upsert(cur, request, now=now)

    def resolve_with_state(
        self,
        request: InterventionRequest,
        job: ReproductionJob,
        task: Task | None,
    ) -> None:
        """CAS 式原子解决请求，防止同一请求被两次回答。"""

        job.touch()
        if task is not None:
            task.touch()
        now = iso(utc_now())
        with self.db.transaction() as cur:
            row = cur.execute(
                "SELECT status FROM intervention_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown intervention request: {request.request_id}")
            if row["status"] != InterventionStatus.PENDING.value:
                raise ValueError(
                    f"intervention {request.request_id} is already {row['status']}"
                )
            self._upsert(cur, request, now=now)
            self._upsert_job(cur, job)
            if task is not None:
                self._upsert_task(cur, task)

    def get(self, request_id: str) -> Optional[InterventionRequest]:
        row = self.db.query_one(
            "SELECT payload FROM intervention_requests WHERE request_id = ?",
            (request_id,),
        )
        return self._deserialize(loads(row["payload"])) if row is not None else None

    def list_by_job(
        self,
        job_id: str,
        status: InterventionStatus | None = None,
    ) -> list[InterventionRequest]:
        if status is None:
            rows = self.db.query_all(
                "SELECT payload FROM intervention_requests "
                "WHERE job_id = ? ORDER BY created_at",
                (job_id,),
            )
        else:
            rows = self.db.query_all(
                "SELECT payload FROM intervention_requests "
                "WHERE job_id = ? AND status = ? ORDER BY created_at",
                (job_id, status.value),
            )
        return [self._deserialize(loads(row["payload"])) for row in rows]

    def get_pending_for_job(self, job_id: str) -> Optional[InterventionRequest]:
        rows = self.list_by_job(job_id, InterventionStatus.PENDING)
        return rows[0] if rows else None

    @staticmethod
    def _upsert(cur, request: InterventionRequest, *, now: str | None) -> None:
        cur.execute(
            """
            INSERT INTO intervention_requests
                (request_id, job_id, task_id, kind, status, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                request.request_id,
                request.job_id,
                request.task_id,
                request.kind.value,
                request.status.value,
                dumps(request.to_dict()),
                iso(request.created_at),
                now or iso(utc_now()),
            ),
        )

    @staticmethod
    def _upsert_job(cur, job: ReproductionJob) -> None:
        cur.execute(
            """
            INSERT INTO jobs (job_id, status, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                job.job_id,
                job.status.value,
                dumps(job.to_dict()),
                iso(job.created_at),
                iso(job.updated_at),
            ),
        )

    @staticmethod
    def _upsert_task(cur, task: Task) -> None:
        cur.execute(
            """
            INSERT INTO tasks
                (task_id, job_id, status, task_type, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                task.task_id,
                task.job_id,
                task.status.value,
                task.definition.task_type,
                dumps(task.to_dict()),
                iso(task.created_at),
                iso(task.updated_at),
            ),
        )

    @staticmethod
    def _deserialize(data: dict) -> InterventionRequest:
        return InterventionRequest(
            request_id=data["request_id"],
            job_id=data["job_id"],
            task_id=data.get("task_id"),
            kind=InterventionKind(data["kind"]),
            status=InterventionStatus(data["status"]),
            question=data["question"],
            reason=data.get("reason", ""),
            input_schema=data.get("input_schema", {}),
            previous_job_status=JobStatus(data["previous_job_status"]),
            metadata=data.get("metadata", {}),
            response=data.get("response", {}),
            created_at=_parse_dt(data.get("created_at")) or utc_now(),
            expires_at=_parse_dt(data.get("expires_at")),
            responded_at=_parse_dt(data.get("responded_at")),
            responded_by=data.get("responded_by", ""),
        )


class TaskCheckpointRepository:
    """任务内安全检查点的持久化。

    Checkpoint 保存的是子 Agent 已完成的、可重复使用的逻辑步骤结果，
    而不是线程栈或进程内存。scope_hash 将它绑定到当前任务输入，避免
    某次任务定义变更后错误复用旧结果。
    """

    def __init__(self, db: Database):
        self.db = db

    def save(
        self,
        *,
        task_id: str,
        checkpoint_key: str,
        scope_hash: str,
        attempt_id: str,
        payload: dict,
    ) -> None:
        now = iso(utc_now())
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO task_checkpoints
                    (task_id, checkpoint_key, scope_hash, attempt_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id, checkpoint_key) DO UPDATE SET
                    scope_hash=excluded.scope_hash,
                    attempt_id=excluded.attempt_id,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    task_id,
                    checkpoint_key,
                    scope_hash,
                    attempt_id,
                    dumps(payload),
                    now,
                    now,
                ),
            )

    def get(
        self, *, task_id: str, checkpoint_key: str, scope_hash: str
    ) -> dict | None:
        row = self.db.query_one(
            """
            SELECT payload FROM task_checkpoints
            WHERE task_id = ? AND checkpoint_key = ? AND scope_hash = ?
            """,
            (task_id, checkpoint_key, scope_hash),
        )
        return loads(row["payload"]) if row is not None else None

    def list_by_task(self, task_id: str) -> list[dict]:
        rows = self.db.query_all(
            """
            SELECT checkpoint_key, scope_hash, attempt_id, payload, created_at, updated_at
            FROM task_checkpoints WHERE task_id = ? ORDER BY updated_at
            """,
            (task_id,),
        )
        return [
            {
                "checkpoint_key": row["checkpoint_key"],
                "scope_hash": row["scope_hash"],
                "attempt_id": row["attempt_id"],
                "payload": loads(row["payload"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]


class ToolInvocationRepository:
    """逐次工具调用审计记录；每次调用完成即落库。"""

    def __init__(self, db: Database):
        self.db = db

    def record(self, job_id: str, log: ToolInvocationLog) -> None:
        payload = log.to_dict()
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO tool_invocations
                    (invocation_id, job_id, task_id, attempt_id, sequence,
                     tool_name, succeeded, replayed, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.invocation_id or new_id("tool"),
                    job_id,
                    log.task_id,
                    log.attempt_id,
                    log.sequence,
                    log.tool_name,
                    int(log.succeeded),
                    int(log.replayed),
                    dumps(payload),
                    iso(log.timestamp),
                ),
            )

    def list_by_task(self, task_id: str) -> list[dict]:
        rows = self.db.query_all(
            "SELECT payload FROM tool_invocations WHERE task_id = ? ORDER BY created_at, sequence",
            (task_id,),
        )
        return [loads(row["payload"]) for row in rows]


class ExperimentRunRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, run: ExperimentRun) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO experiment_runs (run_id, job_id, experiment_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload
                """,
                (
                    run.run_id,
                    run.job_id,
                    run.experiment_id,
                    dumps(run.to_dict()),
                    iso(run.started_at),
                ),
            )

    def exists(self, run_id: str) -> bool:
        return self.db.query_one(
            "SELECT 1 FROM experiment_runs WHERE run_id = ?", (run_id,)
        ) is not None

    def list_by_job(self, job_id: str) -> list[ExperimentRun]:
        rows = self.db.query_all(
            "SELECT payload FROM experiment_runs WHERE job_id = ? ORDER BY created_at",
            (job_id,),
        )
        return [self._deserialize(loads(r["payload"])) for r in rows]

    def latest_full_run(self, job_id: str, experiment_id: str) -> Optional[ExperimentRun]:
        runs = [
            r
            for r in self.list_by_job(job_id)
            if r.experiment_id == experiment_id
            and r.tier == ExperimentTier.FULL_EXPERIMENT
        ]
        if not runs:
            return None
        return max(runs, key=lambda r: r.started_at)

    @staticmethod
    def _deserialize(data: dict) -> ExperimentRun:
        return ExperimentRun(
            experiment_id=data["experiment_id"],
            job_id=data["job_id"],
            tier=ExperimentTier(data["tier"]),
            run_id=data["run_id"],
            run_type=data.get("run_type", "reduced"),
            git_commit=data.get("git_commit", ""),
            container_digest=data.get("container_digest", ""),
            config_digest=data.get("config_digest", ""),
            dataset_digest=data.get("dataset_digest", ""),
            dataset_manifest=data.get("dataset_manifest", {}),
            model_identifier=data.get("model_identifier", ""),
            seed=data.get("seed"),
            hardware_identifier=data.get("hardware_identifier", ""),
            command=data.get("command", ""),
            exit_code=data.get("exit_code"),
            metrics=data.get("metrics", {}),
            log_path=data.get("log_path", ""),
            started_at=_parse_dt(data.get("started_at")) or utc_now(),
            completed_at=_parse_dt(data.get("completed_at")),
            tier_command_verified=bool(data.get("tier_command_verified", False)),
        )


class VerificationRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, record: VerificationRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO verification_records
                    (verification_id, job_id, task_id, run_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(verification_id) DO UPDATE SET payload=excluded.payload
                """,
                (
                    record.verification_id,
                    record.job_id,
                    record.task_id,
                    record.run_id,
                    dumps(record.to_dict()),
                    iso(record.completed_at),
                ),
            )

    def list_by_job(self, job_id: str) -> list[VerificationRecord]:
        rows = self.db.query_all(
            "SELECT payload FROM verification_records WHERE job_id = ? ORDER BY created_at",
            (job_id,),
        )
        return [self._deserialize(loads(row["payload"])) for row in rows]

    @staticmethod
    def _deserialize(data: dict) -> VerificationRecord:
        return VerificationRecord(
            verification_id=data["verification_id"],
            job_id=data["job_id"],
            task_id=data["task_id"],
            run_id=data.get("run_id", ""),
            expected_metric_names=data.get("expected_metric_names", []),
            observed_metric_names=data.get("observed_metric_names", []),
            missing_metrics=data.get("missing_metrics", []),
            comparisons=[
                MetricComparison(
                    metric=item["metric"],
                    paper_value=float(item["paper_value"]),
                    reproduced_value=float(item["reproduced_value"]),
                    tolerance_type=ToleranceType(item["tolerance_type"]),
                    tolerance=float(item["tolerance"]),
                    within_tolerance=bool(item["within_tolerance"]),
                )
                for item in data.get("comparisons", [])
            ],
            run_actually_executed=bool(data.get("run_actually_executed", False)),
            provenance_verified=bool(data.get("provenance_verified", False)),
            anti_cheat_passed=bool(data.get("anti_cheat_passed", False)),
            is_fully_traceable=bool(data.get("is_fully_traceable", False)),
            mock=bool(data.get("mock", False)),
            verification_valid=bool(data.get("verification_valid", False)),
            completed_at=_parse_dt(data.get("completed_at")) or utc_now(),
        )
class ReflectionRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, report: ReflectionReport) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO reflection_reports (reflection_id, job_id, round, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(reflection_id) DO UPDATE SET payload=excluded.payload
                """,
                (
                    report.reflection_id,
                    report.job_id,
                    report.round,
                    dumps(report.to_dict()),
                    iso(report.created_at),
                ),
            )

    def list_by_job(self, job_id: str) -> list[ReflectionReport]:
        rows = self.db.query_all(
            "SELECT payload FROM reflection_reports WHERE job_id = ? ORDER BY created_at",
            (job_id,),
        )
        return [self._deserialize(loads(r["payload"])) for r in rows]

    @staticmethod
    def _deserialize(data: dict) -> ReflectionReport:
        comparisons = [
            MetricComparison(
                metric=item["metric"],
                paper_value=float(item["paper_value"]),
                reproduced_value=float(item["reproduced_value"]),
                tolerance_type=ToleranceType(item["tolerance_type"]),
                tolerance=float(item["tolerance"]),
                within_tolerance=bool(item["within_tolerance"]),
            )
            for item in data.get("trigger_metrics", [])
        ]
        hypotheses = [
            ReflectionHypothesis(
                category=item.get("category", ""),
                description=item.get("description", ""),
                priority=int(item.get("priority", 0)),
                confidence=float(item.get("confidence", 0.0)),
                required_checks=item.get("required_checks", []),
                hypothesis_id=item.get("id", ""),
            )
            for item in data.get("hypotheses", [])
        ]
        findings = [
            AuditFinding(
                audit_task_id=item.get("audit_task_id", ""),
                check_dimension=item.get("check_dimension", ""),
                result=AuditResultType(item["result"]),
                detail=item.get("detail", ""),
                evidence_refs=item.get("evidence_refs", []),
            )
            for item in data.get("audit_findings", [])
        ]
        return ReflectionReport(
            job_id=data["job_id"],
            round=int(data.get("round", 0)),
            trigger_metrics=comparisons,
            run_id=data.get("run_id", ""),
            likely_source=data.get("likely_source", "unknown"),
            reflection_id=data["reflection_id"],
            hypotheses=hypotheses,
            audit_context=dict(data.get("audit_context", {}) or {}),
            audit_task_ids=data.get("audit_task_ids", []),
            audit_findings=findings,
            audit_result=AuditResultType(data["audit_result"])
            if data.get("audit_result")
            else None,
            confirmed_issue=data.get("confirmed_issue", ""),
            recommended_rerun_scope=RerunScope(data["recommended_rerun_scope"])
            if data.get("recommended_rerun_scope")
            else None,
            repair_task_ids=data.get("repair_task_ids", []),
            rerun_triggered=bool(data.get("rerun_triggered", False)),
            created_at=_parse_dt(data.get("created_at")) or utc_now(),
        )


class DynamicToolRepository:
    """Workspace-scoped persistence for generated tool candidates.

    Built-in tools never appear in these tables.  Consequently lifecycle
    maintenance can only mutate tools created by the self-growth subsystem and
    cannot accidentally age or delete the fixed harness toolset.
    """

    def __init__(self, db: Database):
        self.db = db

    def save(self, record: dict[str, Any]) -> None:
        now = str(record.get("updated_at") or iso(utc_now()))
        created_at = str(record.get("created_at") or now)
        payload = dict(record)
        payload["created_at"] = created_at
        payload["updated_at"] = now
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO dynamic_tools
                    (tool_id, tool_name, status, life, max_life, support_count,
                     failure_count, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                    tool_name=excluded.tool_name,
                    status=excluded.status,
                    life=excluded.life,
                    max_life=excluded.max_life,
                    support_count=excluded.support_count,
                    failure_count=excluded.failure_count,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    str(payload["tool_id"]),
                    str(payload["tool_name"]),
                    str(payload["status"]),
                    int(payload["life"]),
                    int(payload["max_life"]),
                    int(payload.get("support_count", 0)),
                    int(payload.get("failure_count", 0)),
                    dumps(payload),
                    created_at,
                    now,
                ),
            )

    def get(self, tool_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT payload FROM dynamic_tools WHERE tool_id = ?", (tool_id,)
        )
        return loads(row["payload"]) if row is not None else None

    def get_by_name(self, tool_name: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT payload FROM dynamic_tools WHERE tool_name = ?", (tool_name,)
        )
        return loads(row["payload"]) if row is not None else None

    def list_all(self) -> list[dict[str, Any]]:
        rows = self.db.query_all(
            "SELECT payload FROM dynamic_tools ORDER BY created_at, tool_id"
        )
        return [loads(row["payload"]) for row in rows]

    def add_evidence(self, evidence: dict[str, Any]) -> bool:
        """Persist one independent occurrence; return False for a replay."""

        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO dynamic_tool_evidence
                    (evidence_id, tool_id, job_id, task_id, attempt_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id, task_id) DO NOTHING
                """,
                (
                    str(evidence["evidence_id"]),
                    str(evidence["tool_id"]),
                    str(evidence["job_id"]),
                    str(evidence["task_id"]),
                    str(evidence["attempt_id"]),
                    dumps(evidence),
                    str(evidence.get("created_at") or iso(utc_now())),
                ),
            )
            return cur.rowcount == 1

    def save_with_evidence(
        self, record: dict[str, Any], evidence: dict[str, Any]
    ) -> bool:
        """Atomically persist a valid occurrence and its support-count update."""

        now = str(record.get("updated_at") or iso(utc_now()))
        created_at = str(record.get("created_at") or now)
        payload = dict(record)
        payload["created_at"] = created_at
        payload["updated_at"] = now
        with self.db.transaction() as cur:
            duplicate = cur.execute(
                "SELECT 1 FROM dynamic_tool_evidence WHERE tool_id = ? AND task_id = ?",
                (str(evidence["tool_id"]), str(evidence["task_id"])),
            ).fetchone()
            if duplicate is not None:
                return False
            # Parent first so the evidence foreign key is always satisfied.
            cur.execute(
                """
                INSERT INTO dynamic_tools
                    (tool_id, tool_name, status, life, max_life, support_count,
                     failure_count, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                    tool_name=excluded.tool_name,
                    status=excluded.status,
                    life=excluded.life,
                    max_life=excluded.max_life,
                    support_count=excluded.support_count,
                    failure_count=excluded.failure_count,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    str(payload["tool_id"]),
                    str(payload["tool_name"]),
                    str(payload["status"]),
                    int(payload["life"]),
                    int(payload["max_life"]),
                    int(payload.get("support_count", 0)),
                    int(payload.get("failure_count", 0)),
                    dumps(payload),
                    created_at,
                    now,
                ),
            )
            cur.execute(
                """
                INSERT INTO dynamic_tool_evidence
                    (evidence_id, tool_id, job_id, task_id, attempt_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id, task_id) DO NOTHING
                """,
                (
                    str(evidence["evidence_id"]),
                    str(evidence["tool_id"]),
                    str(evidence["job_id"]),
                    str(evidence["task_id"]),
                    str(evidence["attempt_id"]),
                    dumps(evidence),
                    str(evidence.get("created_at") or iso(utc_now())),
                ),
            )
            if cur.rowcount == 1:
                return True
            return False

    def list_evidence(self, tool_id: str) -> list[dict[str, Any]]:
        rows = self.db.query_all(
            "SELECT payload FROM dynamic_tool_evidence "
            "WHERE tool_id = ? ORDER BY created_at, evidence_id",
            (tool_id,),
        )
        return [loads(row["payload"]) for row in rows]
