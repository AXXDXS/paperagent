"""SQLite 持久化层：任务状态的唯一事实来源（设计文档 §3 原则 15-16）。

设计原则回顾：
    原则 15：任务实时状态保存在数据库；
    原则 16：数据库是任务状态的唯一事实来源；
    原则 17：Markdown 记忆不能替代任务数据库。

工程取舍与复用说明：
    - 采用 SQLite 而非引入 Postgres/Redis 等外部服务，
      理由：设计文档的 MVP 范围（§22）强调"单机可运行"，
      且系统的强一致性需求（任务状态唯一来源）用 SQLite 的
      WAL 模式 + 显式事务即可满足，不需要引入分布式组件的复杂度。
    - PRAGMA 配置参考了 Pi 项目 ``pi-storage-sqlite-node`` 的
      SQLite 使用方式（journal_mode=WAL 提升并发读写性能、
      busy_timeout 应对并发锁竞争），详见
      ``doc/pi-项目分析.md`` 第 8.3 节。
    - 表结构在 Task/Job 之外还包含 ``task_leases``，用于落地
      DeerFlow 的 Lease+心跳 Worker 归属思想（见
      ``scheduler/lease.py``），使得"心跳"和"超时"判断都能在
      进程重启后从数据库恢复，而不是依赖内存状态。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 6

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    task_type TEXT NOT NULL,
    creation_key TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs (job_id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_job_status ON tasks (job_id, status);

CREATE TABLE IF NOT EXISTS task_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    task_id TEXT,
    event_type TEXT NOT NULL,
    event_key TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON task_events (job_id, event_id);

CREATE TABLE IF NOT EXISTS task_attempts (
    attempt_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
);
CREATE INDEX IF NOT EXISTS idx_task_attempts_task
    ON task_attempts (task_id, attempt_number);

CREATE TABLE IF NOT EXISTS intervention_requests (
    request_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs (job_id),
    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
);
CREATE INDEX IF NOT EXISTS idx_interventions_job_status
    ON intervention_requests (job_id, status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_pending_intervention_per_job
    ON intervention_requests (job_id) WHERE status = 'PENDING';

CREATE TABLE IF NOT EXISTS task_checkpoints (
    task_id TEXT NOT NULL,
    checkpoint_key TEXT NOT NULL,
    scope_hash TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (task_id, checkpoint_key),
    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON task_checkpoints (task_id, updated_at);

CREATE TABLE IF NOT EXISTS tool_invocations (
    invocation_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    succeeded INTEGER NOT NULL,
    replayed INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
);
CREATE INDEX IF NOT EXISTS idx_tool_invocations_task ON tool_invocations (task_id, sequence);
CREATE INDEX IF NOT EXISTS idx_tool_invocations_job ON tool_invocations (job_id, created_at);

CREATE TABLE IF NOT EXISTS task_leases (
    task_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_job ON experiment_runs (job_id);

CREATE TABLE IF NOT EXISTS reflection_reports (
    reflection_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_job ON context_snapshots (job_id, created_at);

CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_job ON evidence_records (job_id);

CREATE TABLE IF NOT EXISTS verification_records (
    verification_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verifications_job ON verification_records (job_id, created_at);

CREATE TABLE IF NOT EXISTS dynamic_tools (
    tool_id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    life INTEGER NOT NULL,
    max_life INTEGER NOT NULL,
    support_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_status
    ON dynamic_tools (status, updated_at);

CREATE TABLE IF NOT EXISTS dynamic_tool_evidence (
    evidence_id TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (tool_id) REFERENCES dynamic_tools (tool_id),
    UNIQUE (tool_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_dynamic_tool_evidence_tool
    ON dynamic_tool_evidence (tool_id, created_at);
"""


class Database:
    """线程安全的 SQLite 封装。

    使用 ``threading.RLock`` 而非依赖 SQLite 自身的锁，是因为调度器
    在同一进程内可能有多个线程（主循环线程 + 心跳检查线程）同时读写，
    显式锁能让"读-改-写"的复合操作保持原子性，避免只靠数据库事务
    仍然出现的竞态（例如"判断 READY 后再写入 DISPATCHED"这类跨语句的
    复合逻辑）。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        try:
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _configure_pragmas(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    def _init_schema(self) -> None:
        with self.transaction() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = cur.execute(
                "SELECT value FROM schema_meta WHERE key = ?", ("schema_version",)
            ).fetchone()
            stored_version = int(row["value"]) if row is not None else 0
            if stored_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {stored_version} is newer than supported "
                    f"schema {SCHEMA_VERSION}"
                )
            cur.executescript(_SCHEMA_SQL)
            self._ensure_column(cur, "tasks", "creation_key", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(cur, "task_events", "event_key", "TEXT NOT NULL DEFAULT ''")
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_creation_key "
                "ON tasks (job_id, creation_key) WHERE creation_key <> ''"
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_event_key "
                "ON task_events (job_id, event_key) WHERE event_key <> ''"
            )
            cur.execute(
                """
                INSERT INTO schema_meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                ("schema_version", str(SCHEMA_VERSION)),
            )

    @staticmethod
    def _ensure_column(
        cur: sqlite3.Cursor, table: str, column: str, declaration: str
    ) -> None:
        columns = {row["name"] for row in cur.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """显式事务上下文管理器：成功提交、异常回滚。"""

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            row = cur.fetchone()
            cur.close()
            return row

    def query_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            return rows

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def loads(text: str) -> Any:
    return json.loads(text)
