"""Human-in-the-loop 介入请求领域模型。

介入请求是独立于日志事件的持久化实体：事件只回答“发生过什么”，
而本实体还负责回答“系统正在等谁、等什么、输入格式是什么、是否已经
回答”。这样进程重启后仍可继续同一次人工协作，而不是依赖终端里的文字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.domain.enums import InterventionKind, InterventionStatus, JobStatus


@dataclass
class InterventionRequest:
    """一次结构化、可审计、可恢复的人工介入请求。"""

    job_id: str
    kind: InterventionKind
    question: str
    reason: str
    input_schema: dict[str, Any]
    previous_job_status: JobStatus
    task_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: new_id("intervention"))
    status: InterventionStatus = InterventionStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    expires_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    responded_by: str = ""

    @property
    def is_pending(self) -> bool:
        return self.status == InterventionStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "question": self.question,
            "reason": self.reason,
            "input_schema": self.input_schema,
            "previous_job_status": self.previous_job_status.value,
            "metadata": self.metadata,
            "response": self.response,
            "created_at": iso(self.created_at),
            "expires_at": iso(self.expires_at),
            "responded_at": iso(self.responded_at),
            "responded_by": self.responded_by,
        }
