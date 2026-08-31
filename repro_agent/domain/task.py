"""任务定义协议与任务实体（设计文档 §7、§13、§18.2）。

任务是本系统调度的最小单元。字段设计严格对齐设计文档 §7 的任务定义协议
JSON 示例，并补充了调度器运行所需的运行时字段（心跳、尝试次数等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.domain.enums import FailureType, TaskStatus


@dataclass
class TaskDefinition:
    """任务的静态定义部分（创建后不可变，对应设计文档 §7 JSON 协议）。"""

    objective: str
    task_type: str
    dependencies: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    expected_duration_seconds: int = 300
    soft_timeout_seconds: int = 600
    hard_timeout_seconds: int = 1200
    heartbeat_interval_seconds: int = 30
    # 兼容旧任务快照；动态报备机制不再使用固定的存活宽限期做业务裁决。
    # 底层活动信号与上层报备租约已经解耦，见 AgentReport。
    liveness_grace_seconds: int = 120
    # 正常进度报告不消耗额度；只有预计完成时间到达后仍未完成、主 Agent
    # 主动查询并批准继续时才累计一次。达到上限后终止，防止无限等待。
    max_overrun_reports: int = 3
    failure_report_required: bool = True
    priority: int = 0
    max_attempts: int = 3
    parent_task_id: Optional[str] = None


class AgentReportType(str, Enum):
    """业务报备类型；底层活动使用独立的 ``ActivitySignal``。"""

    STARTED = "started"
    PROGRESS = "progress"
    EXTENSION = "extension"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentReport:
    """子/主 Agent 之间的结构化报备合同。

    ``eta_seconds`` 是预计剩余时间，``next_report_after_seconds`` 是下次
    必须报备的间隔。当前实现默认二者相同，但分开持久化，便于未来让
    长实验在预计完成前主动发送阶段报告。``EXTENSION`` 只会在原截止
    时间到达、主 Agent 主动 pull 且确认执行仍存活后产生。
    """

    attempt_id: str
    report_type: AgentReportType
    progress: float = 0.0
    current_step: str = ""
    eta_seconds: Optional[float] = None
    next_report_after_seconds: Optional[float] = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    reported_at: datetime = field(default_factory=utc_now)
    reported_by: str = "push"
    sequence: int = 0
    report_id: str = field(default_factory=lambda: new_id("report"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "attempt_id": self.attempt_id,
            "sequence": self.sequence,
            "report_type": self.report_type.value,
            "progress": self.progress,
            "current_step": self.current_step,
            "eta_seconds": self.eta_seconds,
            "next_report_after_seconds": self.next_report_after_seconds,
            "reason": self.reason,
            "evidence": self.evidence,
            "reported_at": iso(self.reported_at),
            "reported_by": self.reported_by,
        }


@dataclass
class ActivitySignal:
    """底层活动快照，只用于诊断线程/进程是否存在。

    活动信号不会刷新 ``Task.next_report_due_at``，也不会清零或增加
    ``overrun_report_count``。这样 Docker 日志、工具开始/结束等低层
    信号不会冒充业务进度，更不会让长任务无限续命。
    """

    progress: float = 0.0
    current_step: str = ""
    last_completed_step: str = ""
    last_log_position: int = 0
    updated_at: datetime = field(default_factory=utc_now)
    eta_seconds: Optional[float] = None
    reported_by: str = "push"

    def to_dict(self) -> dict[str, Any]:
        return {
            "progress": self.progress,
            "current_step": self.current_step,
            "last_completed_step": self.last_completed_step,
            "last_log_position": self.last_log_position,
            "updated_at": iso(self.updated_at),
            "eta_seconds": self.eta_seconds,
            "reported_by": self.reported_by,
        }


# 公共兼容名：旧调用方仍可读取 Heartbeat，但其语义已收窄为活动快照。
Heartbeat = ActivitySignal


@dataclass
class FailureReport:
    """失败报告（设计文档 §14）。"""

    failure_type: FailureType
    failed_step: str
    last_successful_step: str = ""
    error_message: str = ""
    partial_outputs: list[str] = field(default_factory=list)
    likely_causes: list[str] = field(default_factory=list)
    recommended_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "failed_step": self.failed_step,
            "last_successful_step": self.last_successful_step,
            "error_message": self.error_message,
            "partial_outputs": self.partial_outputs,
            "likely_causes": self.likely_causes,
            "recommended_action": self.recommended_action,
            "metadata": self.metadata,
        }


@dataclass
class Task:
    """任务运行时实体（设计文档 §18.2 ``Task``）。"""

    job_id: str
    definition: TaskDefinition
    task_id: str = field(default_factory=lambda: new_id("task"))
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    attempt: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    dispatched_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    heartbeat: Optional[Heartbeat] = None
    # Keep active push and pull observations separate. ``heartbeat`` remains
    # the backwards-compatible latest business heartbeat view.
    last_push_heartbeat: Optional[Heartbeat] = None
    last_pull_heartbeat: Optional[Heartbeat] = None
    failure_report: Optional[FailureReport] = None
    outputs: dict[str, Any] = field(default_factory=dict)
    # 心跳/超时判定的"上次已知活跃指标"（日志长度等），
    # 用于 §13.3 软超时判定中的"日志是否增长"检查。
    last_activity_signature: str = ""
    # Every retry gets a new attempt identity. Results from any older
    # identity are audit-only and cannot mutate this task.
    active_attempt_id: str = ""
    # 分布式/多 Worker 场景下的执行租约持有者（参考 DeerFlow 的
    # Lease+心跳 Worker 归属模型，见 scheduler/lease.py 的说明）。
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    # 动态报备租约。最新状态保存在 Task payload，完整流水复用现有
    # task_events，避免新增一套重复的报告表/状态机。
    latest_agent_report: Optional[AgentReport] = None
    next_report_due_at: Optional[datetime] = None
    report_sequence: int = 0
    overrun_report_count: int = 0
    reporting_exhausted: bool = False

    # ---- 派生属性 ----

    @property
    def objective(self) -> str:
        return self.definition.objective

    @property
    def dependencies(self) -> list[str]:
        return self.definition.dependencies

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_summary_dict(self) -> dict[str, Any]:
        """用于 DAG 摘要 / 上下文注入的精简视图（避免把全部字段塞进上下文）。"""

        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "task_type": self.definition.task_type,
            "status": self.status.value,
            "attempt": self.attempt,
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent,
            "next_report_due_at": iso(self.next_report_due_at),
            "overrun_report_count": self.overrun_report_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "parent_task_id": self.definition.parent_task_id,
            "objective": self.objective,
            "task_type": self.definition.task_type,
            "dependencies": self.dependencies,
            "inputs": self.definition.inputs,
            "allowed_tools": self.definition.allowed_tools,
            "forbidden_actions": self.definition.forbidden_actions,
            "expected_outputs": self.definition.expected_outputs,
            "completion_criteria": self.definition.completion_criteria,
            "expected_duration_seconds": self.definition.expected_duration_seconds,
            "soft_timeout_seconds": self.definition.soft_timeout_seconds,
            "hard_timeout_seconds": self.definition.hard_timeout_seconds,
            "heartbeat_interval_seconds": self.definition.heartbeat_interval_seconds,
            "liveness_grace_seconds": self.definition.liveness_grace_seconds,
            "max_overrun_reports": self.definition.max_overrun_reports,
            "failure_report_required": self.definition.failure_report_required,
            "priority": self.definition.priority,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "attempt": self.attempt,
            "max_attempts": self.definition.max_attempts,
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "dispatched_at": iso(self.dispatched_at),
            "started_at": iso(self.started_at),
            "completed_at": iso(self.completed_at),
            "heartbeat": self.heartbeat.to_dict() if self.heartbeat else None,
            "last_push_heartbeat": self.last_push_heartbeat.to_dict()
            if self.last_push_heartbeat
            else None,
            "last_pull_heartbeat": self.last_pull_heartbeat.to_dict()
            if self.last_pull_heartbeat
            else None,
            "failure_report": self.failure_report.to_dict()
            if self.failure_report
            else None,
            "outputs": self.outputs,
            "last_activity_signature": self.last_activity_signature,
            "active_attempt_id": self.active_attempt_id,
            "lease_owner": self.lease_owner,
            "lease_expires_at": iso(self.lease_expires_at),
            "latest_agent_report": self.latest_agent_report.to_dict()
            if self.latest_agent_report
            else None,
            "next_report_due_at": iso(self.next_report_due_at),
            "report_sequence": self.report_sequence,
            "overrun_report_count": self.overrun_report_count,
            "reporting_exhausted": self.reporting_exhausted,
        }
