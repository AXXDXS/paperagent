"""任务定义协议与任务实体（设计文档 §7、§13、§18.2）。

任务是本系统调度的最小单元。字段设计严格对齐设计文档 §7 的任务定义协议
JSON 示例，并补充了调度器运行所需的运行时字段（心跳、尝试次数等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    # 子智能体超过多久没有主动 push 心跳，主智能体判定为"未按要求汇报"，
    # 从而触发强制状态查询（pull）。用户需求原文是"约 2 分钟"，这里
    # 设为独立可配置字段而不是硬编码，默认 120 秒对齐需求原文，
    # 同时允许单个任务按需调整（例如已知会长时间静默的重型任务）。
    liveness_grace_seconds: int = 120
    failure_report_required: bool = True
    priority: int = 0
    max_attempts: int = 3
    parent_task_id: Optional[str] = None


@dataclass
class Heartbeat:
    """子智能体心跳（设计文档 §13.2）。

    调度器依据心跳判断任务是否"卡死"：如果超过
    ``heartbeat_interval_seconds`` 的若干倍仍未收到心跳，
    但尚未到硬超时，则标记为疑似卡死并触发状态探测。

    ``reported_by`` 字段区分心跳的来源，用于满足"必须保证一定是
    子智能体先主动汇报"这一约束：
        - ``"push"``：子智能体运行线程主动调用
          ``BaseSubAgent.report_progress()`` 写入的心跳，是正常、
          被信任的心跳来源；
        - ``"pull"``：主智能体在子智能体超过
          ``heartbeat_interval_seconds * 4``（约 2 分钟量级，具体见
          ``scheduler.subagent_liveness.LivenessPolicy``）仍未收到
          ``push`` 心跳时，通过 ``get_subagent_status`` 主动查询得到
          的状态快照。``pull`` 心跳只能证明"进程/线程还活着"，不能
          替代子智能体自己上报的真实业务进度，因此
          ``LivenessPolicy`` 在判定"是否已确认死亡"时只信任
          ``push`` 心跳的新鲜度，``pull`` 心跳仅用于"这一次强制查询
          本身是否成功"的记录，不会重置卡死计时。
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
        }
