"""反思与审计数据结构（设计文档 §11、§18.4）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.domain.enums import AuditResultType, RerunScope
from repro_agent.domain.experiment import MetricComparison


@dataclass
class ReflectionHypothesis:
    """反思假设（§11.4 ``hypotheses``）。"""

    category: str
    description: str
    priority: int
    confidence: float
    required_checks: list[str] = field(default_factory=list)
    hypothesis_id: str = field(default_factory=lambda: new_id("hyp"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.hypothesis_id,
            "category": self.category,
            "description": self.description,
            "priority": self.priority,
            "confidence": self.confidence,
            "required_checks": self.required_checks,
        }


@dataclass
class AuditFinding:
    """单个审计任务的结论。"""

    audit_task_id: str
    check_dimension: str  # 对应 §11.3 的 A-J 检查维度
    result: AuditResultType
    detail: str
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_task_id": self.audit_task_id,
            "check_dimension": self.check_dimension,
            "result": self.result.value,
            "detail": self.detail,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class ReflectionReport:
    """反思报告（§11.4、§18.4）。"""

    job_id: str
    round: int
    trigger_metrics: list[MetricComparison] = field(default_factory=list)
    run_id: str = ""
    likely_source: str = "unknown"
    reflection_id: str = field(default_factory=lambda: new_id("reflection"))
    hypotheses: list[ReflectionHypothesis] = field(default_factory=list)
    # Durable source/task bindings needed to recreate evidence-bearing audit
    # tasks after a controller restart.
    audit_context: dict[str, Any] = field(default_factory=dict)
    audit_task_ids: list[str] = field(default_factory=list)
    audit_findings: list[AuditFinding] = field(default_factory=list)
    audit_result: Optional[AuditResultType] = None
    confirmed_issue: str = ""
    recommended_rerun_scope: Optional[RerunScope] = None
    # 主循环编排状态（§19 主循环消费本报告推进 repair -> rerun 时使用，
    # 不属于"审计结论"本身，只是执行进度追踪，因此单独放在结论字段
    # 之后，避免和 §11.4 定义的核心字段混在一起）。
    repair_task_ids: list[str] = field(default_factory=list)
    rerun_triggered: bool = False
    created_at: datetime = field(default_factory=utc_now)

    @property
    def issue_found(self) -> bool:
        if self.audit_result is None:
            return False
        return self.audit_result not in {
            AuditResultType.NO_OBVIOUS_ERROR_FOUND,
            AuditResultType.RANDOMNESS_LIKELY,
            AuditResultType.UNDISCLOSED_DETAIL_LIKELY,
        }

    def sorted_hypotheses(self) -> list[ReflectionHypothesis]:
        return sorted(self.hypotheses, key=lambda h: h.priority, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "round": self.round,
            "likely_source": self.likely_source,
            "trigger_metrics": [m.to_dict() for m in self.trigger_metrics],
            "hypotheses": [h.to_dict() for h in self.sorted_hypotheses()],
            "audit_context": self.audit_context,
            "audit_task_ids": self.audit_task_ids,
            "audit_findings": [f.to_dict() for f in self.audit_findings],
            "audit_result": self.audit_result.value if self.audit_result else None,
            "confirmed_issue": self.confirmed_issue,
            "recommended_rerun_scope": self.recommended_rerun_scope.value
            if self.recommended_rerun_scope
            else None,
            "repair_task_ids": self.repair_task_ids,
            "rerun_triggered": self.rerun_triggered,
            "created_at": iso(self.created_at),
        }
