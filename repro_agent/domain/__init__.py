"""ReproAgent 核心数据模型（domain 层）。

domain 层不依赖任何存储/网络/LLM 实现，只描述"系统里存在哪些概念、
概念之间如何关联、状态如何流转"，对应设计文档的：
    §2 复现状态、§7 任务定义协议、§9 子智能体输出结构、
    §11 反思与审计、§13 任务状态机、§17 Job 状态机、§18 关键数据结构。
"""

from repro_agent.domain.dag import CycleDetectedError, TaskDAG
from repro_agent.domain.enums import (
    AuditResultType,
    ExperimentTier,
    FailureDecision,
    FailureType,
    FieldProvenance,
    InterventionKind,
    InterventionStatus,
    JobStatus,
    ReproductionStatus,
    RerunScope,
    ResourceStatus,
    TaskStatus,
    ToleranceType,
)
from repro_agent.domain.experiment import (
    ExpectedResult,
    ExperimentRun,
    ExperimentSpec,
    MetricComparison,
    ProvenancedField,
    compare_metrics,
)
from repro_agent.domain.job import JobBudget, JobInputs, ReproductionJob
from repro_agent.domain.intervention import InterventionRequest
from repro_agent.domain.reflection import (
    AuditFinding,
    ReflectionHypothesis,
    ReflectionReport,
)
from repro_agent.domain.task import FailureReport, Heartbeat, Task, TaskDefinition

__all__ = [
    "AuditFinding",
    "AuditResultType",
    "CycleDetectedError",
    "ExpectedResult",
    "ExperimentRun",
    "ExperimentSpec",
    "ExperimentTier",
    "FailureDecision",
    "FailureReport",
    "FailureType",
    "FieldProvenance",
    "Heartbeat",
    "InterventionKind",
    "InterventionRequest",
    "InterventionStatus",
    "JobBudget",
    "JobInputs",
    "JobStatus",
    "MetricComparison",
    "ProvenancedField",
    "ReflectionHypothesis",
    "ReflectionReport",
    "ReproductionJob",
    "ReproductionStatus",
    "RerunScope",
    "ResourceStatus",
    "Task",
    "TaskDAG",
    "TaskDefinition",
    "TaskStatus",
    "ToleranceType",
    "compare_metrics",
]
