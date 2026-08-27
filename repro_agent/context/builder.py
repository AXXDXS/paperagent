"""主智能体决策上下文构造器（设计文档 §16 九段流水线）。

流水线（§16 原文）：
    读取 Job 状态 → 读取 DAG 摘要 → 确定当前决策 → 读取 L0 记忆索引
    → 加载相关 L1 → 必要时加载 L2/L3 → 加载最近事件 → 加载未解决问题
    → 加载当前预算 → 压缩到上下文限制

本模块只负责"组装 + 压缩"，不负责"决策"本身——决策逻辑属于
``orchestrator``，这里产出的是喂给决策逻辑（或直接喂给 LLM Prompt）
的结构化上下文 Envelope，遵循单一职责、便于独立测试压缩效果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from repro_agent.context.budget import (
    CompressionResult,
    ContextSegmentKind,
    ContextSource,
    ContextSegment,
    SegmentPriority,
    compress_segments,
)
from repro_agent.domain.dag import TaskDAG
from repro_agent.domain.job import ReproductionJob
from repro_agent.memory.manager import MainAgentCapability, MemoryManager

_MUST_KEEP = SegmentPriority.MUST_KEEP
_COMPRESSIBLE_LOW = SegmentPriority.COMPRESSIBLE_LOW
_COMPRESSIBLE_MEDIUM = SegmentPriority.COMPRESSIBLE_MEDIUM
_COMPRESSIBLE_HIGH = SegmentPriority.COMPRESSIBLE_HIGH


@dataclass
class UnresolvedIssue:
    """未解决问题条目（参数冲突/数据缺失/反思假设等，§16.1 必须保留）。"""

    kind: str  # "data_missing" | "param_conflict" | "reflection_hypothesis" | ...
    description: str
    related_task_id: str = ""


@dataclass
class ContextEnvelope:
    """发送给主智能体的版本化结构化上下文协议。"""

    segments: list[ContextSegment]
    compression: CompressionResult
    schema_version: str = "1.0"
    context_type: str = "main_agent_decision"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_type": self.context_type,
            "segments": [segment.to_dict() for segment in self.segments],
            "compression": self.compression.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class DecisionContext:
    """构造好的、已压缩的决策上下文。"""

    envelope: ContextEnvelope
    compression: CompressionResult
    current_decision: str

    @property
    def text(self) -> str:
        """模型消息仍以字符串传输，但内容是可验证的 JSON Envelope。"""

        return self.envelope.to_json()


class ContextBuilder:
    """按 §16 九段流水线组装并压缩主智能体的决策上下文。"""

    def __init__(self, memory_manager: MemoryManager, capability: MainAgentCapability):
        self.memory_manager = memory_manager
        self.capability = capability

    def build(
        self,
        *,
        job: ReproductionJob,
        dag: TaskDAG,
        current_decision: str,
        recent_events: list[dict[str, Any]],
        unresolved_issues: list[UnresolvedIssue],
        relevant_topics: list[tuple[str, str]] | None = None,  # (section, topic)
        expand_full_topics: list[tuple[str, str]] | None = None,
        max_tokens: int = 8000,
    ) -> DecisionContext:
        segments: list[ContextSegment] = []

        # 1. Job 状态（必须保留）
        segments.append(
            ContextSegment(
                name="job_status",
                kind=ContextSegmentKind.JOB_STATE,
                source=ContextSource.JOB,
                content=self._build_job_status(job),
                priority=_MUST_KEEP,
            )
        )

        # 2. DAG 摘要（必须保留：活跃任务、任务依赖属于此段）
        segments.append(
            ContextSegment(
                name="dag_summary",
                kind=ContextSegmentKind.TASK_GRAPH,
                source=ContextSource.SCHEDULER,
                content=self._build_dag_summary(dag),
                priority=_MUST_KEEP,
            )
        )

        # 3. 当前决策（必须保留）
        segments.append(
            ContextSegment(
                name="current_decision",
                kind=ContextSegmentKind.CURRENT_DECISION,
                source=ContextSource.ORCHESTRATOR,
                content={"description": current_decision},
                priority=_MUST_KEEP,
            )
        )

        # 4. L0 记忆索引（可压缩：属于"与当前决策无关的信息"的候选，
        #    但索引本身很小，通常不会被真正丢弃）
        l0_index = self.memory_manager.read_l0_index(self.capability)
        segments.append(
            ContextSegment(
                name="memory_l0_index",
                kind=ContextSegmentKind.MEMORY_INDEX,
                source=ContextSource.MEMORY,
                content=self._build_l0_index(l0_index),
                priority=_COMPRESSIBLE_HIGH,
            )
        )

        # 5. 相关 L1 摘要（中等优先级：已被验证过的旧结论用得少时可压缩）
        for section, topic in relevant_topics or []:
            summary = self.memory_manager.read_l1_summary(self.capability, section, topic)
            if summary:
                segments.append(
                    ContextSegment(
                        name=f"memory_l1::{section}.{topic}",
                        kind=ContextSegmentKind.MEMORY_ENTRY,
                        source=ContextSource.MEMORY,
                        content={
                            "level": "L1",
                            "section": section,
                            "topic": topic,
                            "summary": summary,
                        },
                        priority=_COMPRESSIBLE_MEDIUM,
                    )
                )

        # 6. 必要时加载 L2/L3（仅验证/冲突处理/反思审计场景，调用方显式传入）
        for section, topic in expand_full_topics or []:
            full = self.memory_manager.read_full_entry(self.capability, section, topic)
            if full:
                segments.append(
                    ContextSegment(
                        name=f"memory_full::{section}.{topic}",
                        kind=ContextSegmentKind.MEMORY_ENTRY,
                        source=ContextSource.MEMORY,
                        content={
                            "level": "full",
                            "section": section,
                            "topic": topic,
                            "entry": full,
                        },
                        priority=_COMPRESSIBLE_HIGH,
                    )
                )

        # 7. 最近事件（低优先级：重复事件/已解决问题优先压缩）
        segments.append(
            ContextSegment(
                name="recent_events",
                kind=ContextSegmentKind.EVENT_STREAM,
                source=ContextSource.EVENT_STORE,
                content={"events": recent_events[-30:]},
                priority=_COMPRESSIBLE_LOW,
                metadata={"max_events": 30},
            )
        )

        # 8. 未解决问题（必须保留）
        segments.append(
            ContextSegment(
                name="unresolved_issues",
                kind=ContextSegmentKind.ISSUE_LIST,
                source=ContextSource.ORCHESTRATOR,
                content=self._build_unresolved_issues(unresolved_issues),
                priority=_MUST_KEEP,
            )
        )

        # 9. 当前预算（必须保留：系统约束的一部分）
        segments.append(
            ContextSegment(
                name="budget",
                kind=ContextSegmentKind.BUDGET_STATE,
                source=ContextSource.JOB,
                content=self._build_budget(job),
                priority=_MUST_KEEP,
            )
        )

        compression = compress_segments(segments, max_tokens=max_tokens)
        envelope = ContextEnvelope(
            segments=compression.kept_segments,
            compression=compression,
        )
        return DecisionContext(
            envelope=envelope,
            compression=compression,
            current_decision=current_decision,
        )

    # ---- 各段结构化 ----

    @staticmethod
    def _build_job_status(job: ReproductionJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "reflection_round": {
                "current": job.reflection_round,
                "maximum": job.budget.max_reflection_rounds,
            },
            "full_experiment_reruns": {
                "current": job.full_experiment_rerun_count,
                "maximum": job.budget.max_full_experiment_reruns,
            },
            "target_experiments": list(job.inputs.target_experiments),
        }

    @staticmethod
    def _build_dag_summary(dag: TaskDAG) -> dict[str, Any]:
        summary = dag.summary()
        active = [
            t.task_id
            for t in dag.all_tasks()
            if t.status.value in {"running", "dispatched", "ready"}
        ]
        return {
            "task_count": len(dag.all_tasks()),
            "status_counts": summary,
            "active_task_ids": active,
        }

    @staticmethod
    def _build_l0_index(index: dict[str, list[str]]) -> dict[str, list[str]]:
        return {section: list(entries[:20]) for section, entries in index.items()}

    @staticmethod
    def _build_unresolved_issues(issues: list[UnresolvedIssue]) -> dict[str, Any]:
        return {
            "issues": [
                {
                    "kind": issue.kind,
                    "description": issue.description,
                    "related_task_id": issue.related_task_id,
                }
                for issue in issues
            ]
        }

    @staticmethod
    def _build_budget(job: ReproductionJob) -> dict[str, Any]:
        exhausted, reason = job.budget_exhausted()
        return {
            "limits": job.budget.to_dict(),
            "usage": {
                "gpu_hours_used": job.gpu_hours_used,
                "model_call_cost_usd": job.model_call_cost_usd,
                "model_input_tokens_used": job.model_input_tokens_used,
                "model_output_tokens_used": job.model_output_tokens_used,
                "model_calls_made": job.model_calls_made,
            },
            "exhausted": exhausted,
            "exhaustion_reason": reason,
        }
