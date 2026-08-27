"""主智能体决策上下文管理（设计文档 §16：上下文压缩 + 上下文恢复）。"""

from repro_agent.context.budget import (
    CompressionResult,
    ContextSegment,
    ContextSegmentKind,
    ContextSource,
    SegmentPriority,
    compress_segments,
)
from repro_agent.context.builder import (
    ContextBuilder,
    ContextEnvelope,
    DecisionContext,
    UnresolvedIssue,
)
from repro_agent.context.snapshot import ContextSnapshot, SnapshotStore

__all__ = [
    "CompressionResult",
    "ContextBuilder",
    "ContextEnvelope",
    "ContextSegment",
    "ContextSegmentKind",
    "ContextSnapshot",
    "ContextSource",
    "DecisionContext",
    "SegmentPriority",
    "SnapshotStore",
    "UnresolvedIssue",
    "compress_segments",
]
