"""上下文长度预算与分段压缩策略（设计文档 §16.1）。

设计文档把"决策上下文"要读取的信息分成固定的九段流水线（Job 状态 →
DAG 摘要 → 当前决策 → L0 索引 → 相关 L1 → 必要时的 L2/L3 → 最近事件
→ 未解决问题 → 当前预算 → 压缩到上下文限制），本模块提供压缩这最后
一步：把各段渲染成文本后，按"必须保留 / 优先压缩"的分类丢弃低优先级
内容直到落在预算内。

复用来源：
    压缩顺序与优先级直接来自设计文档 §16.1 原文的两张清单
    （"优先压缩" 6 项 / "必须保留" 9 项）。具体的"按段落估算 token
    数、超预算时从最低优先级开始砍"的实现方式参考了 DeepCode
    ``ConciseMemoryAgent``"结构化摘要压缩上下文"以及 DeerFlow
    Checkpoint 系统里"full snapshot + delta"分层持久化对"不是所有
    历史都要全量保留"这一理念的共同体现（详见
    ``doc/DeepCode_Paper2Code_架构分析.md`` 第 8 节、
    ``doc/DeerFlow_架构分析.md`` 第 4 节）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class SegmentPriority(IntEnum):
    """数值越大越优先保留。"""

    COMPRESSIBLE_LOW = 0  # 已解决错误、重复事件——最先被砍
    COMPRESSIBLE_MEDIUM = 1  # 成功工具调用日志、已完成任务详情
    COMPRESSIBLE_HIGH = 2  # 与当前决策无关但可能仍有用的信息
    MUST_KEEP = 3  # 用户目标/系统约束/活跃任务等——绝不压缩


class ContextSegmentKind(str, Enum):
    """上下文段的业务语义类型。

    这里只描述内容是什么，不表达安全等级或指令优先级。后续若需要增加
    内容治理策略，可以在不改变现有 Envelope 形状的前提下扩展元数据。
    """

    JOB_STATE = "job_state"
    TASK_GRAPH = "task_graph"
    CURRENT_DECISION = "current_decision"
    MEMORY_INDEX = "memory_index"
    MEMORY_ENTRY = "memory_entry"
    EVENT_STREAM = "event_stream"
    ISSUE_LIST = "issue_list"
    BUDGET_STATE = "budget_state"


class ContextSource(str, Enum):
    """产生上下文段的系统组件。"""

    JOB = "job"
    SCHEDULER = "scheduler"
    ORCHESTRATOR = "orchestrator"
    MEMORY = "memory"
    EVENT_STORE = "event_store"


@dataclass
class ContextSegment:
    """一段可独立评估、压缩和序列化的结构化上下文内容。"""

    name: str
    kind: ContextSegmentKind
    source: ContextSource
    content: Any
    priority: SegmentPriority
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "source": self.source.value,
            "priority": self.priority.name.lower(),
            "content": self.content,
            "metadata": self.metadata,
        }

    def serialized(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def approx_tokens(self) -> int:
        # 粗略估算：中英文混合场景下，1 token ≈ 2.2 个字符，
        # 足够用于预算裁剪的相对比较，不追求和具体分词器完全一致。
        return max(1, int(len(self.serialized()) / 2.2))

    def truncated(self, max_chars: int) -> "ContextSegment":
        """返回保持 Envelope 合法的截断副本。

        不能直接切整个 JSON 字符串，否则会得到不可解析的上下文。这里把
        原内容序列化后放入结构化 preview，并显式标记发生了截断。
        """

        serialized_content = json.dumps(self.content, ensure_ascii=False, sort_keys=True)
        return ContextSegment(
            name=self.name,
            kind=self.kind,
            source=self.source,
            content={
                "truncated": True,
                "original_content_type": type(self.content).__name__,
                "preview": serialized_content[:max(0, max_chars)],
            },
            priority=self.priority,
            metadata={**self.metadata, "content_truncated": True},
        )


@dataclass
class CompressionResult:
    kept_segments: list[ContextSegment] = field(default_factory=list)
    dropped_segments: list[str] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 0

    def render(self) -> str:
        """兼容调用方的结构化 JSON 渲染。"""

        return json.dumps(
            [segment.to_dict() for segment in self.kept_segments],
            ensure_ascii=False,
            indent=2,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "approx_tokens": self.total_tokens,
            "kept_segment_names": [segment.name for segment in self.kept_segments],
            "dropped_segment_names": list(self.dropped_segments),
        }


def compress_segments(
    segments: list[ContextSegment], *, max_tokens: int
) -> CompressionResult:
    """按优先级从低到高逐段丢弃，直到落在 ``max_tokens`` 预算内。

    ``MUST_KEEP`` 段永远不会被丢弃——即使总量超出预算，也只截断
    (而非丢弃) 这类段落的内容，保证"用户目标/活跃任务/当前错误"等
    关键信息始终存在于上下文中（§16.1 "必须保留"清单）。
    """

    total = sum(s.approx_tokens() for s in segments)
    dropped: list[str] = []
    working = list(segments)

    # 按优先级从低到高排序，逐个丢弃直到预算达标或只剩 MUST_KEEP。
    while total > max_tokens:
        droppable = [s for s in working if s.priority != SegmentPriority.MUST_KEEP]
        if not droppable:
            break
        droppable.sort(key=lambda s: s.priority)
        victim = droppable[0]
        working.remove(victim)
        dropped.append(victim.name)
        total -= victim.approx_tokens()

    # 如果连 MUST_KEEP 段本身都超预算，做保底截断。截断后仍保持每一段
    # 是合法结构，而不是对最终 JSON 字符串做可能破坏语法的字符切片。
    if total > max_tokens:
        must_keep_budget = max_tokens // max(1, sum(1 for s in working))
        truncated = []
        for seg in working:
            if seg.priority == SegmentPriority.MUST_KEEP and seg.approx_tokens() > must_keep_budget:
                cut_chars = int(must_keep_budget * 2.2)
                truncated.append(seg.truncated(cut_chars))
            else:
                truncated.append(seg)
        working = truncated
        total = sum(s.approx_tokens() for s in working)

    # 保持原始相对顺序，方便阅读时逻辑连贯
    order_index = {s.name: i for i, s in enumerate(segments)}
    working.sort(key=lambda s: order_index.get(s.name, 0))

    return CompressionResult(
        kept_segments=working,
        dropped_segments=dropped,
        total_tokens=total,
        max_tokens=max_tokens,
    )
