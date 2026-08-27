"""候选记忆（设计文档 §15.2）。

流程（§15.2 原文）：
    子智能体生成候选记忆
        → 主智能体验证
        → Memory Manager 检查
        → 冲突检查
        → 敏感信息检查
        → 写入正式 Markdown

子智能体只能写到自己任务沙箱的 ``output/candidate_memory.md``
（§15.2），不能直接写入正式记忆目录；只有通过下面这条流水线的
候选记忆才能"转正"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from repro_agent.domain.common import iso, new_id, utc_now


@dataclass
class CandidateMemory:
    """子智能体产出的候选记忆条目。"""

    task_id: str
    topic: str  # 记忆主题，对应 L0 索引的一个条目
    summary: str  # L1 摘要
    details: dict[str, Any] = field(default_factory=dict)  # L2 细节
    evidence_refs: list[str] = field(default_factory=list)  # L3 证据引用（论文原文/代码行/日志路径）
    candidate_id: str = field(default_factory=lambda: new_id("cand_mem"))
    created_at: datetime = field(default_factory=utc_now)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.topic}",
            "",
            f"- candidate_id: {self.candidate_id}",
            f"- task_id: {self.task_id}",
            f"- created_at: {iso(self.created_at)}",
            "",
            "## 摘要 (L1)",
            self.summary,
            "",
            "## 细节 (L2)",
        ]
        for key, value in self.details.items():
            lines.append(f"- **{key}**: {value}")
        lines.extend(["", "## 证据 (L3)"])
        for ref in self.evidence_refs:
            lines.append(f"- {ref}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "topic": self.topic,
            "summary": self.summary,
            "details": self.details,
            "evidence_refs": self.evidence_refs,
            "created_at": iso(self.created_at),
        }
