"""正式记忆管理器（设计文档 §15）。

目录结构严格对齐 §15.1：

    /project_memory/<job_id>/
    ├── index/         L0 索引
    ├── paper/
    ├── code/
    ├── data/
    ├── model/
    ├── environment/
    ├── experiments/
    ├── comparison/
    ├── reflection/
    ├── tasks/
    ├── failures/
    ├── evidence/
    └── archive/

访问控制（§3 原则 12-14、§15.1）：
    "只有主智能体可以读取正式记忆；Memory Manager 可以读写；
    子智能体不可读取。"

    本实现用 ``_MainAgentToken`` 这种"权限令牌"模式在类型系统层面
    体现这一约束：``MemoryManager.read_l0`` / ``read_l1`` 等读接口
    要求调用方传入一个只有 orchestrator 层才会持有的
    ``MainAgentCapability`` 实例；子智能体运行时代码（agents/* 包）
    永远不会被注入这个 capability 对象，因此即使子智能体代码
    "手滑" import 了 MemoryManager，也无法通过类型/运行时检查读取
    正式记忆——这是比"文档约定"更强的工程约束。

复用来源：
    Markdown 文件 + 渐进式披露（L0 索引 → L1 摘要 → L2 细节 → L3
    证据）的整体思路来自设计文档 §15.3 原文；"写完即可归档、用
    结构化摘要代替完整历史"这一压缩理念参考了 DeepCode
    ``ConciseMemoryAgent``"写文件即清空上下文，用结构化摘要文件
    代替完整对话历史"的设计（``doc/DeepCode_Paper2Code_架构分析.md``
    第 8 节），在这里体现为 L1 摘要文件本身就是"可以单独喂给主智能体
    上下文、不需要连着 L2/L3 一起加载"的独立单元。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_agent.memory.candidate import CandidateMemory
from repro_agent.memory.validation import MemoryValidationResult, validate_candidate

_MEMORY_SECTIONS = (
    "index",
    "paper",
    "code",
    "data",
    "model",
    "environment",
    "experiments",
    "comparison",
    "reflection",
    "tasks",
    "failures",
    "evidence",
    "archive",
)

# 主题到目录分区的映射：约定候选记忆的 topic 前缀决定它落在哪个分区，
# 例如 topic="paper.main_experiment_params" -> paper/ 分区。
_DEFAULT_SECTION = "tasks"


@dataclass(frozen=True)
class MainAgentCapability:
    """只有 orchestrator 构造并持有的"主智能体身份令牌"。

    子智能体代码路径中永远不会创建或接收这个对象的实例——它不从
    任何公共工厂函数中默认产生，只在 ``orchestrator/main_agent.py``
    初始化时构造一次并持有，从而在实践中保证"子智能体不可读取正式
    记忆"（§3 原则 12-14）。
    """

    job_id: str


class MemoryPermissionError(PermissionError):
    """尝试在没有 ``MainAgentCapability`` 的情况下读取正式记忆。"""


class MemoryManager:
    """正式记忆的读写管理器（§15.1："Memory Manager 可以读写"）。"""

    def __init__(self, job_id: str, memory_root: str | Path):
        self.job_id = job_id
        self.root = Path(memory_root) / job_id
        for section in _MEMORY_SECTIONS:
            (self.root / section).mkdir(parents=True, exist_ok=True)

    # ---- 写入（Memory Manager 拥有读写权限，不需要 capability） ----

    def promote_candidate(
        self, candidate: CandidateMemory, *, section: str | None = None
    ) -> MemoryValidationResult:
        """把候选记忆经过验证流水线后写入正式 Markdown（§15.2）。"""

        target_section = section or self._infer_section(candidate.topic)
        existing = self._load_section_candidates(target_section)
        result = validate_candidate(candidate, existing)
        if not result.accepted:
            return result

        file_path = self._entry_path(target_section, candidate.topic)
        file_path.write_text(candidate.to_markdown(), encoding="utf-8")
        self._update_index(target_section, candidate)
        return result

    def _infer_section(self, topic: str) -> str:
        prefix = topic.split(".", 1)[0].lower()
        return prefix if prefix in _MEMORY_SECTIONS else _DEFAULT_SECTION

    def _entry_path(self, section: str, topic: str) -> Path:
        safe_name = topic.replace("/", "_").replace(" ", "_")
        return self.root / section / f"{safe_name}.md"

    def _load_section_candidates(self, section: str) -> list[CandidateMemory]:
        """从已写入的 Markdown 反解出足够做冲突检测的最小结构。

        注意：这里不追求"完美反解 Markdown"，只提取 topic/summary，
        details 的冲突检测依赖调用方在同一进程内传入完整候选对象；
        跨进程重启后的冲突检测精度会降级为"仅按 topic 去重提示"，
        这是一个显式的简化取舍（完整实现需要为记忆条目额外维护一份
        结构化的 sidecar JSON，超出当前 MVP 范围）。
        """

        section_dir = self.root / section
        results = []
        for md_file in section_dir.glob("*.md"):
            topic = md_file.stem
            results.append(
                CandidateMemory(task_id="", topic=topic, summary="", details={})
            )
        return results

    def _update_index(self, section: str, candidate: CandidateMemory) -> None:
        """维护 L0 索引文件：每个分区一个 index 文件，列出主题清单。"""

        index_path = self.root / "index" / f"{section}.md"
        line = f"- {candidate.topic}: {candidate.summary[:80]}"
        existing_lines = []
        if index_path.exists():
            existing_lines = index_path.read_text(encoding="utf-8").splitlines()
        # 去重：同一 topic 只保留最新一条摘要
        filtered = [
            l for l in existing_lines if not l.startswith(f"- {candidate.topic}:")
        ]
        filtered.append(line)
        index_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

    def append_reflection(self, reflection_markdown: str, reflection_id: str) -> None:
        path = self.root / "reflection" / f"{reflection_id}.md"
        path.write_text(reflection_markdown, encoding="utf-8")

    def append_failure(self, failure_markdown: str, task_id: str) -> None:
        path = self.root / "failures" / f"{task_id}.md"
        path.write_text(failure_markdown, encoding="utf-8")

    # ---- 读取（仅主智能体可读，§3 原则 12） ----

    def read_l0_index(self, capability: MainAgentCapability) -> dict[str, list[str]]:
        self._require_capability(capability)
        index: dict[str, list[str]] = {}
        index_dir = self.root / "index"
        for md_file in sorted(index_dir.glob("*.md")):
            index[md_file.stem] = md_file.read_text(encoding="utf-8").splitlines()
        return index

    def read_l1_summary(
        self, capability: MainAgentCapability, section: str, topic: str
    ) -> str | None:
        """读取 L1 摘要（默认读取粒度，§15.3："主智能体默认读取 L0 和相关 L1"）。"""

        self._require_capability(capability)
        path = self._entry_path(section, topic)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return _extract_section(text, "## 摘要 (L1)", "## 细节 (L2)")

    def read_full_entry(
        self, capability: MainAgentCapability, section: str, topic: str
    ) -> str | None:
        """读取完整条目（含 L2 细节 + L3 证据），仅在验证/冲突处理/反思
        审计时才展开（§15.3："只有验证、冲突处理和反思审计时才展开
        L2、L3"）——是否展开的决策权在调用方（orchestrator），本方法
        只负责在被合法调用时提供完整内容。
        """

        self._require_capability(capability)
        path = self._entry_path(section, topic)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _require_capability(capability: Any) -> None:
        if not isinstance(capability, MainAgentCapability):
            raise MemoryPermissionError(
                "reading formal project memory requires a MainAgentCapability token; "
                "sub-agents must never receive one (design doc §3 principle 12)"
            )


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text:
        return ""
    start = text.index(start_marker) + len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()
