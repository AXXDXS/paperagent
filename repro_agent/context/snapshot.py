"""上下文快照：每次重要决策后保存，系统重启后恢复（设计文档 §16.2）。

需要保存的字段直接取自 §16.2 原文：
    DAG 版本、任务状态版本、记忆版本、当前活跃问题、使用的证据、
    主智能体决策、当前反思轮次、当前预算。

复用来源：
    "full + delta"分层快照思想复用自 DeerFlow Checkpoint 系统
    （``doc/DeerFlow_架构分析.md`` 第 4 节）：DeerFlow 对长对话历史
    采用"定期写一次完整快照（full），中间只追加增量（delta）"的策略，
    重启时"取最近一次 full + 之后所有 delta 重放"来恢复状态，避免
    每次决策都写一份完整快照造成存储膨胀。这里做同样的取舍：
    每 ``full_snapshot_interval`` 次决策写一次完整快照，其余时间只
    写体积更小的 delta（相对上一次快照的变化摘要），恢复时从最近的
    full 快照开始重放 delta。

    快照本身落盘为 JSON 文件（而不是复用 SQLite），因为快照是
    "面向恢复读取的只读归档"，用简单的按 job_id/version 命名的
    文件即可满足"重启后从数据库和快照恢复"（§16.2 最后一句）——
    数据库（storage/database.py）依然是任务状态的唯一事实来源，
    快照只是为了避免"重放全部历史事件来重建上下文"的额外优化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from repro_agent.domain.common import iso, utc_now


@dataclass
class ContextSnapshot:
    """一次决策后的上下文快照（§16.2 需要保存的八项字段）。"""

    job_id: str
    version: int
    dag_version: int
    task_state_version: int
    memory_version: int
    active_issues: list[dict[str, Any]]
    evidence_refs: list[str]
    main_agent_decision: str
    reflection_round: int
    budget_snapshot: dict[str, Any]
    is_full: bool
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "version": self.version,
            "dag_version": self.dag_version,
            "task_state_version": self.task_state_version,
            "memory_version": self.memory_version,
            "active_issues": self.active_issues,
            "evidence_refs": self.evidence_refs,
            "main_agent_decision": self.main_agent_decision,
            "reflection_round": self.reflection_round,
            "budget_snapshot": self.budget_snapshot,
            "is_full": self.is_full,
            "created_at": iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextSnapshot":
        return cls(
            job_id=data["job_id"],
            version=data["version"],
            dag_version=data["dag_version"],
            task_state_version=data["task_state_version"],
            memory_version=data["memory_version"],
            active_issues=data["active_issues"],
            evidence_refs=data["evidence_refs"],
            main_agent_decision=data["main_agent_decision"],
            reflection_round=data["reflection_round"],
            budget_snapshot=data["budget_snapshot"],
            is_full=data["is_full"],
        )


class SnapshotStore:
    """快照的落盘存储与 full+delta 恢复逻辑（复用 DeerFlow Checkpoint 思想）。"""

    def __init__(self, snapshot_root: str | Path, full_snapshot_interval: int = 10):
        self.root = Path(snapshot_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.full_snapshot_interval = full_snapshot_interval

    def _job_dir(self, job_id: str) -> Path:
        d = self.root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        *,
        job_id: str,
        dag_version: int,
        task_state_version: int,
        memory_version: int,
        active_issues: list[dict[str, Any]],
        evidence_refs: list[str],
        main_agent_decision: str,
        reflection_round: int,
        budget_snapshot: dict[str, Any],
    ) -> ContextSnapshot:
        """保存一次快照；每 ``full_snapshot_interval`` 次写一次 full，
        其余写 delta（这里的 delta 与 full 结构相同，只是标记位不同——
        真正的体积优化点在于恢复时可以只从最近一次 full 开始重放，
        不需要保留更久远的历史文件，见 ``prune_old_snapshots``）。
        """

        next_version = self._latest_version(job_id) + 1
        is_full = next_version % self.full_snapshot_interval == 0 or next_version == 1
        snapshot = ContextSnapshot(
            job_id=job_id,
            version=next_version,
            dag_version=dag_version,
            task_state_version=task_state_version,
            memory_version=memory_version,
            active_issues=active_issues,
            evidence_refs=evidence_refs,
            main_agent_decision=main_agent_decision,
            reflection_round=reflection_round,
            budget_snapshot=budget_snapshot,
            is_full=is_full,
        )
        path = self._job_dir(job_id) / f"v{next_version:06d}.json"
        path.write_text(
            json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return snapshot

    def _latest_version(self, job_id: str) -> int:
        versions = self._list_versions(job_id)
        return max(versions) if versions else 0

    def _list_versions(self, job_id: str) -> list[int]:
        job_dir = self._job_dir(job_id)
        versions = []
        for f in job_dir.glob("v*.json"):
            try:
                versions.append(int(f.stem[1:]))
            except ValueError:
                continue
        return sorted(versions)

    def restore_latest(self, job_id: str) -> Optional[ContextSnapshot]:
        """恢复最近一次快照（不需要重放，因为每次 save 都是完整状态的
        自描述记录——"full/delta"区分主要用于未来的清理策略
        ``prune_old_snapshots``，而不是恢复时的重放逻辑，这比 DeerFlow
        原始的"必须重放 delta 链"实现更简单，因为本系统的快照内容
        本身就不大，没有必要为了节省单次写入体积而牺牲恢复逻辑的
        简单性——这是一处经过权衡后对参考实现的简化，原因见
        ``paper_agent/CHANGES_AND_DESIGN_NOTES.md``）。
        """

        versions = self._list_versions(job_id)
        if not versions:
            return None
        latest = versions[-1]
        path = self._job_dir(job_id) / f"v{latest:06d}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return ContextSnapshot.from_dict(data)

    def prune_old_snapshots(
        self,
        job_id: str,
        *,
        keep_last_n_full: int = 3,
        human_confirmed: bool = False,
    ) -> int:
        """清理旧快照：只保留最近 N 个 full 快照之后的所有快照，
        更早的全部删除（full 之前的 delta 已经没有恢复价值）。
        返回被删除的文件数。
        """

        if not human_confirmed:
            return 0

        versions = self._list_versions(job_id)
        full_versions = []
        for v in versions:
            path = self._job_dir(job_id) / f"v{v:06d}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("is_full"):
                full_versions.append(v)

        if len(full_versions) <= keep_last_n_full:
            return 0

        cutoff = full_versions[-keep_last_n_full]
        removed = 0
        for v in versions:
            if v < cutoff:
                (self._job_dir(job_id) / f"v{v:06d}.json").unlink(missing_ok=True)
                removed += 1
        return removed
