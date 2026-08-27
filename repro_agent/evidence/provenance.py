"""证据链核心：产物 / 代码 / 配置 / 论文依据的哈希互锁记录。

复用来源：
    直接对应 paper-replication-paper 的"三个 wrapper 生成证据链"设计
    （``track-run`` → ``register-target-artifact`` → ``record-comparison``，
    见 ``doc/paper-replication-paper_架构分析.md`` 第 5.1 节），并结合
    设计文档 §10.5"正式实验必须绑定 git_commit/container_digest/
    config_digest/dataset_digest/model_identifier/random_seed/
    hardware_identifier"的七元组要求，把两者合并成统一的证据记录：

    - ``RunRecord``：对应 track-run，记录一次真实命令执行的完整凭证；
    - ``ArtifactProvenance``：对应 register-target-artifact，把某个
      产物文件与其代码/配置/论文依据的 SHA-256 锁定在一起；
    - ``ComparisonEvidence``：对应 record-comparison，记录复现结果与
      论文结果的比较依据。

    任何一环被事后篡改（比如偷偷换了实现代码却没有重新走一遍流程），
    ``verify_provenance`` 都会通过重新计算磁盘文件的哈希发现不一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.evidence.anti_cheat import (
    check_code_path_not_inside_paper_or_artifacts,
    scan_suspicious_markers,
)
from repro_agent.evidence.hashing import sha256_of_file


@dataclass
class RunRecord:
    """对应 track-run：真实执行一条命令的完整凭证。"""

    run_id: str
    task_id: str
    command: list[str]
    exit_code: int
    stdout_tail: str
    stderr_tail: str
    expected_output_paths: list[str] = field(default_factory=list)
    output_hashes: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "expected_output_paths": self.expected_output_paths,
            "output_hashes": self.output_hashes,
            "started_at": iso(self.started_at),
            "completed_at": iso(self.completed_at),
        }


@dataclass
class ArtifactProvenance:
    """对应 register-target-artifact：把产物与代码/配置/论文依据锁定。"""

    target_id: str
    run_id: str
    artifact_path: str
    code_path: str
    config_path: str
    paper_trace_path: str
    method_components: list[str] = field(default_factory=list)
    implementation_summary: str = ""
    claim_mode: str = "reproduction"  # "reproduction" | "baseline"
    provenance_id: str = field(default_factory=lambda: new_id("prov"))
    artifact_sha256: str = ""
    code_sha256: str = ""
    config_sha256: str = ""
    paper_trace_sha256: str = ""
    created_at: datetime = field(default_factory=utc_now)
    generated_by: str = "repro_agent.evidence.provenance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "target_id": self.target_id,
            "run_id": self.run_id,
            "artifact_path": self.artifact_path,
            "code_path": self.code_path,
            "config_path": self.config_path,
            "paper_trace_path": self.paper_trace_path,
            "method_components": self.method_components,
            "implementation_summary": self.implementation_summary,
            "claim_mode": self.claim_mode,
            "artifact_sha256": self.artifact_sha256,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "paper_trace_sha256": self.paper_trace_sha256,
            "created_at": iso(self.created_at),
            "generated_by": self.generated_by,
        }


@dataclass
class ComparisonEvidence:
    """对应 record-comparison：复现结果与论文结果的比较依据。"""

    target_id: str
    acceptance_mode: str
    comparison_metric: dict[str, Any]
    note: str = ""
    evidence_id: str = field(default_factory=lambda: new_id("cmp"))
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "target_id": self.target_id,
            "acceptance_mode": self.acceptance_mode,
            "comparison_metric": self.comparison_metric,
            "note": self.note,
            "created_at": iso(self.created_at),
        }


class ProvenanceError(RuntimeError):
    """证据链构建/校验失败（应导致该产物不能被判定为可信复现证据）。"""


def register_artifact_provenance(
    *,
    target_id: str,
    run: RunRecord,
    artifact_path: str,
    code_path: str,
    config_path: str,
    paper_trace_path: str,
    method_components: list[str],
    implementation_summary: str,
    claim_mode: str = "reproduction",
    forbidden_dir_markers: tuple[str, ...] = ("artifacts/", "paper/"),
) -> ArtifactProvenance:
    """构建产物证据链记录，构建前先做反作弊检查（§5.1/§5.2）。

    与 paper-replication 的行为保持一致：
        - 校验 run 必须成功（exit_code == 0）；
        - code_path/config_path 不能位于 artifacts/、paper/ 目录下；
        - claim_mode="baseline" 时对方法标签/摘要做禁用词扫描；
        - 所有引用文件都必须真实存在，计算并落盘其 SHA-256。
    """

    if not run.succeeded:
        raise ProvenanceError(
            f"cannot register artifact provenance: run {run.run_id} did not succeed "
            f"(exit_code={run.exit_code})"
        )

    for path_label, path_value in (
        ("code_path", code_path),
        ("config_path", config_path),
    ):
        finding = check_code_path_not_inside_paper_or_artifacts(
            path_value, forbidden_dir_markers
        )
        if not finding.passed:
            raise ProvenanceError(f"{path_label} 校验失败: {'; '.join(finding.reasons)}")

    if claim_mode == "baseline":
        finding = scan_suspicious_markers(implementation_summary, " ".join(method_components))
        if not finding.passed:
            raise ProvenanceError(
                "baseline_faithful 声明未通过禁用词扫描: " + "; ".join(finding.reasons)
            )

    def _hash_or_raise(label: str, path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise ProvenanceError(f"{label} 文件不存在: {path}")
        return sha256_of_file(p)

    provenance = ArtifactProvenance(
        target_id=target_id,
        run_id=run.run_id,
        artifact_path=artifact_path,
        code_path=code_path,
        config_path=config_path,
        paper_trace_path=paper_trace_path,
        method_components=method_components,
        implementation_summary=implementation_summary,
        claim_mode=claim_mode,
        artifact_sha256=_hash_or_raise("artifact_path", artifact_path),
        code_sha256=_hash_or_raise("code_path", code_path),
        config_sha256=_hash_or_raise("config_path", config_path),
        paper_trace_sha256=_hash_or_raise("paper_trace_path", paper_trace_path),
    )
    return provenance


def verify_provenance(provenance: ArtifactProvenance) -> tuple[bool, list[str]]:
    """重新计算磁盘上文件的哈希，与登记时的哈希比对，检测事后篡改（§5.2）。"""

    errors = []
    checks = [
        ("artifact_path", provenance.artifact_path, provenance.artifact_sha256),
        ("code_path", provenance.code_path, provenance.code_sha256),
        ("config_path", provenance.config_path, provenance.config_sha256),
        ("paper_trace_path", provenance.paper_trace_path, provenance.paper_trace_sha256),
    ]
    for label, path, expected_hash in checks:
        p = Path(path)
        if not p.exists():
            errors.append(f"{label} 文件已不存在: {path}")
            continue
        actual_hash = sha256_of_file(p)
        if actual_hash != expected_hash:
            errors.append(
                f"{label} 哈希不匹配（可能被事后篡改）: expected={expected_hash[:12]}..., "
                f"actual={actual_hash[:12]}..."
            )
    return len(errors) == 0, errors
