"""实验规格与运行记录（设计文档 §9.4、§10.5、§18.3）。

ExperimentSpec 对应 §9.4 的统一实验复现规格 YAML 示例；
ExperimentRun 对应 §18.3，要求绑定 git_commit/container_digest/
config_digest/dataset_digest/model_identifier/random_seed/
hardware_identifier 七元组（§10.5），保证每次正式实验都可追溯。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.domain.enums import ExperimentTier, FieldProvenance, ToleranceType


@dataclass
class ProvenancedField:
    """带来源标注的字段（§9.4：每个字段必须记录来源，冲突不能静默覆盖）。"""

    value: Any
    provenance: FieldProvenance
    source_ref: str = ""  # 例如 "paper.md#sec3.2" 或 "configs/train.yaml:12"
    confidence: float = 1.0
    conflicting_values: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_conflict(self) -> bool:
        return len(self.conflicting_values) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "source_ref": self.source_ref,
            "confidence": self.confidence,
            "conflicting_values": self.conflicting_values,
        }


@dataclass
class ExpectedResult:
    """论文预期指标及容差策略（§11.1）。"""

    metric: str
    value: float
    tolerance_type: ToleranceType = ToleranceType.ABSOLUTE
    tolerance: float = 0.0
    # 容差确定依据（§11.1）：论文是否报告标准差 / 多随机种子 /
    # benchmark 官方容差 / 用户指定 / 历史波动等，写清楚依据而非拍脑袋。
    tolerance_basis: str = ""

    def within_tolerance(self, reproduced_value: float) -> bool:
        diff = reproduced_value - self.value
        if self.tolerance_type == ToleranceType.ABSOLUTE:
            return abs(diff) <= self.tolerance
        if self.tolerance_type == ToleranceType.RELATIVE:
            denom = abs(self.value) if self.value != 0 else 1e-9
            return abs(diff) / denom <= self.tolerance
        if self.tolerance_type == ToleranceType.STD_MULTIPLE:
            return abs(diff) <= self.tolerance
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "tolerance_type": self.tolerance_type.value,
            "tolerance": self.tolerance,
            "tolerance_basis": self.tolerance_basis,
        }


@dataclass
class ExperimentSpec:
    """统一实验复现规格（§9.4）。"""

    experiment_id: str
    target_claim: str
    job_id: str
    expected_results: dict[str, ExpectedResult] = field(default_factory=dict)
    fields: dict[str, ProvenancedField] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    spec_id: str = field(default_factory=lambda: new_id("spec"))
    created_at: datetime = field(default_factory=utc_now)

    def unresolved_conflicts(self) -> list[str]:
        """返回存在冲突且未解决的字段名列表（§9.4：存在冲突时不能静默覆盖）。"""

        return [name for name, f in self.fields.items() if f.has_conflict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "experiment_id": self.experiment_id,
            "target_claim": self.target_claim,
            "job_id": self.job_id,
            "expected_results": {
                k: v.to_dict() for k, v in self.expected_results.items()
            },
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "resources": self.resources,
            "created_at": iso(self.created_at),
        }


@dataclass
class ExperimentRun(object):
    """实验运行记录（§18.3），必须绑定完整的可追溯七元组（§10.5）。"""

    experiment_id: str
    job_id: str
    tier: ExperimentTier
    run_id: str = field(default_factory=lambda: new_id("run"))
    run_type: str = "reduced"
    git_commit: str = ""
    container_digest: str = ""
    config_digest: str = ""
    dataset_digest: str = ""
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    model_identifier: str = ""
    seed: Optional[int] = None
    hardware_identifier: str = ""
    command: str = ""
    exit_code: Optional[int] = None
    metrics: dict[str, float] = field(default_factory=dict)
    log_path: str = ""
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    tier_command_verified: bool = False

    def is_fully_traceable(self) -> bool:
        """校验是否满足 §10.5 正式实验绑定要求（仅对 full_experiment 强制）。"""

        if self.tier != ExperimentTier.FULL_EXPERIMENT:
            return True
        required = [
            self.git_commit,
            self.container_digest,
            self.config_digest,
            self.model_identifier,
            self.hardware_identifier,
        ]
        return (
            all(bool(v) for v in required)
            and self._dataset_manifest_is_valid()
            and self.seed is not None
        )

    def _dataset_manifest_is_valid(self) -> bool:
        """Accept legacy digests, but validate the explicit v1 manifest when present."""

        if not self.dataset_manifest:
            return bool(self.dataset_digest)
        kind = self.dataset_manifest.get("kind")
        items = self.dataset_manifest.get("items")
        declared = self.dataset_manifest.get("digest", self.dataset_digest)
        if kind not in {"none", "external"} or not isinstance(items, list):
            return False
        if kind == "none" and items:
            return False
        semantic = {
            "version": self.dataset_manifest.get("version", 1),
            "kind": kind,
            "items": items,
        }
        digest = hashlib.sha256(
            json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return declared == f"dataset-manifest:v1:{digest}" and self.dataset_digest == declared

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "job_id": self.job_id,
            "tier": self.tier.value,
            "run_type": self.run_type,
            "git_commit": self.git_commit,
            "container_digest": self.container_digest,
            "config_digest": self.config_digest,
            "dataset_digest": self.dataset_digest,
            "dataset_manifest": self.dataset_manifest,
            "model_identifier": self.model_identifier,
            "seed": self.seed,
            "hardware_identifier": self.hardware_identifier,
            "command": self.command,
            "exit_code": self.exit_code,
            "metrics": self.metrics,
            "log_path": self.log_path,
            "started_at": iso(self.started_at),
            "completed_at": iso(self.completed_at),
            "tier_command_verified": self.tier_command_verified,
            "is_fully_traceable": self.is_fully_traceable(),
        }


@dataclass
class MetricComparison:
    """单指标对比结果（§11.1）。"""

    metric: str
    paper_value: float
    reproduced_value: float
    tolerance_type: ToleranceType
    tolerance: float
    within_tolerance: bool

    @property
    def difference(self) -> float:
        return self.reproduced_value - self.paper_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "paper_value": self.paper_value,
            "reproduced_value": self.reproduced_value,
            "difference": self.difference,
            "tolerance_type": self.tolerance_type.value,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


def compare_metrics(
    spec: ExperimentSpec, run: ExperimentRun
) -> list[MetricComparison]:
    """把实验规格中的预期指标与实际运行结果逐一比较（§11.1）。"""

    comparisons: list[MetricComparison] = []
    for metric, expected in spec.expected_results.items():
        if metric not in run.metrics:
            continue
        reproduced = run.metrics[metric]
        comparisons.append(
            MetricComparison(
                metric=metric,
                paper_value=expected.value,
                reproduced_value=reproduced,
                tolerance_type=expected.tolerance_type,
                tolerance=expected.tolerance,
                within_tolerance=expected.within_tolerance(reproduced),
            )
        )
    return comparisons
