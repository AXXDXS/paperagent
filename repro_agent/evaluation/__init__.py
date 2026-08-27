"""分级实验执行门禁 + 容差策略 + 结果差距综合判定（设计文档 §10、§11.1）。"""

from repro_agent.evaluation.gap_analysis import (
    GapSignal,
    ablation_trend_inconsistent,
    metric_abnormally_extreme,
    pipeline_succeeded_but_conclusion_not_reproduced,
    ranking_inconsistent,
    training_curve_anomalous,
    variance_exceeds_paper_report,
)
from repro_agent.evaluation.tier_gate import TierGate, TierGateDecision
from repro_agent.evaluation.tolerance import ToleranceInputs, determine_tolerance

__all__ = [
    "GapSignal",
    "TierGate",
    "TierGateDecision",
    "ToleranceInputs",
    "ablation_trend_inconsistent",
    "determine_tolerance",
    "metric_abnormally_extreme",
    "pipeline_succeeded_but_conclusion_not_reproduced",
    "ranking_inconsistent",
    "training_curve_anomalous",
    "variance_exceeds_paper_report",
]
