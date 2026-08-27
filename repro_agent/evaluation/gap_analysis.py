"""结果差距触发条件综合判定（设计文档 §11.1 完整触发条件清单）。

``reflection_controller.ReflectionController.result_gap_detected`` 只
覆盖了"关键指标超出容差"这一最容易机器判定的条件；本模块补充其余
几条需要更多上下文的触发条件的判定辅助函数，供主智能体在处理结果
验证子智能体的输出时组合使用：

    - 关键指标超出容差（已在 reflection_controller 覆盖）；
    - 方法与 baseline 排序不一致；
    - 消融实验趋势不一致；
    - 指标异常偏高或偏低；
    - 结果波动远大于论文报告；
    - 训练曲线明显异常；
    - 实验运行成功但论文结论未复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from repro_agent.domain.experiment import MetricComparison


@dataclass
class GapSignal:
    triggered: bool
    reason: str


def ranking_inconsistent(
    reproduced_ranking: list[str], paper_ranking: list[str]
) -> GapSignal:
    """方法与 baseline 排序不一致（比较两个方法名有序列表的相对顺序）。"""

    if reproduced_ranking == paper_ranking:
        return GapSignal(triggered=False, reason="")
    return GapSignal(
        triggered=True,
        reason=f"复现排序 {reproduced_ranking} 与论文排序 {paper_ranking} 不一致",
    )


def ablation_trend_inconsistent(
    reproduced_deltas: dict[str, float], paper_deltas: dict[str, float]
) -> GapSignal:
    """消融实验趋势不一致：同一个消融项在复现和论文中的增益方向（正/负）不同。"""

    inconsistent = []
    for key, paper_delta in paper_deltas.items():
        reproduced_delta = reproduced_deltas.get(key)
        if reproduced_delta is None:
            continue
        if (paper_delta > 0) != (reproduced_delta > 0):
            inconsistent.append(key)
    if not inconsistent:
        return GapSignal(triggered=False, reason="")
    return GapSignal(triggered=True, reason=f"消融项趋势不一致: {inconsistent}")


def metric_abnormally_extreme(
    reproduced_value: float, paper_value: float, *, extreme_ratio: float = 2.0
) -> GapSignal:
    """指标异常偏高或偏低：复现值与论文值相差超过 ``extreme_ratio`` 倍。"""

    if paper_value == 0:
        return GapSignal(triggered=False, reason="")
    ratio = reproduced_value / paper_value if paper_value != 0 else float("inf")
    if ratio > extreme_ratio or ratio < (1 / extreme_ratio):
        return GapSignal(
            triggered=True,
            reason=f"复现值 {reproduced_value} 相对论文值 {paper_value} 比例异常 ({ratio:.2f}x)",
        )
    return GapSignal(triggered=False, reason="")


def variance_exceeds_paper_report(
    reproduced_std: float, paper_reported_std: float, *, factor: float = 3.0
) -> GapSignal:
    """结果波动远大于论文报告（超过论文报告标准差的 ``factor`` 倍）。"""

    if reproduced_std > paper_reported_std * factor:
        return GapSignal(
            triggered=True,
            reason=f"复现结果标准差 {reproduced_std} 远超论文报告的 {paper_reported_std} "
            f"({factor}x 阈值)",
        )
    return GapSignal(triggered=False, reason="")


def training_curve_anomalous(loss_history: list[float]) -> GapSignal:
    """训练曲线明显异常：出现 NaN、持续上升、或长期不下降。"""

    if not loss_history:
        return GapSignal(triggered=False, reason="")
    if any(v != v for v in loss_history):  # NaN 检测（NaN != NaN）
        return GapSignal(triggered=True, reason="训练 loss 出现 NaN")
    if len(loss_history) >= 10:
        first_half_avg = sum(loss_history[: len(loss_history) // 2]) / (len(loss_history) // 2)
        second_half_avg = sum(loss_history[len(loss_history) // 2 :]) / (
            len(loss_history) - len(loss_history) // 2
        )
        if second_half_avg >= first_half_avg:
            return GapSignal(triggered=True, reason="训练 loss 后半段未能低于前半段，疑似未收敛")
    return GapSignal(triggered=False, reason="")


def pipeline_succeeded_but_conclusion_not_reproduced(
    comparisons: list[MetricComparison], pipeline_exit_code: int
) -> GapSignal:
    """实验运行成功但论文结论未复现：管线本身跑通（exit_code==0），
    但关键指标显著偏离，说明"流程正确、结果不对"，这类 gap 通常更
    值得深入审计（相对于"流程都没跑通"的情况，更可能是参数/数据/
    理解层面的问题而非纯粹的工程 bug）。
    """

    if pipeline_exit_code != 0:
        return GapSignal(triggered=False, reason="")
    failing = [c for c in comparisons if not c.within_tolerance]
    if not failing:
        return GapSignal(triggered=False, reason="")
    return GapSignal(
        triggered=True,
        reason=f"管线成功执行但 {len(failing)} 个指标未复现论文结论: "
        f"{[c.metric for c in failing]}",
    )
