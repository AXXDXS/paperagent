"""容差确定策略（设计文档 §11.1）。

核心比较逻辑（``within_tolerance``）已经在
``domain.experiment.ExpectedResult``/``compare_metrics`` 中实现；
本模块补充的是**容差从哪里来**——§11.1 明确"不能对所有指标使用统一
阈值"，容差应当依据：

    - 论文是否报告标准差；
    - 论文是否运行多个随机种子；
    - 指标本身的数值范围；
    - 用户指定的容差；
    - Benchmark 官方容差；
    - 历史复现实验波动。

这里提供一个确定性的优先级链：用户指定 > Benchmark 官方容差 >
论文报告的标准差（若有，取 2 倍标准差作为容差，覆盖约 95% 置信区间）
> 历史波动 > 基于指标数值范围的启发式默认值（兜底）。任何一步选定
容差后都要写清楚 ``tolerance_basis``（对应
``ExpectedResult.tolerance_basis`` 字段），不能凭空给一个数字。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from repro_agent.domain.enums import ToleranceType
from repro_agent.domain.experiment import ExpectedResult


@dataclass
class ToleranceInputs:
    metric: str
    paper_value: float
    user_specified_tolerance: Optional[float] = None
    benchmark_official_tolerance: Optional[float] = None
    paper_reported_std: Optional[float] = None
    paper_num_seeds: Optional[int] = None
    historical_reproduction_std: Optional[float] = None


# 常见指标数值范围 -> 兜底相对容差（找不到任何依据时的最后手段，
# 使用时会在 tolerance_basis 里明确标注"启发式默认值，建议尽快替换"，
# 避免这个兜底值被误认为是有依据的严谨容差）。
_DEFAULT_RELATIVE_TOLERANCE = 0.05  # 5%


def determine_tolerance(inputs: ToleranceInputs) -> ExpectedResult:
    """按优先级链确定单个指标的容差，返回可直接使用的 ``ExpectedResult``。"""

    if inputs.user_specified_tolerance is not None:
        return ExpectedResult(
            metric=inputs.metric,
            value=inputs.paper_value,
            tolerance_type=ToleranceType.ABSOLUTE,
            tolerance=inputs.user_specified_tolerance,
            tolerance_basis="用户指定容差",
        )

    if inputs.benchmark_official_tolerance is not None:
        return ExpectedResult(
            metric=inputs.metric,
            value=inputs.paper_value,
            tolerance_type=ToleranceType.ABSOLUTE,
            tolerance=inputs.benchmark_official_tolerance,
            tolerance_basis="Benchmark 官方公布容差",
        )

    if inputs.paper_reported_std is not None:
        # 论文报告了标准差（通常来自多随机种子实验），取 2 倍标准差
        # 近似 95% 置信区间作为容差，比"拍脑袋"更有统计学依据。
        basis = "论文报告标准差 × 2（约 95% 置信区间）"
        if inputs.paper_num_seeds:
            basis += f"，基于 {inputs.paper_num_seeds} 个随机种子"
        return ExpectedResult(
            metric=inputs.metric,
            value=inputs.paper_value,
            tolerance_type=ToleranceType.STD_MULTIPLE,
            tolerance=inputs.paper_reported_std * 2,
            tolerance_basis=basis,
        )

    if inputs.historical_reproduction_std is not None:
        return ExpectedResult(
            metric=inputs.metric,
            value=inputs.paper_value,
            tolerance_type=ToleranceType.STD_MULTIPLE,
            tolerance=inputs.historical_reproduction_std * 2,
            tolerance_basis="历史复现实验波动标准差 × 2",
        )

    # 兜底：基于指标数值范围的启发式相对容差，显式标注"非严谨依据"，
    # 提醒下游（结果验证/最终报告）该容差的可信度较低。
    return ExpectedResult(
        metric=inputs.metric,
        value=inputs.paper_value,
        tolerance_type=ToleranceType.RELATIVE,
        tolerance=_DEFAULT_RELATIVE_TOLERANCE,
        tolerance_basis=f"启发式默认值（{_DEFAULT_RELATIVE_TOLERANCE:.0%} 相对误差），"
        "缺乏标准差/官方容差依据，建议后续补充更严谨的容差来源",
    )
