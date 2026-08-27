"""分级实验执行门禁（设计文档 §10："系统不得直接运行正式实验"）。

五级门禁严格线性递进：
    STATIC_CHECK → UNIT_TEST → SMOKE_TEST → REDUCED_EXPERIMENT → FULL_EXPERIMENT

``TierGate`` 是本系统里"系统不得直接运行正式实验"这条硬约束的唯一
执行点：任何想要创建 ``FULL_EXPERIMENT`` 级任务的代码路径（包括
反思闭环的"最小范围重跑"）都必须先调用 ``TierGate.can_advance``，
拿到明确的"允许晋级"判定后才能创建任务，orchestrator 不会绕过这个
检查直接拼装任务。
"""

from __future__ import annotations

from dataclasses import dataclass

from repro_agent.domain.enums import ExperimentTier
from repro_agent.domain.experiment import ExperimentRun

_TIER_ORDER = [
    ExperimentTier.STATIC_CHECK,
    ExperimentTier.UNIT_TEST,
    ExperimentTier.SMOKE_TEST,
    ExperimentTier.REDUCED_EXPERIMENT,
    ExperimentTier.FULL_EXPERIMENT,
]


@dataclass
class TierGateDecision:
    allowed: bool
    reason: str = ""
    next_tier: ExperimentTier | None = None


class TierGate:
    """依据既往运行记录判定是否允许晋级到下一级实验。"""

    def evaluate(self, experiment_id: str, runs: list[ExperimentRun]) -> TierGateDecision:
        """给定某个实验目标已有的所有运行记录，判定下一步应该跑哪一级。

        规则：从 STATIC_CHECK 开始，找到第一个"尚未有成功记录"的等级，
        即为下一步应该执行的等级；如果该等级的前置等级还没有成功记录，
        则不允许直接跳级（例如没有冒烟测试成功记录，不允许创建缩小
        规模实验任务）。
        """

        succeeded_tiers = {
            r.tier for r in runs if r.experiment_id == experiment_id and r.exit_code == 0
        }

        for idx, tier in enumerate(_TIER_ORDER):
            if tier in succeeded_tiers:
                continue
            if idx == 0:
                return TierGateDecision(allowed=True, next_tier=tier)
            previous_tier = _TIER_ORDER[idx - 1]
            if previous_tier not in succeeded_tiers:
                return TierGateDecision(
                    allowed=False,
                    reason=f"前置等级 {previous_tier.value} 尚未成功通过，"
                    f"不允许直接创建 {tier.value} 任务",
                )
            return TierGateDecision(allowed=True, next_tier=tier)

        # 所有等级都已成功，说明正式实验也已跑通，理论上不应再"晋级"，
        # 但允许重复运行正式实验（例如反思闭环触发的重跑）。
        return TierGateDecision(allowed=True, next_tier=ExperimentTier.FULL_EXPERIMENT)

    def can_run_full_experiment(self, experiment_id: str, runs: list[ExperimentRun]) -> bool:
        """§10.5: "只有前四级通过后才能运行"正式实验的显式判定。"""

        required = _TIER_ORDER[:-1]
        succeeded_tiers = {
            r.tier for r in runs if r.experiment_id == experiment_id and r.exit_code == 0
        }
        return all(t in succeeded_tiers for t in required)
