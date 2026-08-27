"""最终复现结论判定（设计文档 §2 复现成功的定义 + §20 第十部分）。

把 §2 给出的十一种 ``ReproductionStatus`` 收敛为一个确定性判定函数：
给定"管线是否跑通到正式实验""指标对比结果""是否因资源缺失阻塞"
"反思审计是否confirmed 存在流程问题"这几项事实，机械地推导出最终
状态，不依赖 LLM 主观判断——"是否复现成功"这个结论必须是可复算、
可审计的，不能是一次性的模型输出。
"""

from __future__ import annotations

from dataclasses import dataclass

from repro_agent.domain.enums import ReproductionStatus
from repro_agent.domain.experiment import MetricComparison


@dataclass
class ConclusionInputs:
    environment_ready: bool
    pipeline_executable: bool
    smoke_test_passed: bool
    reduced_experiment_passed: bool
    full_experiment_completed: bool
    blocked_by_missing_resource: bool
    comparisons: list[MetricComparison]
    # 反思审计确认"流程/配置/代码本身有问题但未修复"时，不能声称
    # NOT_REPRODUCED（那意味着"流程正确但结果没对上"），应该更谨慎地
    # 停留在更早的状态，等修复后重新判定。
    reflection_confirmed_unresolved_issue: bool = False
    trend_consistent: bool = False  # 消融/排序趋势是否与论文一致


def determine_final_status(inputs: ConclusionInputs) -> ReproductionStatus:
    """按 §2 的定义顺序（由弱到强）逐级判定，返回能达到的最高状态。"""

    if inputs.blocked_by_missing_resource:
        return ReproductionStatus.BLOCKED_BY_MISSING_RESOURCE

    if not inputs.environment_ready:
        # 环境都没搭起来，理论上不应该走到这个判定函数，兜底返回
        # PIPELINE_ONLY 之前的最低状态由调用方直接使用 JobStatus 表达，
        # 这里返回 NOT_REPRODUCED 作为保守兜底。
        return ReproductionStatus.NOT_REPRODUCED

    if not inputs.pipeline_executable:
        return ReproductionStatus.ENVIRONMENT_READY

    if not inputs.smoke_test_passed:
        return ReproductionStatus.PIPELINE_EXECUTABLE

    if not inputs.full_experiment_completed:
        return ReproductionStatus.REDUCED_EXPERIMENT_PASSED if inputs.reduced_experiment_passed else ReproductionStatus.SMOKE_TEST_PASSED

    if not inputs.comparisons:
        # 正式实验跑完了，但没有可比较的指标（比如论文未报告可比指标），
        # 只能停留在"仅跑通流程"。
        return ReproductionStatus.PIPELINE_ONLY

    if inputs.reflection_confirmed_unresolved_issue:
        # §3 原则 23："反思流程确认执行过程无误后才能向用户报告真实差距"。
        # 如果确认了有未修复的流程问题，不能声称任何"复现"结论，
        # 只能停留在 FULL_EXPERIMENT_COMPLETED（跑完了，但结论不可信）。
        return ReproductionStatus.FULL_EXPERIMENT_COMPLETED

    all_within = all(c.within_tolerance for c in inputs.comparisons)
    if all_within:
        return ReproductionStatus.FULLY_REPRODUCED

    any_within = any(c.within_tolerance for c in inputs.comparisons)
    if any_within:
        return ReproductionStatus.PARTIALLY_REPRODUCED

    if inputs.trend_consistent:
        return ReproductionStatus.TREND_REPRODUCED

    return ReproductionStatus.NOT_REPRODUCED


def render_conclusion_text(status: ReproductionStatus, comparisons: list[MetricComparison]) -> str:
    """把最终状态渲染成 §20 第十部分要求的自然语言结论文本。"""

    descriptions = {
        ReproductionStatus.FULLY_REPRODUCED: "所有关键指标均落在容差范围内，判定为**完全复现**。",
        ReproductionStatus.PARTIALLY_REPRODUCED: "主要实验已完成，但部分指标存在差距，判定为**部分复现**。",
        ReproductionStatus.TREND_REPRODUCED: "绝对指标数值与论文不同，但主要趋势一致，判定为**趋势复现**。",
        ReproductionStatus.PIPELINE_ONLY: "实验流程已成功跑通，但尚未验证或未能验证论文指标，判定为**只跑通流程**。",
        ReproductionStatus.ENVIRONMENT_READY: "复现环境已经构建，但实验管线尚未被证明可执行。",
        ReproductionStatus.PIPELINE_EXECUTABLE: "实验管线可以执行，但尚未通过冒烟测试。",
        ReproductionStatus.NOT_REPRODUCED: "实验流程正确执行，但复现结果未能满足论文指标，判定为**未复现**。",
        ReproductionStatus.BLOCKED_BY_MISSING_RESOURCE: "因关键数据、模型或算力资源缺失，复现流程被阻塞，判定为**缺少资源**。",
        ReproductionStatus.FULL_EXPERIMENT_COMPLETED: "正式实验已运行完成，但反思审计确认执行流程中存在尚未修复的问题，"
        "因此暂不形成论文复现结论。",
        ReproductionStatus.VERIFIED_REPRODUCTION_GAP: "正式实验的指标差距已经过独立验证和反思审计，"
        "未发现可修复的流程错误；报告该差距，但不声称复现成功。",
        ReproductionStatus.REDUCED_EXPERIMENT_PASSED: "缩小规模实验结果合理，但尚未运行正式实验。",
        ReproductionStatus.SMOKE_TEST_PASSED: "冒烟测试通过，尚未进行缩小规模实验。",
    }
    base = descriptions.get(status, f"最终状态: {status.value}")
    detail_lines = [
        f"- {c.metric}: 论文={c.paper_value}, 复现={c.reproduced_value}, "
        f"差值={c.difference:+.3f}, {'通过' if c.within_tolerance else '未通过'}容差"
        for c in comparisons
    ]
    return base + ("\n\n" + "\n".join(detail_lines) if detail_lines else "")
