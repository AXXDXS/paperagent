"""最终报告生成器（设计文档 §20 最终报告结构，共十部分）。

报告以 Markdown 形式输出，数据来源全部是已经持久化的领域对象
（Job、Task 事件、ExperimentRun、ReflectionReport、
MetricComparison），不重新调用任何 LLM——报告是"对已发生事实的
结构化汇总"，不是新的生成任务，这样可以保证报告内容与系统实际记录
的证据完全一致（可审计），而不会因为 LLM 生成引入新的不一致或幻觉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from repro_agent.domain.experiment import ExperimentRun, MetricComparison
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.reflection import ReflectionReport
from repro_agent.domain.task import Task
from repro_agent.domain.verification import VerificationRecord


@dataclass
class ReportInputs:
    """组装最终报告所需的全部结构化输入。"""

    job: ReproductionJob
    paper_analysis_output: dict[str, Any] = field(default_factory=dict)
    code_analysis_output: dict[str, Any] = field(default_factory=dict)
    experiment_spec_output: dict[str, Any] = field(default_factory=dict)
    resource_check_output: dict[str, Any] = field(default_factory=dict)
    environment_build_output: dict[str, Any] = field(default_factory=dict)
    experiment_runs: list[ExperimentRun] = field(default_factory=list)
    final_comparisons: list[MetricComparison] = field(default_factory=list)
    failed_tasks: list[Task] = field(default_factory=list)
    reflection_reports: list[ReflectionReport] = field(default_factory=list)
    final_conclusion: str = ""
    verification_records: list[VerificationRecord] = field(default_factory=list)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    is_mock: bool = False


class FinalReportGenerator:
    """按 §20 的十个部分组装 Markdown 最终报告。"""

    def generate(self, inputs: ReportInputs) -> str:
        sections = [
            self._section_1_goal(inputs),
            self._section_2_paper_config(inputs),
            self._section_3_code_pipeline(inputs),
            self._section_4_diffs(inputs),
            self._section_5_resources(inputs),
            self._section_6_runs(inputs),
            # 实验参数设置：红线要求"最终的报告要有本次试验的参数设置"，
            # 单独成节而不是散落在第二部分（论文原始参数）或第六部分
            # （运行记录表格，只有摘要列），把"本次实际执行所采用的完整
            # 参数快照"集中呈现，供审计/复算时不需要再去翻日志拼凑。
            self._section_6b_run_parameters(inputs),
            self._section_7_comparison(inputs),
            self._section_8_errors_and_fixes(inputs),
            self._section_9_reflection_audit(inputs),
            self._section_9b_human_interventions(inputs),
            self._section_10_conclusion(inputs),
        ]
        return "\n\n".join(sections) + "\n"

    # ---- 第一部分：复现目标 ----
    def _section_1_goal(self, inputs: ReportInputs) -> str:
        job = inputs.job
        return (
            "# 复现报告\n\n"
            "## 第一部分：复现目标\n"
            f"- 论文: {job.inputs.paper_path}\n"
            f"- 代码版本: {job.inputs.repository_path}\n"
            f"- 目标实验: {job.inputs.target_experiments}\n"
            f"- 目标指标: {list(inputs.experiment_spec_output.get('expected_results', {}).keys())}\n"
            f"- 复现范围: job_id={job.job_id}"
        )

    # ---- 第二部分：论文配置 ----
    def _section_2_paper_config(self, inputs: ReportInputs) -> str:
        params = inputs.paper_analysis_output.get("extracted_parameters", [])
        lines = ["## 第二部分：论文配置"]
        for p in params[:50]:
            lines.append(
                f"- {p.get('name')} = {p.get('value')} "
                f"(来源: {p.get('provenance')}, 章节: {p.get('section')}, "
                f"推断: {p.get('is_inferred')})"
            )
        if not params:
            lines.append("- (无提取到的参数)")
        return "\n".join(lines)

    # ---- 第三部分：代码流程 ----
    def _section_3_code_pipeline(self, inputs: ReportInputs) -> str:
        code = inputs.code_analysis_output
        return (
            "## 第三部分：代码流程\n"
            f"- 入口: {code.get('entry_points', [])}\n"
            f"- 数据流程: {code.get('data_pipeline_summary', '')}\n"
            f"- 模型流程: {code.get('model_pipeline_summary', '')}\n"
            f"- 训练流程: {code.get('training_pipeline_summary', '')}\n"
            f"- 评测流程: {code.get('evaluation_pipeline_summary', '')}\n"
            f"- 输出位置: {code.get('experiment_output_paths', [])}"
        )

    # ---- 第四部分：论文与代码差异 ----
    def _section_4_diffs(self, inputs: ReportInputs) -> str:
        fields = inputs.experiment_spec_output.get("fields", {})
        lines = ["## 第四部分：论文与代码差异", "", "| 字段 | 采用值 | 来源 | 是否存在冲突 |", "|---|---|---|---|"]
        for name, meta in fields.items():
            has_conflict = bool(meta.get("conflicting_values"))
            lines.append(f"| {name} | {meta.get('value')} | {meta.get('provenance')} | {has_conflict} |")
        if not fields:
            lines.append("| (无字段记录) | | | |")
        return "\n".join(lines)

    # ---- 第五部分：资源状态 ----
    def _section_5_resources(self, inputs: ReportInputs) -> str:
        res = inputs.resource_check_output
        return (
            "## 第五部分：资源状态\n"
            f"- 数据: {res.get('dataset_status', {})}\n"
            f"- 模型: {res.get('model_status', {})}\n"
            f"- GPU: {res.get('gpu_info', {})}\n"
            f"- 环境: {inputs.environment_build_output.get('import_test_passed')}\n"
            f"- 缺失资源: {res.get('blocking_issues', [])}"
        )

    # ---- 第六部分：运行记录 ----
    def _section_6_runs(self, inputs: ReportInputs) -> str:
        lines = ["## 第六部分：运行记录", "", "| Tier | 命令 | 退出码 | git_commit | config_digest |", "|---|---|---|---|---|"]
        for run in inputs.experiment_runs:
            lines.append(
                f"| {run.tier.value} | `{run.command}` | {run.exit_code} | "
                f"{run.git_commit[:8]} | {run.config_digest[:8]} |"
            )
        if not inputs.experiment_runs:
            lines.append("| (无运行记录) | | | | |")
        return "\n".join(lines)

    # ---- 第六部分附加：本次试验的完整参数设置 ----
    #
    # 红线要求原文："最后的报告要有本次试验的参数设置"。与第二部分
    # （论文原始声明的参数，可能包含论文自身的多个候选值/未消解冲突）
    # 和第六部分运行记录表（只列可追溯性摘要列）不同，本节汇总的是
    # "本次复现实际采用并执行"的最终参数快照，三层来源清晰区分：
    #   1. 实验规格采用值（§9.4 ExperimentSpec.fields，经过冲突消解后
    #      main agent 最终确定采用的字段值 + provenance）；
    #   2. 每次实际运行绑定的可追溯七元组（§10.5：git_commit/
    #      container_digest/config_digest/dataset_digest/
    #      model_identifier/seed/hardware_identifier）+ 完整执行命令；
    #   3. 预期指标的容差策略（用于解释第七部分"通过/未通过"判定依据）。
    def _section_6b_run_parameters(self, inputs: ReportInputs) -> str:
        lines = ["## 第六部分附加：本次试验的参数设置"]

        fields = inputs.experiment_spec_output.get("fields", {})
        lines.append("\n### 6.1 实验规格采用的参数值")
        lines.append("")
        lines.append("| 参数 | 采用值 | 来源 | 来源引用 | 置信度 |")
        lines.append("|---|---|---|---|---:|")
        if fields:
            for name, meta in fields.items():
                lines.append(
                    f"| {name} | {meta.get('value')} | {meta.get('provenance')} | "
                    f"{meta.get('source_ref', '')} | {meta.get('confidence', 1.0)} |"
                )
        else:
            lines.append("| (无字段记录) | | | | |")

        lines.append("\n### 6.2 各次运行的完整参数快照（可追溯七元组）")
        lines.append("")
        if inputs.experiment_runs:
            for run in inputs.experiment_runs:
                lines.append(f"#### run_id={run.run_id} (tier={run.tier.value}, run_type={run.run_type})")
                lines.append(f"- 完整命令: `{run.command}`")
                lines.append(f"- git_commit: {run.git_commit or '(未记录)'}")
                lines.append(f"- container_digest: {run.container_digest or '(未记录)'}")
                lines.append(f"- config_digest: {run.config_digest or '(未记录)'}")
                lines.append(f"- dataset_digest: {run.dataset_digest or '(未记录)'}")
                lines.append(f"- model_identifier: {run.model_identifier or '(未记录)'}")
                lines.append(f"- random_seed: {run.seed if run.seed is not None else '(未记录)'}")
                lines.append(f"- hardware_identifier: {run.hardware_identifier or '(未记录)'}")
                lines.append(
                    f"- 是否满足 §10.5 正式实验完整可追溯性要求: "
                    f"{'是' if run.is_fully_traceable() else '否（非正式实验或存在缺失字段）'}"
                )
        else:
            lines.append("(无运行记录)")

        expected_results = inputs.experiment_spec_output.get("expected_results", {})
        lines.append("\n### 6.3 预期指标容差策略")
        lines.append("")
        lines.append("| 指标 | 论文值 | 容差类型 | 容差 | 容差依据 |")
        lines.append("|---|---:|---|---:|---|")
        if expected_results:
            for name, meta in expected_results.items():
                lines.append(
                    f"| {name} | {meta.get('value')} | {meta.get('tolerance_type')} | "
                    f"{meta.get('tolerance')} | {meta.get('tolerance_basis', '')} |"
                )
        else:
            lines.append("| (无预期指标记录) | | | | |")

        return "\n".join(lines)

    # ---- 第七部分：结果对比 ----
    def _section_7_comparison(self, inputs: ReportInputs) -> str:
        lines = [
            "## 第七部分：结果对比",
            "",
            "| 指标 | 论文结果 | 复现结果 | 差值 | 容差 | 状态 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for c in inputs.final_comparisons:
            status = "通过" if c.within_tolerance else "未通过"
            lines.append(
                f"| {c.metric} | {c.paper_value:.2f} | {c.reproduced_value:.2f} | "
                f"{c.difference:+.2f} | {c.tolerance:.2f} | {status} |"
            )
        if not inputs.final_comparisons:
            lines.append("| (无对比数据) | | | | | |")
        return "\n".join(lines)

    # ---- 第八部分：错误和修复 ----
    def _section_8_errors_and_fixes(self, inputs: ReportInputs) -> str:
        lines = ["## 第八部分：错误和修复"]
        for task in inputs.failed_tasks:
            fr = task.failure_report
            if fr is None:
                continue
            lines.append(
                f"- 任务 {task.task_id}: {fr.failure_type.value} —— {fr.error_message[:200]}\n"
                f"  - 推荐动作: {fr.recommended_action}\n"
                f"  - 重试次数: {task.attempt}"
            )
        if len(lines) == 1:
            lines.append("- (无失败任务记录)")
        return "\n".join(lines)

    # ---- 第九部分：反思审计 ----
    def _section_9_reflection_audit(self, inputs: ReportInputs) -> str:
        lines = ["## 第九部分：反思审计"]
        for report in inputs.reflection_reports:
            lines.append(f"### 反思轮次 {report.round} ({report.reflection_id})")
            lines.append(
                f"- 触发指标: {[m.metric for m in report.trigger_metrics]}"
            )
            for h in report.sorted_hypotheses():
                lines.append(f"  - 假设 [{h.category}] {h.description} (优先级 {h.priority})")
            lines.append(f"- 审计结果: {report.audit_result.value if report.audit_result else '进行中'}")
            lines.append(f"- 确认问题: {report.confirmed_issue or '(未确认具体问题)'}")
            lines.append(
                f"- 建议重跑范围: "
                f"{report.recommended_rerun_scope.value if report.recommended_rerun_scope else '(无)'}"
            )
        if len(lines) == 1:
            lines.append("- (未触发反思闭环)")
        return "\n".join(lines)

    def _section_9b_human_interventions(self, inputs: ReportInputs) -> str:
        lines = [
            "## 第九部分附加：人工介入审计",
            "",
            "| 请求 | 任务 | 类型 | 状态 | 回答者 | 回答字段 |",
            "|---|---|---|---|---|---|",
        ]
        for request in inputs.interventions:
            lines.append(
                f"| {request.get('request_id')} | {request.get('task_id') or '-'} | "
                f"{request.get('kind')} | {request.get('status')} | "
                f"{request.get('responded_by') or '-'} | "
                f"{request.get('response_fields', [])} |"
            )
        if not inputs.interventions:
            lines.append("| (无人工介入) | | | | | |")
        return "\n".join(lines)

    # ---- 第十部分：最终结论 ----
    def _section_10_conclusion(self, inputs: ReportInputs) -> str:
        mode_warning = (
            "\n\n> 注意：本报告来自 mock 执行，仅用于验证 Agent 编排流程，"
            "不能作为论文真实复现结论。"
            if inputs.is_mock
            else ""
        )
        return (
            "## 第十部分：最终结论\n"
            f"{inputs.final_conclusion or '(结论待生成)'}\n\n"
            f"- 最终复现状态: "
            f"{inputs.job.final_reproduction_status.value if inputs.job.final_reproduction_status else '未确定'}"
            f"{mode_warning}"
        )
