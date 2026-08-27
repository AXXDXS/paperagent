"""回归测试：最终报告新增"本次试验的参数设置"章节。

覆盖需求点（红线："最后的报告要有本次试验的参数设置"）：
    - 报告中出现"## 第六部分附加：本次试验的参数设置"章节；
    - 该章节包含三部分：6.1 实验规格采用值、6.2 各次运行完整参数快照、
      6.3 预期指标容差策略；
    - 当没有运行记录时，6.2 应有明确的"无运行记录"占位文案，而不是
      空表——保证报告不会因为数据缺失而让读者误以为该字段被遗漏；
    - 当提供了 ExperimentRun 数据时，6.2 应列出完整的可追溯七元组。
"""

from __future__ import annotations

from repro_agent.domain.enums import ExperimentTier, FieldProvenance, ToleranceType
from repro_agent.domain.experiment import ExperimentRun, ExpectedResult, MetricComparison, ProvenancedField
from repro_agent.observability.report import FinalReportGenerator, ReportInputs


def _base_inputs(job) -> ReportInputs:
    return ReportInputs(job=job)


def test_report_contains_new_parameters_section_with_empty_data(job):
    generator = FinalReportGenerator()
    report = generator.generate(_base_inputs(job))

    assert "## 第六部分附加：本次试验的参数设置" in report
    assert "### 6.1 实验规格采用的参数值" in report
    assert "### 6.2 各次运行的完整参数快照（可追溯七元组）" in report
    assert "### 6.3 预期指标容差策略" in report
    assert "(无运行记录)" in report  # 空运行记录的明确占位文案


def test_report_run_parameters_section_includes_full_traceability_septuple(job):
    generator = FinalReportGenerator()
    run = ExperimentRun(
        experiment_id="exp_main",
        job_id=job.job_id,
        tier=ExperimentTier.FULL_EXPERIMENT,
        run_type="full",
        git_commit="abc123def456",
        container_digest="sha256:deadbeef",
        config_digest="cfg-digest-123",
        dataset_digest="ds-digest-456",
        model_identifier="resnet50:v1",
        seed=42,
        hardware_identifier="gpu:0:A100",
        command="python train.py --config cfg.yaml",
        exit_code=0,
        metrics={"accuracy": 0.93},
    )
    inputs = _base_inputs(job)
    inputs.experiment_runs = [run]

    report = generator.generate(inputs)

    # 6.2 必须包含完整的七元组字段值（不裁剪、不省略），供审计/复算
    for expected_substr in [
        "run_id=",
        "python train.py --config cfg.yaml",
        "abc123def456",
        "sha256:deadbeef",
        "cfg-digest-123",
        "ds-digest-456",
        "resnet50:v1",
        "random_seed: 42",
        "gpu:0:A100",
        "是",  # is_fully_traceable() -> 是
    ]:
        assert expected_substr in report, f"6.2 section missing expected field: {expected_substr}"


def test_report_run_parameters_section_includes_spec_fields_and_tolerances(job):
    generator = FinalReportGenerator()
    inputs = _base_inputs(job)
    inputs.experiment_spec_output = {
        "fields": {
            "learning_rate": ProvenancedField(
                value=0.001,
                provenance=FieldProvenance.PAPER_EXPLICIT,
                source_ref="paper.md#sec4.1",
                confidence=0.95,
            ).to_dict(),
            "batch_size": ProvenancedField(
                value=32,
                provenance=FieldProvenance.CODE_DEFAULT,
                source_ref="configs/train.yaml:12",
                confidence=0.8,
            ).to_dict(),
        },
        "expected_results": {
            "accuracy": ExpectedResult(
                metric="accuracy",
                value=0.95,
                tolerance_type=ToleranceType.ABSOLUTE,
                tolerance=0.01,
                tolerance_basis="论文未报告标准差，按经验取 ±0.01",
            ).to_dict(),
        },
    }

    report = generator.generate(inputs)

    assert "| learning_rate | 0.001 |" in report
    assert "paper.md#sec4.1" in report
    assert "0.95" in report
    assert "| batch_size | 32 |" in report
    assert "configs/train.yaml:12" in report
    # 6.3 容差策略表
    assert "| accuracy | 0.95 | absolute | 0.01 |" in report
    assert "论文未报告标准差" in report


def test_report_section_appears_between_runs_and_comparison_sections(job):
    """顺序断言：新章节应该位于第六部分（运行记录）之后、第七部分（结果对比）之前。"""

    generator = FinalReportGenerator()
    report = generator.generate(_base_inputs(job))

    pos_runs = report.find("## 第六部分：运行记录")
    pos_params = report.find("## 第六部分附加：本次试验的参数设置")
    pos_comparison = report.find("## 第七部分：结果对比")
    assert pos_runs != -1 and pos_params != -1 and pos_comparison != -1
    assert pos_runs < pos_params < pos_comparison
