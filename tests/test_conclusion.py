from __future__ import annotations

from repro_agent.domain.enums import ReproductionStatus
from repro_agent.observability.conclusion import (
    ConclusionInputs,
    determine_final_status,
    render_conclusion_text,
)


def _inputs(**overrides) -> ConclusionInputs:
    values = {
        "environment_ready": True,
        "pipeline_executable": False,
        "smoke_test_passed": False,
        "reduced_experiment_passed": False,
        "full_experiment_completed": False,
        "blocked_by_missing_resource": False,
        "comparisons": [],
    }
    values.update(overrides)
    return ConclusionInputs(**values)


def test_environment_only_is_not_mislabeled_not_reproduced() -> None:
    assert determine_final_status(_inputs()) == ReproductionStatus.ENVIRONMENT_READY


def test_executable_pipeline_before_smoke_has_distinct_status() -> None:
    assert determine_final_status(
        _inputs(pipeline_executable=True)
    ) == ReproductionStatus.PIPELINE_EXECUTABLE


def test_verified_gap_has_explicit_honest_description() -> None:
    text = render_conclusion_text(ReproductionStatus.VERIFIED_REPRODUCTION_GAP, [])

    assert "审计" in text
    assert "差距" in text
