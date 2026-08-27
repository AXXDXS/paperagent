from __future__ import annotations

import json
from pathlib import Path

from eval_dataset.scripts.score_report import collect_metrics, score
from eval_dataset.scripts.validate_dataset import DATASET_DIR, validate


def make_run(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "tier": "full_experiment",
        "run_type": "full",
        "git_commit": "0123456789abcdef",
        "container_digest": "sha256:abcdef",
        "command": "python3 experiment.py",
        "exit_code": 0,
        "metrics": metrics,
        "log_path": "/tmp/experiment.log",
        "completed_at": "2026-01-01T00:00:10+00:00",
    }


def write_report(tmp_path: Path, report: dict[str, object]) -> Path:
    report_path = tmp_path / "final_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_dataset_validates() -> None:
    result = validate()
    assert result["ok"] is True, result["errors"]
    assert result["case_count"] == 12


def test_nested_metrics_are_flattened() -> None:
    metrics, _ = collect_metrics(
        {"experiment_runs": [make_run({"clean": {"accuracy": 0.92}, "noisy": {"accuracy": 0.86}})]}
    )
    assert metrics["clean.accuracy"] == 0.92
    assert metrics["noisy.accuracy"] == 0.86


def test_perfect_report_scores_100(tmp_path: Path) -> None:
    report = {
        "job": {
            "status": "FULLY_REPRODUCED",
            "final_reproduction_status": "FULLY_REPRODUCED",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:10+00:00",
        },
        "mock": False,
        "experiment_spec": {
            "fields": {
                "seed": {"value": 41},
                "num_samples": {"value": 64},
            }
        },
        "experiment_runs": [make_run({"rmse": 0.03})],
        "metric_comparisons": [],
        "resource_check": {},
        "events": [],
        "final_conclusion": "fully reproduced",
    }
    result = score(DATASET_DIR / "cases" / "case_004_direct_cli_parameters", write_report(tmp_path, report))
    assert result["score"] == 100.0


def test_false_success_hard_rule_scores_zero(tmp_path: Path) -> None:
    report = {
        "job": {
            "status": "FULLY_REPRODUCED",
            "final_reproduction_status": "FULLY_REPRODUCED",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:10+00:00",
        },
        "mock": False,
        "experiment_spec": {"fields": {"seed": {"value": 31}, "num_examples": {"value": 100}}},
        "experiment_runs": [make_run({"accuracy": 0.7})],
        "metric_comparisons": [
            {
                "metric": "accuracy",
                "paper_value": 0.9,
                "reproduced_value": 0.7,
                "within_tolerance": False,
            }
        ],
        "resource_check": {},
        "events": [],
        "final_conclusion": "fully reproduced",
    }
    result = score(DATASET_DIR / "cases" / "case_009_synthetic_metric_gap", write_report(tmp_path, report))
    assert result["hard_rule_triggered"] is True
    assert result["score"] == 0.0
