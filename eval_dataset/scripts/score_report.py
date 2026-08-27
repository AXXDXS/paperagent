#!/usr/bin/env python3
"""Score one ReproAgent final_report.json against one evaluation case."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


FALSE_SUCCESS_STATUSES = {"FULLY_REPRODUCED", "PARTIALLY_REPRODUCED"}


def load_json(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{file_path} must contain a JSON object")
    return value


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, child_prefix))
    elif prefix:
        result[prefix] = value
    return result


def lookup(flattened: dict[str, Any], candidate_names: list[str]) -> tuple[str | None, Any]:
    for name in candidate_names:
        if name in flattened:
            return name, flattened[name]
    for name in candidate_names:
        suffix = f".{name}"
        matches = [(key, value) for key, value in flattened.items() if key.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
    return None, None


def collect_metrics(report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed: dict[str, Any] = {}
    runs = report.get("experiment_runs")
    run_records = [item for item in runs if isinstance(item, dict)] if isinstance(runs, list) else []
    for run in run_records:
        metrics = run.get("metrics")
        if isinstance(metrics, dict):
            observed.update(flatten(metrics))

    comparisons = report.get("metric_comparisons")
    if isinstance(comparisons, list):
        for item in comparisons:
            if not isinstance(item, dict) or not isinstance(item.get("metric"), str):
                continue
            if "reproduced_value" in item:
                observed[item["metric"]] = item["reproduced_value"]
    return observed, run_records


def full_runs(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        run
        for run in run_records
        if str(run.get("tier", "")).lower() == "full_experiment"
        or str(run.get("run_type", "")).lower() in {"full", "full_experiment"}
    ]


def compare_metric(observed: float, spec: dict[str, Any]) -> tuple[bool, float]:
    expected = float(spec["value"])
    tolerance = float(spec["tolerance"])
    absolute_difference = abs(observed - expected)
    if spec["tolerance_type"] == "exact":
        return absolute_difference <= tolerance, absolute_difference
    if spec["tolerance_type"] == "absolute":
        return absolute_difference <= tolerance, absolute_difference
    denominator = abs(expected) if expected != 0 else 1e-12
    relative_difference = absolute_difference / denominator
    return relative_difference <= tolerance, relative_difference


def equal_parameter(observed: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(observed, list) and len(observed) == len(expected) and all(
            equal_parameter(left, right) for left, right in zip(observed, expected)
        )
    left_number = numeric(observed)
    right_number = numeric(expected)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-7, abs_tol=1e-9)
    return observed == expected


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def evidence_checks(
    report: dict[str, Any],
    required: list[str],
    observed_metrics: dict[str, Any],
    successful_runs: list[dict[str, Any]],
) -> dict[str, bool]:
    resource_check = report.get("resource_check") if isinstance(report.get("resource_check"), dict) else {}
    comparisons = report.get("metric_comparisons") if isinstance(report.get("metric_comparisons"), list) else []
    job = report.get("job") if isinstance(report.get("job"), dict) else {}
    events = report.get("events") if isinstance(report.get("events"), list) else []
    final_text = str(report.get("final_conclusion", "")).lower()

    blocking_issues = resource_check.get("blocking_issues")
    statuses = []
    for field_name in ("dataset_status", "model_status", "checkpoint_status"):
        field_value = resource_check.get(field_name)
        if isinstance(field_value, dict):
            statuses.extend(str(value).upper() for value in field_value.values())

    all_checks = {
        "experiment_log": any(bool(run.get("log_path")) for run in successful_runs),
        "metrics_file": bool(observed_metrics),
        "container_digest": any(
            bool(run.get("container_digest")) and str(run.get("container_digest")).lower() != "mock"
            for run in successful_runs
        ),
        "repository_commit": any(bool(run.get("git_commit")) for run in successful_runs),
        "metric_comparison": any(
            isinstance(item, dict) and item.get("within_tolerance") is False for item in comparisons
        ),
        "resource_blocker": bool(blocking_issues)
        or any(status in {"MISSING", "REQUIRES_AUTHORIZATION", "REQUIRES_CREDENTIALS", "RESTRICTED"} for status in statuses),
        "input_error": job.get("status") == "FAILED"
        and not successful_runs
        and (bool(events) or any(word in final_text for word in ("invalid", "pdf", "parse", "输入", "解析"))),
    }
    return {name: all_checks.get(name, False) for name in required}


def score(case_dir: Path, report_path: Path) -> dict[str, Any]:
    task = load_json(case_dir / "task.json")
    gold = load_json(case_dir / "gold.json")
    report = load_json(report_path)

    job = report.get("job") if isinstance(report.get("job"), dict) else {}
    job_status = job.get("status")
    reproduction_status = job.get("final_reproduction_status")
    expected_job_statuses = gold["expected_job_statuses"]
    expected_reproduction_statuses = gold["expected_reproduction_statuses"]

    job_status_ok = job_status in expected_job_statuses
    reproduction_status_ok = reproduction_status in expected_reproduction_statuses
    status_score = (12.5 if job_status_ok else 0.0) + (12.5 if reproduction_status_ok else 0.0)

    hard_rule_triggered = not gold["success_expected"] and reproduction_status in FALSE_SUCCESS_STATUSES

    observed_metrics, run_records = collect_metrics(report)
    candidate_full_runs = full_runs(run_records)
    successful_runs = [run for run in candidate_full_runs if run.get("exit_code") == 0]

    if gold["requires_real_execution"]:
        execution_score = 0.0
        execution_score += 5.0 if report.get("mock") is False else 0.0
        execution_score += 5.0 if candidate_full_runs else 0.0
        execution_score += 7.0 if successful_runs else 0.0
        execution_score += 4.0 if any(bool(run.get("command")) for run in successful_runs) else 0.0
        execution_score += 4.0 if any(bool(run.get("completed_at")) for run in successful_runs) else 0.0
    else:
        execution_score = 25.0 if job_status_ok and not successful_runs else 0.0

    metric_details: dict[str, Any] = {}
    metric_specs = gold["expected_observed_metrics"]
    metric_pass_count = 0
    for metric_name, spec in metric_specs.items():
        matched_key, raw_value = lookup(observed_metrics, [metric_name])
        observed_value = numeric(raw_value)
        if observed_value is None:
            metric_details[metric_name] = {"passed": False, "reason": "missing_or_non_numeric"}
            continue
        passed, measured_difference = compare_metric(observed_value, spec)
        metric_details[metric_name] = {
            "passed": passed,
            "matched_key": matched_key,
            "observed": observed_value,
            "expected": spec["value"],
            "tolerance_type": spec["tolerance_type"],
            "tolerance": spec["tolerance"],
            "measured_difference": measured_difference,
        }
        metric_pass_count += int(passed)
    if metric_specs:
        metric_score = 25.0 * metric_pass_count / len(metric_specs)
    else:
        metric_score = 25.0 if status_score == 25.0 else 0.0

    fields = report.get("experiment_spec", {}).get("fields", {}) if isinstance(report.get("experiment_spec"), dict) else {}
    parameter_values: dict[str, Any] = {}
    if isinstance(fields, dict):
        for field_name, field_record in fields.items():
            if isinstance(field_record, dict) and "value" in field_record:
                parameter_values[field_name] = field_record["value"]
            else:
                parameter_values[field_name] = field_record
    flattened_parameters = flatten(parameter_values)
    parameter_details: dict[str, Any] = {}
    parameter_pass_count = 0
    for parameter_name, spec in gold["key_parameters"].items():
        aliases = spec.get("aliases", [parameter_name]) if isinstance(spec, dict) else [parameter_name]
        expected_value = spec.get("value") if isinstance(spec, dict) else spec
        matched_key, observed_value = lookup(flattened_parameters, list(dict.fromkeys([parameter_name, *aliases])))
        passed = matched_key is not None and equal_parameter(observed_value, expected_value)
        parameter_details[parameter_name] = {
            "passed": passed,
            "matched_key": matched_key,
            "observed": observed_value,
            "expected": expected_value,
        }
        parameter_pass_count += int(passed)
    if gold["key_parameters"]:
        parameter_score = 10.0 * parameter_pass_count / len(gold["key_parameters"])
    else:
        parameter_score = 10.0 if status_score == 25.0 else 0.0

    required_evidence = gold["required_evidence"]
    evidence = evidence_checks(report, required_evidence, observed_metrics, successful_runs)
    if required_evidence:
        evidence_score = 10.0 * sum(evidence.values()) / len(required_evidence)
    else:
        evidence_score = 10.0

    created_at = parse_time(job.get("created_at"))
    updated_at = parse_time(job.get("updated_at"))
    elapsed_seconds = (updated_at - created_at).total_seconds() if created_at and updated_at else None
    budget_limit = task["budget"]["max_runtime_seconds"]
    within_budget = elapsed_seconds is not None and 0 <= elapsed_seconds <= budget_limit
    budget_score = 5.0 if within_budget else 0.0

    breakdown = {
        "terminal_status": round(status_score, 2),
        "real_execution": round(execution_score, 2),
        "observed_metrics": round(metric_score, 2),
        "key_parameters": round(parameter_score, 2),
        "evidence": round(evidence_score, 2),
        "runtime_budget": round(budget_score, 2),
    }
    total = 0.0 if hard_rule_triggered else round(sum(breakdown.values()), 2)
    return {
        "schema_version": 1,
        "case_id": task["case_id"],
        "score": total,
        "max_score": 100.0,
        "hard_rule_triggered": hard_rule_triggered,
        "breakdown": breakdown,
        "status": {
            "observed_job_status": job_status,
            "observed_reproduction_status": reproduction_status,
            "job_status_ok": job_status_ok,
            "reproduction_status_ok": reproduction_status_ok,
        },
        "metrics": metric_details,
        "parameters": parameter_details,
        "evidence": evidence,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "max_runtime_seconds": budget_limit,
            "within_budget": within_budget,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="Case directory containing task.json and gold.json.")
    parser.add_argument("report", type=Path, help="ReproAgent final_report.json.")
    parser.add_argument("--output", type=Path, help="Optional path for the score JSON.")
    args = parser.parse_args()

    try:
        result = score(args.case_dir.resolve(), args.report.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
