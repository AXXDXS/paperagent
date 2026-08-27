#!/usr/bin/env python3
"""Validate the small ReproAgent evaluation dataset using only the stdlib."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parents[1]
REQUIRED_TASK_KEYS = {
    "schema_version",
    "case_id",
    "title",
    "category",
    "source_kind",
    "paper_path",
    "repository_path",
    "target_experiment",
    "run",
    "inputs",
    "resources",
    "budget",
}
REQUIRED_GOLD_KEYS = {
    "schema_version",
    "case_id",
    "success_expected",
    "requires_real_execution",
    "expected_job_statuses",
    "expected_reproduction_statuses",
    "expected_observed_metrics",
    "key_parameters",
    "required_evidence",
}
EXPECTED_CATEGORY_COUNTS = {
    "direct": 4,
    "repair": 3,
    "gap": 2,
    "blocked": 2,
    "invalid_input": 1,
}


def load_json(file_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {file_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{file_path} must contain a JSON object")
    return value


def add_missing_keys(errors: list[str], label: str, record: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - record.keys())
    if missing:
        errors.append(f"{label}: missing required keys: {', '.join(missing)}")


def validate_metric_specs(errors: list[str], case_id: str, gold: dict[str, Any]) -> None:
    metrics = gold.get("expected_observed_metrics")
    if not isinstance(metrics, dict):
        errors.append(f"{case_id}: expected_observed_metrics must be an object")
        return
    for metric_name, spec in metrics.items():
        if not isinstance(spec, dict):
            errors.append(f"{case_id}: metric {metric_name} must have an object specification")
            continue
        if set(("value", "tolerance_type", "tolerance")) - spec.keys():
            errors.append(f"{case_id}: metric {metric_name} is missing value/tolerance_type/tolerance")
            continue
        if not isinstance(spec["value"], (int, float)) or isinstance(spec["value"], bool):
            errors.append(f"{case_id}: metric {metric_name}.value must be numeric")
        if spec["tolerance_type"] not in {"absolute", "relative", "exact"}:
            errors.append(f"{case_id}: metric {metric_name} has an invalid tolerance_type")
        if not isinstance(spec["tolerance"], (int, float)) or spec["tolerance"] < 0:
            errors.append(f"{case_id}: metric {metric_name}.tolerance must be non-negative")


def validate() -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        dataset = load_json(DATASET_DIR / "dataset.json")
        splits = load_json(DATASET_DIR / "splits.json")
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "case_count": 0}

    base_value = dataset.get("path_base")
    if not isinstance(base_value, str):
        errors.append("dataset.json: path_base must be a string")
        workspace_root = DATASET_DIR
    else:
        workspace_root = (DATASET_DIR / base_value).resolve()

    case_entries = dataset.get("cases")
    if not isinstance(case_entries, list):
        return {
            "ok": False,
            "errors": errors + ["dataset.json: cases must be an array"],
            "warnings": warnings,
            "case_count": 0,
        }

    seen_ids: set[str] = set()
    categories: Counter[str] = Counter()
    for entry in case_entries:
        if not isinstance(entry, dict):
            errors.append("dataset.json: each case entry must be an object")
            continue
        case_id = entry.get("case_id")
        if not isinstance(case_id, str):
            errors.append("dataset.json: case_id must be a string")
            continue
        if case_id in seen_ids:
            errors.append(f"dataset.json: duplicate case_id {case_id}")
        seen_ids.add(case_id)
        categories[str(entry.get("category"))] += 1

        case_path_value = entry.get("case_path")
        if not isinstance(case_path_value, str):
            errors.append(f"{case_id}: case_path must be a string")
            continue
        case_dir = DATASET_DIR / case_path_value
        task_path = case_dir / "task.json"
        gold_path = case_dir / "gold.json"
        if not task_path.is_file() or not gold_path.is_file():
            errors.append(f"{case_id}: task.json and gold.json are both required")
            continue

        try:
            task = load_json(task_path)
            gold = load_json(gold_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        add_missing_keys(errors, f"{case_id}/task.json", task, REQUIRED_TASK_KEYS)
        add_missing_keys(errors, f"{case_id}/gold.json", gold, REQUIRED_GOLD_KEYS)
        for label, value in (("task", task.get("case_id")), ("gold", gold.get("case_id"))):
            if value != case_id:
                errors.append(f"{case_id}: {label} case_id is {value!r}")
        for field_name in ("category", "source_kind"):
            if task.get(field_name) != entry.get(field_name):
                errors.append(f"{case_id}: {field_name} differs between dataset.json and task.json")
        if task.get("schema_version") != 1 or gold.get("schema_version") != 1:
            errors.append(f"{case_id}: only schema_version 1 is supported")

        paper_value = task.get("paper_path")
        repository_value = task.get("repository_path")
        paper_path = workspace_root / paper_value if isinstance(paper_value, str) else workspace_root / "__invalid__"
        repository_path = (
            workspace_root / repository_value
            if isinstance(repository_value, str)
            else workspace_root / "__invalid__"
        )
        if not paper_path.is_file():
            errors.append(f"{case_id}: paper_path does not exist: {paper_value}")
        if not repository_path.is_dir():
            errors.append(f"{case_id}: repository_path is not a directory: {repository_value}")

        run = task.get("run")
        if not isinstance(run, dict) or not isinstance(run.get("command"), list) or not run.get("command"):
            errors.append(f"{case_id}: run.command must be a non-empty array")
        elif not all(isinstance(token, str) and token for token in run["command"]):
            errors.append(f"{case_id}: every run.command token must be a non-empty string")

        inputs = task.get("inputs")
        missing_paths = set(task.get("intentionally_missing_paths", []))
        declared_input_paths: set[str] = set()
        if not isinstance(inputs, dict):
            errors.append(f"{case_id}: inputs must be an object")
        else:
            for input_kind in ("dataset_paths", "model_paths", "checkpoint_paths"):
                values = inputs.get(input_kind)
                if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                    errors.append(f"{case_id}: inputs.{input_kind} must be an array of strings")
                    continue
                declared_input_paths.update(values)
                for relative_path in values:
                    resolved = workspace_root / relative_path
                    if relative_path in missing_paths:
                        if resolved.exists():
                            errors.append(f"{case_id}: intentionally missing path unexpectedly exists: {relative_path}")
                    elif not resolved.exists():
                        errors.append(f"{case_id}: required input path does not exist: {relative_path}")
        undeclared_missing = missing_paths - declared_input_paths
        if undeclared_missing:
            errors.append(
                f"{case_id}: intentionally_missing_paths are not declared inputs: "
                + ", ".join(sorted(undeclared_missing))
            )

        max_runtime = task.get("budget", {}).get("max_runtime_seconds") if isinstance(task.get("budget"), dict) else None
        if not isinstance(max_runtime, int) or max_runtime <= 0:
            errors.append(f"{case_id}: budget.max_runtime_seconds must be a positive integer")

        validate_metric_specs(errors, case_id, gold)
        if not isinstance(gold.get("expected_job_statuses"), list) or not gold.get("expected_job_statuses"):
            errors.append(f"{case_id}: expected_job_statuses must be a non-empty array")
        if not isinstance(gold.get("expected_reproduction_statuses"), list) or not gold.get(
            "expected_reproduction_statuses"
        ):
            errors.append(f"{case_id}: expected_reproduction_statuses must be a non-empty array")

        if task.get("category") == "invalid_input" and paper_path.is_file():
            if paper_path.read_bytes()[:5] == b"%PDF-":
                errors.append(f"{case_id}: invalid_input fixture unexpectedly has a valid PDF header")
        if task.get("category") == "blocked" and not missing_paths:
            errors.append(f"{case_id}: blocked case must declare at least one intentionally missing path")

    if dict(categories) != EXPECTED_CATEGORY_COUNTS:
        errors.append(f"category counts are {dict(categories)}, expected {EXPECTED_CATEGORY_COUNTS}")

    split_ids: list[str] = []
    for split_name in ("development", "evaluation"):
        values = splits.get(split_name)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            errors.append(f"splits.json: {split_name} must be an array of strings")
            continue
        split_ids.extend(values)
    duplicate_split_ids = [item for item, count in Counter(split_ids).items() if count > 1]
    if duplicate_split_ids:
        errors.append("splits.json: duplicate case ids: " + ", ".join(sorted(duplicate_split_ids)))
    if set(split_ids) != seen_ids:
        errors.append(
            "splits.json must cover every case exactly once; difference: "
            + ", ".join(sorted(set(split_ids) ^ seen_ids))
        )

    case_directories = {item.name for item in (DATASET_DIR / "cases").iterdir() if item.is_dir()}
    unindexed = sorted(case_directories - seen_ids)
    if unindexed:
        warnings.append("unindexed case directories: " + ", ".join(unindexed))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "case_count": len(seen_ids),
        "category_counts": dict(sorted(categories.items())),
        "workspace_root": str(workspace_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    result = validate()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        state = "PASS" if result["ok"] else "FAIL"
        print(f"[{state}] {result['case_count']} cases")
        if result.get("category_counts"):
            print("categories:", json.dumps(result["category_counts"], ensure_ascii=False, sort_keys=True))
        for warning in result["warnings"]:
            print(f"warning: {warning}")
        for error in result["errors"]:
            print(f"error: {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
