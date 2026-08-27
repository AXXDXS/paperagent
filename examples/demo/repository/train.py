"""Deterministic workload used by the one-command ReproAgent demo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


FEATURES = [
    0.05, 0.12, 0.18, 0.24, 0.31, 0.38, 0.43, 0.49, 0.52, 0.58,
    0.61, 0.66, 0.71, 0.76, 0.81, 0.86, 0.90, 0.93, 0.96, 0.99,
]
LABELS = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]


def predict(features: list[float]) -> list[int]:
    return [int(value >= 0.5) for value in features]


def accuracy(predictions: list[int], labels: list[int]) -> float:
    correct = sum(int(prediction == label) for prediction, label in zip(predictions, labels))
    return correct / len(labels)


def run(tier: str, output_dir: Path) -> dict[str, float]:
    limit = {
        "smoke_test": 4,
        "reduced_experiment": 10,
        "full_experiment": len(FEATURES),
    }.get(tier, len(FEATURES))
    labels = LABELS[:limit]
    predictions = predict(FEATURES[:limit])
    metrics = {"accuracy": accuracy(predictions, labels)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions), encoding="utf-8"
    )
    (output_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=("smoke_test", "reduced_experiment", "full_experiment"),
        default="full_experiment",
    )
    args = parser.parse_args()
    output_dir = Path(os.environ.get("REPRO_AGENT_OUTPUT_DIR", "output"))
    metrics = run(args.tier, output_dir)
    print(json.dumps({"tier": args.tier, "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
