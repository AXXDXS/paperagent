from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    targets = [1.0, 2.0, 3.0, 4.0]
    predictions = [1.01, 1.98, 3.03, 3.98]
    mae = sum(abs(left - right) for left, right in zip(targets, predictions)) / len(targets)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "mean_absolute_error": mae,
                "seed": args.seed,
                "num_evaluation_samples": len(targets),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
