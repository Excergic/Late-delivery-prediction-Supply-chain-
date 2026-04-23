"""
Golden parity check runner.

Input CSV must include:
- actual_score
- expected_score

Optional:
- actual_label
- expected_label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.parity import parity_report


def run_parity_check(input_path: str, tolerance: float) -> dict[str, Any]:
    df = pd.read_csv(input_path)
    required_columns = {"actual_score", "expected_score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    report = parity_report(
        actual_scores=np.asarray(df["actual_score"], dtype=float),
        expected_scores=np.asarray(df["expected_score"], dtype=float),
        tolerance=tolerance,
    )

    if {"actual_label", "expected_label"}.issubset(df.columns):
        label_match_rate = float(np.mean(df["actual_label"] == df["expected_label"]))
        report["label_match_rate"] = label_match_rate

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run golden-set parity checks.")
    parser.add_argument("--input", required=True, help="Path to golden comparison CSV.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Absolute score tolerance.")
    parser.add_argument(
        "--output",
        default="data/golden_parity_report.json",
        help="Path to write JSON parity report.",
    )
    args = parser.parse_args()

    report = run_parity_check(args.input, args.tolerance)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print("Golden parity check complete.")
    print(
        f"  within_tolerance={report['within_tolerance']} "
        f"max_abs_delta={report['max_abs_delta']:.8f}"
    )
    return 0 if bool(report["within_tolerance"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
