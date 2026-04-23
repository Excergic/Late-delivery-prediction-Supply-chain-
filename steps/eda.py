"""ZenML EDA step for reproducible dataset diagnostics."""

from __future__ import annotations

import json
from typing import Any, cast

import pandas as pd
from zenml import step

from core.eda import run_eda


@step
def run_eda_step(df: pd.DataFrame, target_col: str = "Late_delivery_risk") -> str:
    """
    Run EDA and return a JSON artifact string for tracking.
    Keeping this in the pipeline prevents notebook-only drift.
    """
    report = cast(dict[str, Any], run_eda(df, target_col=target_col))
    summary = {
        "n_samples": report["target"]["n_samples"],
        "positive_rate": report["target"]["positive_rate"],
        "imbalance": report["imbalance_assessment"]["severity"],
    }
    print(f"EDA summary: {summary}")
    return json.dumps(report, indent=2)
