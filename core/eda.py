"""
Exploratory data analysis helpers for reproducible, pipeline-based diagnostics.
"""

from __future__ import annotations

import pandas as pd


def summarize_target_distribution(df: pd.DataFrame, target_col: str) -> dict[str, float]:
    """Return target balance diagnostics for binary classification."""
    if target_col not in df.columns:
        return {"n_samples": float(len(df)), "positive_rate": 0.0, "negative_rate": 0.0}
    positive_rate = float(df[target_col].mean()) if len(df) else 0.0
    return {
        "n_samples": float(len(df)),
        "positive_rate": positive_rate,
        "negative_rate": 1.0 - positive_rate,
    }


def summarize_feature_distributions(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Basic per-feature summary for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    summary: dict[str, dict[str, float]] = {}
    for col in numeric_df.columns:
        s = numeric_df[col]
        summary[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "p01": float(s.quantile(0.01)),
            "p50": float(s.quantile(0.50)),
            "p99": float(s.quantile(0.99)),
            "missing_rate": float(s.isnull().mean()),
        }
    return summary


def summarize_numeric_correlations(df: pd.DataFrame, top_k: int = 15) -> list[dict[str, float | str]]:
    """Return top absolute pairwise numeric correlations."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return []
    corr = numeric_df.corr(numeric_only=True).abs()
    pairs: list[dict[str, float | str]] = []
    columns = list(corr.columns)
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            pairs.append({"feature_a": col_a, "feature_b": col_b, "abs_corr": float(corr.loc[col_a, col_b])})
    pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
    return pairs[:top_k]


def class_imbalance_recommendation(
    positive_rate: float,
    n_samples: int,
    metric_name: str = "f2",
    model_family: str = "tree",
) -> dict[str, str]:
    """
    Decision helper from the imbalance framework:
    1) minority ratio, 2) metric sensitivity, 3) model type, 4) threshold first.
    """
    minority_ratio = min(positive_rate, 1.0 - positive_rate)
    minority_count = int(round(n_samples * minority_ratio))
    if minority_ratio < 0.05 and minority_count < 1000:
        severity = "high"
    elif minority_ratio < 0.2:
        severity = "moderate"
    else:
        severity = "low"

    recommendation = "Threshold tuning first; no resampling by default."
    if severity == "high" and metric_name.lower() in {"f2", "recall"} and model_family != "tree":
        recommendation = "Tune threshold first, then consider class_weight or SMOTE on train only."

    return {
        "severity": severity,
        "minority_ratio": f"{minority_ratio:.4f}",
        "minority_count": str(minority_count),
        "recommendation": recommendation,
    }


def run_eda(df: pd.DataFrame, target_col: str = "Late_delivery_risk") -> dict[str, object]:
    """Run an EDA bundle suitable for artifact logging."""
    target = summarize_target_distribution(df, target_col)
    return {
        "target": target,
        "feature_distributions": summarize_feature_distributions(df.drop(columns=[target_col], errors="ignore")),
        "top_correlations": summarize_numeric_correlations(df.drop(columns=[target_col], errors="ignore")),
        "imbalance_assessment": class_imbalance_recommendation(
            positive_rate=float(target["positive_rate"]),
            n_samples=int(target["n_samples"]),
            metric_name="f2",
            model_family="tree",
        ),
    }
