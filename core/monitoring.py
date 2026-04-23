"""
Production monitoring helpers for batch model operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.evaluation import compute_metrics


def compute_data_health_metrics(
    df: pd.DataFrame,
    tracked_columns: list[str],
) -> dict[str, Any]:
    """Compute null-rate and volume metrics for layer-1 monitoring."""
    null_rates: dict[str, float] = {}
    for col in tracked_columns:
        if col in df.columns:
            null_rates[col] = float(df[col].isnull().mean())

    return {
        "row_count": int(len(df)),
        "null_rates": null_rates,
        "overall_null_rate": float(df.isnull().mean().mean()) if len(df.columns) else 0.0,
    }


def compute_prediction_health_metrics(
    prediction_scores: np.ndarray,
    prediction_labels: np.ndarray,
) -> dict[str, float]:
    """Compute fast label-free prediction health metrics."""
    if len(prediction_scores) == 0:
        return {
            "mean_score": 0.0,
            "std_score": 0.0,
            "p95_score": 0.0,
            "predicted_positive_rate": 0.0,
        }
    return {
        "mean_score": float(np.mean(prediction_scores)),
        "std_score": float(np.std(prediction_scores)),
        "p95_score": float(np.percentile(prediction_scores, 95)),
        "predicted_positive_rate": float(np.mean(prediction_labels)),
    }


def compute_performance_metrics_with_labels(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute delayed-label performance metrics."""
    return compute_metrics(y_true, y_scores, threshold=threshold)


def build_alerts(
    data_health: dict[str, Any],
    prediction_health: dict[str, float],
    performance_metrics: dict[str, float] | None,
    config: dict[str, Any],
) -> list[str]:
    """Generate actionable alerts from configured thresholds."""
    alerts: list[str] = []

    min_rows = int(config.get("monitor_min_rows", 100))
    if int(data_health["row_count"]) < min_rows:
        alerts.append(
            f"Row count {data_health['row_count']} below minimum threshold {min_rows}"
        )

    null_rate_threshold = float(config.get("monitor_null_rate_threshold", 0.20))
    for col, rate in data_health.get("null_rates", {}).items():
        if float(rate) > null_rate_threshold:
            alerts.append(f"Null rate for '{col}' is {rate:.2%} > {null_rate_threshold:.2%}")

    predicted_positive_rate = float(prediction_health.get("predicted_positive_rate", 0.0))
    max_positive_rate_shift = float(config.get("monitor_max_positive_rate_shift", 0.10))
    baseline_positive_rate = float(config.get("monitor_baseline_predicted_positive_rate", 0.50))
    if abs(predicted_positive_rate - baseline_positive_rate) > max_positive_rate_shift:
        alerts.append(
            "Predicted positive rate shift exceeds threshold: "
            f"current={predicted_positive_rate:.3f}, baseline={baseline_positive_rate:.3f}, "
            f"max_shift={max_positive_rate_shift:.3f}"
        )

    if performance_metrics is not None:
        min_f2 = float(config.get("monitor_min_f2", 0.70))
        min_recall = float(config.get("monitor_min_recall", 0.75))
        if float(performance_metrics.get("f2", 0.0)) < min_f2:
            alerts.append(
                f"F2 below threshold: {performance_metrics.get('f2', 0.0):.4f} < {min_f2:.4f}"
            )
        if float(performance_metrics.get("recall", 0.0)) < min_recall:
            alerts.append(
                "Recall below threshold: "
                f"{performance_metrics.get('recall', 0.0):.4f} < {min_recall:.4f}"
            )

    return alerts
