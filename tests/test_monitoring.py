"""
Unit tests for monitoring helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.monitoring import (
    build_alerts,
    compute_data_health_metrics,
    compute_performance_metrics_with_labels,
    compute_prediction_health_metrics,
)


def test_compute_data_health_metrics_tracks_null_rates() -> None:
    df = pd.DataFrame(
        {
            "late_delivery_risk_score": [0.1, 0.2, None],
            "predicted_late": [0, 1, 0],
            "order_id": [1, 2, 3],
        }
    )
    metrics = compute_data_health_metrics(df, ["late_delivery_risk_score", "predicted_late"])
    assert metrics["row_count"] == 3
    assert "late_delivery_risk_score" in metrics["null_rates"]
    assert metrics["null_rates"]["late_delivery_risk_score"] > 0.0


def test_compute_prediction_health_metrics_outputs_expected_keys() -> None:
    scores = np.array([0.1, 0.4, 0.8, 0.9], dtype=float)
    labels = (scores >= 0.5).astype(int)
    metrics = compute_prediction_health_metrics(scores, labels)
    assert set(["mean_score", "std_score", "p95_score", "predicted_positive_rate"]).issubset(
        metrics.keys()
    )


def test_compute_performance_metrics_with_labels_returns_f2() -> None:
    y_true = np.array([0, 1, 1, 0, 1], dtype=int)
    y_scores = np.array([0.2, 0.8, 0.6, 0.3, 0.9], dtype=float)
    metrics = compute_performance_metrics_with_labels(y_true, y_scores, threshold=0.5)
    assert "f2" in metrics
    assert 0.0 <= metrics["f2"] <= 1.0


def test_build_alerts_flags_prediction_shift() -> None:
    data_health = {"row_count": 500, "null_rates": {"predicted_late": 0.0}}
    prediction_health = {"predicted_positive_rate": 0.90}
    config = {
        "monitor_min_rows": 100,
        "monitor_null_rate_threshold": 0.20,
        "monitor_baseline_predicted_positive_rate": 0.55,
        "monitor_max_positive_rate_shift": 0.10,
    }
    alerts = build_alerts(
        data_health=data_health,
        prediction_health=prediction_health,
        performance_metrics=None,
        config=config,
    )
    assert any("Predicted positive rate shift exceeds threshold" in alert for alert in alerts)


def test_build_alerts_flags_f2_and_recall_below_threshold() -> None:
    data_health = {"row_count": 500, "null_rates": {}}
    prediction_health = {"predicted_positive_rate": 0.55}
    performance_metrics = {"f2": 0.60, "recall": 0.65}
    config = {
        "monitor_min_rows": 100,
        "monitor_null_rate_threshold": 0.20,
        "monitor_baseline_predicted_positive_rate": 0.55,
        "monitor_max_positive_rate_shift": 0.10,
        "monitor_min_f2": 0.70,
        "monitor_min_recall": 0.75,
    }
    alerts = build_alerts(
        data_health=data_health,
        prediction_health=prediction_health,
        performance_metrics=performance_metrics,
        config=config,
    )
    assert any("F2 below threshold" in alert for alert in alerts)
    assert any("Recall below threshold" in alert for alert in alerts)


def test_build_alerts_no_alerts_when_healthy() -> None:
    data_health = {"row_count": 500, "null_rates": {"predicted_late": 0.0}}
    prediction_health = {"predicted_positive_rate": 0.55}
    performance_metrics = {"f2": 0.80, "recall": 0.85}
    config = {
        "monitor_min_rows": 100,
        "monitor_null_rate_threshold": 0.20,
        "monitor_baseline_predicted_positive_rate": 0.55,
        "monitor_max_positive_rate_shift": 0.10,
        "monitor_min_f2": 0.70,
        "monitor_min_recall": 0.75,
    }
    alerts = build_alerts(
        data_health=data_health,
        prediction_health=prediction_health,
        performance_metrics=performance_metrics,
        config=config,
    )
    assert alerts == []
