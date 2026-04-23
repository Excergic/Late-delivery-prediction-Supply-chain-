"""
Monitoring step: evaluate data/prediction health and delayed-label performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import pandas as pd
import yaml
from zenml import step

from core.monitoring import (
    build_alerts,
    compute_data_health_metrics,
    compute_performance_metrics_with_labels,
    compute_prediction_health_metrics,
)


@step
def monitor_model(
    config_path: str = "configs/deployment_config.yaml",
) -> Annotated[dict[str, Any], "monitoring_report"]:
    """
    Run layer-1 and layer-2 monitoring checks.
    """
    with open(config_path) as f:
        config = cast(dict[str, Any], yaml.safe_load(f))

    predictions_path = str(config.get("monitor_predictions_path", "data/predictions.csv"))
    labels_path = str(config.get("monitor_labels_path", "data/labeled_predictions.csv"))
    tracked_columns = cast(list[str], config.get("monitor_tracked_columns", []))
    report_output_path = str(config.get("monitor_output_path", "data/monitoring_report.json"))

    predictions_df = pd.read_csv(predictions_path)
    data_health = compute_data_health_metrics(predictions_df, tracked_columns)

    prediction_scores = np.asarray(predictions_df["late_delivery_risk_score"], dtype=float)
    prediction_labels = np.asarray(predictions_df["predicted_late"], dtype=int)
    prediction_health = compute_prediction_health_metrics(prediction_scores, prediction_labels)

    performance_metrics: dict[str, float] | None = None
    if Path(labels_path).exists():
        labeled_df = pd.read_csv(labels_path)
        if "Late_delivery_risk" in labeled_df.columns and "late_delivery_risk_score" in labeled_df.columns:
            y_true = np.asarray(labeled_df["Late_delivery_risk"], dtype=int)
            y_scores = np.asarray(labeled_df["late_delivery_risk_score"], dtype=float)
            threshold = float(config.get("monitor_decision_threshold", 0.5))
            performance_metrics = compute_performance_metrics_with_labels(
                y_true=y_true,
                y_scores=y_scores,
                threshold=threshold,
            )

    alerts = build_alerts(
        data_health=data_health,
        prediction_health=prediction_health,
        performance_metrics=performance_metrics,
        config=config,
    )

    report: dict[str, Any] = {
        "data_health": data_health,
        "prediction_health": prediction_health,
        "performance_metrics": performance_metrics,
        "alerts": alerts,
        "n_alerts": len(alerts),
    }

    output = Path(report_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print("Monitoring run complete.")
    print(f"  alerts={len(alerts)}")
    if alerts:
        for alert in alerts:
            print(f"  - {alert}")
    return report
