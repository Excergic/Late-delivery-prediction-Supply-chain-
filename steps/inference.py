"""
Batch inference step: load registered model bundle, score new orders, write predictions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from zenml import step

from core.preprocessing import drop_columns, extract_date_features, load_features_config


def _resolve_threshold(
    client: MlflowClient,
    model_name: str,
    alias: str,
    default_threshold: float = 0.5,
) -> float:
    """Read the optimal threshold logged with the registered model version run."""
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
        if model_version.run_id is None:
            return default_threshold
        run = client.get_run(model_version.run_id)
        return float(run.data.params.get("optimal_threshold", default_threshold))
    except Exception:
        return default_threshold


@step
def run_inference(
    input_path: str,
    output_path: str = "data/predictions.csv",
    features_config_path: str = "configs/features_config.yaml",
    model_name: str = "supply-chain-late-delivery",
    model_alias: str = "staging",
) -> pd.DataFrame:
    """
    Score a batch of new orders with the registered model bundle.
    """
    df = pd.read_csv(input_path)
    features_config = load_features_config(features_config_path)

    order_ids = df["Order Id"].copy() if "Order Id" in df.columns else pd.Series(range(len(df)))

    date_col = features_config.get("date_column")
    if date_col and date_col in df.columns:
        df = extract_date_features(df, date_col)

    df = drop_columns(df, features_config.get("drop_columns", []))
    target_col = features_config.get("target_column", "Late_delivery_risk")
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    model_uri = f"models:/{model_name}@{model_alias}"
    model_bundle = mlflow.sklearn.load_model(model_uri)

    client = MlflowClient()
    threshold = _resolve_threshold(client, model_name, model_alias, default_threshold=0.5)

    if hasattr(model_bundle, "predict_proba"):
        scores = model_bundle.predict_proba(df)[:, 1]
    else:
        scores = model_bundle.predict(df).astype(float)

    predictions = pd.DataFrame(
        {
            "order_id": order_ids.values,
            "late_delivery_risk_score": scores,
            "predicted_late": (scores >= threshold).astype(int),
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} predictions to {output}")
    return predictions
