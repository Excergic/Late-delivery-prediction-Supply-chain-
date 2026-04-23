"""
Shadow comparison step: score same batch with production and candidate aliases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from zenml import step

from core.deployment import build_shadow_dataframe, summarize_shadow_comparison
from core.preprocessing import drop_columns, extract_date_features, load_features_config


def _resolve_threshold(
    client: MlflowClient,
    model_name: str,
    alias: str,
    default_threshold: float = 0.5,
) -> float:
    """Resolve threshold parameter from the run backing model alias."""
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
        if model_version.run_id is None:
            return default_threshold
        run = client.get_run(model_version.run_id)
        return float(run.data.params.get("optimal_threshold", default_threshold))
    except Exception:
        return default_threshold


def _prepare_features(df: pd.DataFrame, features_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the same non-fitted preprocessing used by inference."""
    order_ids = df["Order Id"].copy() if "Order Id" in df.columns else pd.Series(range(len(df)))
    date_col = features_config.get("date_column")
    if date_col and date_col in df.columns:
        df = extract_date_features(df, str(date_col))

    df = drop_columns(df, cast(list[str], features_config.get("drop_columns", [])))
    target_col = str(features_config.get("target_column", "Late_delivery_risk"))
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    return df, order_ids


@step
def shadow_compare(
    config_path: str = "configs/deployment_config.yaml",
) -> Annotated[dict[str, Any], "shadow_report"]:
    """
    Compare production vs staging model predictions on the same input window.
    """
    with open(config_path) as f:
        config = cast(dict[str, Any], yaml.safe_load(f))

    model_name = str(config.get("model_name", "supply-chain-late-delivery"))
    production_alias = str(config.get("production_alias", "production"))
    candidate_alias = str(config.get("candidate_alias", "staging"))
    input_path = str(config.get("shadow_input_path", "data/DataCoSupplyChainDataset.csv"))
    features_config_path = str(config.get("features_config_path", "configs/features_config.yaml"))
    output_path = str(config.get("shadow_output_path", "data/shadow_comparison.csv"))
    disagreement_alert_threshold = float(config.get("shadow_disagreement_alert_threshold", 0.10))

    raw_df = pd.read_csv(input_path)
    features_config = load_features_config(features_config_path)
    feature_df, order_ids = _prepare_features(raw_df, features_config)

    production_model = mlflow.sklearn.load_model(f"models:/{model_name}@{production_alias}")
    candidate_model = mlflow.sklearn.load_model(f"models:/{model_name}@{candidate_alias}")

    production_scores = np.asarray(production_model.predict_proba(feature_df)[:, 1], dtype=float)
    candidate_scores = np.asarray(candidate_model.predict_proba(feature_df)[:, 1], dtype=float)

    client = MlflowClient()
    threshold = _resolve_threshold(client, model_name, production_alias, default_threshold=0.5)
    summary = summarize_shadow_comparison(production_scores, candidate_scores, threshold)
    summary["should_alert"] = bool(summary["disagreement_rate"] > disagreement_alert_threshold)
    summary["disagreement_alert_threshold"] = disagreement_alert_threshold

    shadow_df = build_shadow_dataframe(order_ids, production_scores, candidate_scores, threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shadow_df.to_csv(output, index=False)

    report: dict[str, Any] = {
        "model_name": model_name,
        "production_alias": production_alias,
        "candidate_alias": candidate_alias,
        "input_path": input_path,
        "output_path": str(output),
        "summary": summary,
    }
    print("Shadow comparison complete.")
    print(f"  disagreement_rate={summary['disagreement_rate']:.4f}")
    print(f"  should_alert={summary['should_alert']}")
    return report
