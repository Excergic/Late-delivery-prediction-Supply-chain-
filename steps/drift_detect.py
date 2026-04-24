"""
Drift detection step: compares reference vs current data windows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, cast

import pandas as pd
import yaml
from zenml import step

from core.drift import (
    detect_chi2_drift,
    detect_ks_drift,
    detect_psi_drift,
    summarize_drift,
)

logger = logging.getLogger(__name__)

_DEFAULT_DEPLOYMENT_CONFIG_PATH = str(
    Path(__file__).parents[1] / "configs" / "deployment_config.yaml"
)


@step
def drift_detect(
    config_path: str = _DEFAULT_DEPLOYMENT_CONFIG_PATH,
) -> Annotated[dict[str, Any], "drift_report"]:
    """
    Run statistical drift checks and return a report.
    """
    with open(config_path) as f:
        config = cast(dict[str, Any], yaml.safe_load(f))

    reference_path = str(config.get("drift_reference_path", "data/reference_window.csv"))
    current_path = str(config.get("drift_current_path", "data/current_window.csv"))

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    numeric_features = list(config.get("drift_numeric_features", []))
    categorical_features = list(config.get("drift_categorical_features", []))
    psi_features = list(config.get("drift_psi_features", numeric_features))

    ks_results = detect_ks_drift(
        reference_df,
        current_df,
        numeric_features=numeric_features,
        p_value_threshold=float(config.get("drift_ks_p_threshold", 0.05)),
    )
    chi2_results = detect_chi2_drift(
        reference_df,
        current_df,
        categorical_features=categorical_features,
        p_value_threshold=float(config.get("drift_chi2_p_threshold", 0.05)),
    )
    psi_results = detect_psi_drift(
        reference_df,
        current_df,
        psi_features=psi_features,
        moderate_threshold=float(config.get("drift_psi_moderate_threshold", 0.10)),
        significant_threshold=float(config.get("drift_psi_significant_threshold", 0.25)),
    )

    summary = summarize_drift(
        ks_results=ks_results,
        chi2_results=chi2_results,
        psi_results=psi_results,
        min_drifted_features_for_alert=int(config.get("drift_min_features_for_alert", 3)),
    )

    report: dict[str, Any] = {
        "reference_path": reference_path,
        "current_path": current_path,
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "ks_results": ks_results,
        "chi2_results": chi2_results,
        "psi_results": psi_results,
        "summary": {
            "should_alert": summary.should_alert,
            "n_drifted_features": summary.n_drifted_features,
            "drifted_features": summary.drifted_features,
            "reason": summary.reason,
        },
    }

    logger.info("Drift detection complete.")
    logger.info("  Drifted features: %d", summary.n_drifted_features)
    logger.info("  Alert: %s", "YES" if summary.should_alert else "NO")
    if summary.drifted_features:
        logger.info("  Features: %s", summary.drifted_features)
    return report
