"""
Drift detection utilities for batch monitoring.

Implements:
- KS test for numeric features
- Chi-squared test for categorical features
- PSI for numeric features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2_contingency, ks_2samp


@dataclass
class DriftSummary:
    """Aggregated drift decision for alerting."""

    should_alert: bool
    n_drifted_features: int
    drifted_features: list[str]
    reason: str


def _clean_numeric(series: pd.Series) -> NDArray[np.float64]:
    """Convert to numeric array and drop NaNs/infs."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return np.asarray(values.to_numpy(dtype=float), dtype=np.float64)


def compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for numeric distributions.

    Threshold convention:
    - < 0.10: stable
    - 0.10 - 0.25: moderate drift
    - > 0.25: significant drift
    """
    ref = _clean_numeric(reference)
    cur = _clean_numeric(current)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(ref, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_ratio = ref_counts / max(ref_counts.sum(), 1)
    cur_ratio = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-6
    ref_ratio = np.clip(ref_ratio, eps, 1.0)
    cur_ratio = np.clip(cur_ratio, eps, 1.0)

    psi = np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio))
    return float(psi)


def detect_ks_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    p_value_threshold: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Detect numeric drift using KS test."""
    results: dict[str, dict[str, Any]] = {}
    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref = _clean_numeric(reference_df[feature])
        cur = _clean_numeric(current_df[feature])
        if len(ref) < 2 or len(cur) < 2:
            continue
        stat, p_value = ks_2samp(ref, cur)
        results[feature] = {
            "test": "ks",
            "statistic": float(stat),
            "p_value": float(p_value),
            "drifted": bool(p_value < p_value_threshold),
        }
    return results


def detect_chi2_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    categorical_features: list[str],
    p_value_threshold: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Detect categorical drift using Chi-squared test."""
    results: dict[str, dict[str, Any]] = {}
    for feature in categorical_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref_vals = reference_df[feature].fillna("MISSING").astype(str)
        cur_vals = current_df[feature].fillna("MISSING").astype(str)
        all_categories = sorted(set(ref_vals.unique()).union(set(cur_vals.unique())))
        if len(all_categories) < 2:
            continue

        ref_counts = ref_vals.value_counts().reindex(all_categories, fill_value=0)
        cur_counts = cur_vals.value_counts().reindex(all_categories, fill_value=0)
        contingency = np.vstack([ref_counts.values, cur_counts.values])
        _, p_value, _, _ = chi2_contingency(contingency)

        results[feature] = {
            "test": "chi2",
            "p_value": float(p_value),
            "drifted": bool(p_value < p_value_threshold),
        }
    return results


def detect_psi_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_features: list[str],
    moderate_threshold: float = 0.10,
    significant_threshold: float = 0.25,
) -> dict[str, dict[str, Any]]:
    """Detect distribution drift with PSI."""
    results: dict[str, dict[str, Any]] = {}
    for feature in psi_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        psi_value = compute_psi(reference_df[feature], current_df[feature])
        if psi_value > significant_threshold:
            severity = "significant"
        elif psi_value > moderate_threshold:
            severity = "moderate"
        else:
            severity = "stable"
        results[feature] = {
            "test": "psi",
            "psi": float(psi_value),
            "severity": severity,
            "drifted": bool(psi_value > moderate_threshold),
        }
    return results


def summarize_drift(
    ks_results: dict[str, dict[str, Any]],
    chi2_results: dict[str, dict[str, Any]],
    psi_results: dict[str, dict[str, Any]],
    min_drifted_features_for_alert: int = 3,
) -> DriftSummary:
    """Combine all tests into one alert decision."""
    drifted: set[str] = set()
    for result_map in (ks_results, chi2_results, psi_results):
        for feature, values in result_map.items():
            if bool(values.get("drifted", False)):
                drifted.add(feature)

    drifted_features = sorted(drifted)
    should_alert = len(drifted_features) >= min_drifted_features_for_alert
    reason = (
        f"{len(drifted_features)} drifted features >= "
        f"{min_drifted_features_for_alert} alert threshold"
        if should_alert
        else f"{len(drifted_features)} drifted features below alert threshold"
    )
    return DriftSummary(
        should_alert=should_alert,
        n_drifted_features=len(drifted_features),
        drifted_features=drifted_features,
        reason=reason,
    )
