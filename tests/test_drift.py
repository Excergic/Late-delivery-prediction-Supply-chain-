"""
Unit tests for core/drift.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.drift import (
    compute_psi,
    detect_chi2_drift,
    detect_ks_drift,
    detect_psi_drift,
    summarize_drift,
)


def _make_reference_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Days for shipment (scheduled)": rng.normal(3.0, 0.8, size=n),
            "Sales": rng.normal(200.0, 50.0, size=n),
            "Shipping Mode": rng.choice(["Standard Class", "Second Class"], size=n, p=[0.8, 0.2]),
        }
    )


def _make_current_df_with_shift(n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Days for shipment (scheduled)": rng.normal(5.0, 0.9, size=n),
            "Sales": rng.normal(320.0, 55.0, size=n),
            "Shipping Mode": rng.choice(["Standard Class", "Second Class"], size=n, p=[0.45, 0.55]),
        }
    )


def test_compute_psi_near_zero_on_similar_distributions() -> None:
    rng = np.random.default_rng(42)
    a = pd.Series(rng.normal(0.0, 1.0, size=2000))
    b = pd.Series(rng.normal(0.0, 1.0, size=2000))
    psi = compute_psi(a, b)
    assert psi < 0.10


def test_compute_psi_detects_large_shift() -> None:
    rng = np.random.default_rng(42)
    a = pd.Series(rng.normal(0.0, 1.0, size=2000))
    b = pd.Series(rng.normal(2.5, 1.0, size=2000))
    psi = compute_psi(a, b)
    assert psi > 0.25


def test_detect_ks_drift_flags_shifted_numeric_feature() -> None:
    reference_df = _make_reference_df()
    current_df = _make_current_df_with_shift()
    results = detect_ks_drift(
        reference_df,
        current_df,
        numeric_features=["Days for shipment (scheduled)"],
        p_value_threshold=0.05,
    )
    assert results["Days for shipment (scheduled)"]["drifted"] is True


def test_detect_chi2_drift_flags_shifted_categorical_feature() -> None:
    reference_df = _make_reference_df()
    current_df = _make_current_df_with_shift()
    results = detect_chi2_drift(
        reference_df,
        current_df,
        categorical_features=["Shipping Mode"],
        p_value_threshold=0.05,
    )
    assert results["Shipping Mode"]["drifted"] is True


def test_summarize_drift_respects_alert_threshold() -> None:
    ks_results = {"f1": {"drifted": True}, "f2": {"drifted": True}}
    chi2_results = {"f3": {"drifted": True}}
    psi_results = {"f4": {"drifted": False}}
    summary = summarize_drift(
        ks_results=ks_results,
        chi2_results=chi2_results,
        psi_results=psi_results,
        min_drifted_features_for_alert=3,
    )
    assert summary.should_alert is True
    assert summary.n_drifted_features == 3


def test_detect_psi_drift_returns_severity_levels() -> None:
    reference_df = _make_reference_df()
    current_df = _make_current_df_with_shift()
    results = detect_psi_drift(
        reference_df,
        current_df,
        psi_features=["Sales"],
        moderate_threshold=0.10,
        significant_threshold=0.25,
    )
    assert results["Sales"]["severity"] in {"moderate", "significant"}
