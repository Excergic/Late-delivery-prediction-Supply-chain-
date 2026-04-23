"""Tests for core/eda.py."""

import pandas as pd

from core.eda import run_eda


def make_df(n: int = 200) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "f1": rng.normal(0, 1, size=n),
            "f2": rng.normal(5, 2, size=n),
            "Shipping Mode": rng.choice(["Standard Class", "First Class"], size=n),
            "Late_delivery_risk": rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        }
    )


def test_run_eda_returns_expected_sections():
    report = run_eda(make_df())
    assert "target" in report
    assert "feature_distributions" in report
    assert "top_correlations" in report
    assert "imbalance_assessment" in report


def test_target_distribution_keys():
    report = run_eda(make_df())
    target = report["target"]
    assert {"n_samples", "positive_rate", "negative_rate"}.issubset(set(target.keys()))
