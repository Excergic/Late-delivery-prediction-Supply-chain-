"""
Tests for core/validation.py.

No ZenML, MLflow, or Evidently imports — pure Python + pytest.
Tests are deterministic: fixed synthetic DataFrames, exact expected outputs.
"""

import pandas as pd

from core.validation import (
    ValidationReport,
    check_allowed_categories,
    check_dtypes,
    check_null_rates,
    check_schema,
    check_target,
    check_value_ranges,
    check_volume,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_COLUMNS = ["Type", "Shipping Mode", "Late_delivery_risk", "Market"]


def make_valid_df(n: int = 1000) -> pd.DataFrame:
    """Synthetic DataFrame that passes all validation checks."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Type": rng.choice(["DEBIT", "CASH", "TRANSFER"], size=n),
        "Shipping Mode": rng.choice(["Standard Class", "First Class"], size=n),
        "Late_delivery_risk": rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        "Market": rng.choice(["Europe", "LATAM"], size=n),
    })


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------

def test_schema_passes_when_all_columns_present():
    df = make_valid_df()
    ok, errors = check_schema(df, VALID_COLUMNS)
    assert ok is True
    assert errors == []


def test_schema_fails_on_missing_column():
    df = make_valid_df().drop(columns=["Shipping Mode"])
    ok, errors = check_schema(df, VALID_COLUMNS)
    assert ok is False
    assert any("Shipping Mode" in e for e in errors)


def test_schema_fails_with_all_missing_columns():
    df = pd.DataFrame()
    ok, errors = check_schema(df, VALID_COLUMNS)
    assert ok is False
    assert len(errors) > 0


# ---------------------------------------------------------------------------
# check_target
# ---------------------------------------------------------------------------

def test_target_passes_for_valid_binary_column():
    df = make_valid_df(2000)
    ok, errors = check_target(df, "Late_delivery_risk")
    assert ok is True
    assert errors == []


def test_target_fails_on_null_values():
    df = make_valid_df()
    df.loc[0, "Late_delivery_risk"] = None
    ok, errors = check_target(df, "Late_delivery_risk")
    assert ok is False
    assert any("null" in e.lower() for e in errors)


def test_target_fails_on_non_binary_values():
    df = make_valid_df()
    df.loc[0, "Late_delivery_risk"] = 2  # Invalid value
    ok, errors = check_target(df, "Late_delivery_risk")
    assert ok is False
    assert any("binary" in e.lower() for e in errors)


def test_target_fails_when_distribution_too_low():
    """Simulate labeling failure: nearly all orders marked on-time."""
    df = pd.DataFrame({"Late_delivery_risk": [0] * 950 + [1] * 50})  # 5% positive
    ok, errors = check_target(df, "Late_delivery_risk", min_positive_rate=0.40)
    assert ok is False
    assert any("distribution" in e.lower() or "label" in e.lower() for e in errors)


def test_target_fails_when_column_missing():
    df = make_valid_df().drop(columns=["Late_delivery_risk"])
    ok, errors = check_target(df, "Late_delivery_risk")
    assert ok is False


# ---------------------------------------------------------------------------
# check_volume
# ---------------------------------------------------------------------------

def test_volume_passes_above_threshold():
    df = make_valid_df(1000)
    ok, errors = check_volume(df, min_rows=100)
    assert ok is True


def test_volume_fails_below_threshold():
    df = make_valid_df(50)
    ok, errors = check_volume(df, min_rows=100)
    assert ok is False
    assert any("50" in e for e in errors)


# ---------------------------------------------------------------------------
# check_null_rates
# ---------------------------------------------------------------------------

def test_null_rates_passes_when_within_threshold():
    df = make_valid_df(1000)
    ok, errors = check_null_rates(df, {"Shipping Mode": 0.05})
    assert ok is True


def test_null_rates_fails_when_exceeded():
    df = make_valid_df(1000)
    # Set 30% of Shipping Mode to null — exceeds 5% threshold
    df.loc[:299, "Shipping Mode"] = None
    ok, errors = check_null_rates(df, {"Shipping Mode": 0.05})
    assert ok is False
    assert any("Shipping Mode" in e for e in errors)


def test_null_rates_skips_missing_columns():
    """If a column in thresholds doesn't exist in df, silently skip it."""
    df = make_valid_df(1000)
    ok, errors = check_null_rates(df, {"NonExistentColumn": 0.01})
    assert ok is True  # Should not fail — column just doesn't exist


# ---------------------------------------------------------------------------
# check_dtypes
# ---------------------------------------------------------------------------

def test_dtypes_pass_for_expected_schema():
    df = make_valid_df(200)
    ok, errors = check_dtypes(df, {"Late_delivery_risk": "numeric", "Shipping Mode": "string"})
    assert ok is True
    assert errors == []


def test_dtypes_fail_for_mismatched_type():
    df = make_valid_df(200)
    df["Late_delivery_risk"] = df["Late_delivery_risk"].astype(str)
    ok, errors = check_dtypes(df, {"Late_delivery_risk": "numeric"})
    assert ok is False
    assert any("expected numeric" in e for e in errors)


# ---------------------------------------------------------------------------
# check_value_ranges
# ---------------------------------------------------------------------------

def test_value_ranges_pass_when_within_bounds():
    df = make_valid_df(200)
    df["Order Item Quantity"] = 3
    ok, errors = check_value_ranges(df, {"Order Item Quantity": {"min": 1, "max": 100}})
    assert ok is True
    assert errors == []


def test_value_ranges_fail_when_out_of_bounds():
    df = make_valid_df(200)
    df["Order Item Quantity"] = 0
    ok, errors = check_value_ranges(df, {"Order Item Quantity": {"min": 1}})
    assert ok is False
    assert any("below min" in e for e in errors)


# ---------------------------------------------------------------------------
# check_allowed_categories
# ---------------------------------------------------------------------------

def test_allowed_categories_pass_for_known_values():
    df = make_valid_df(200)
    ok, errors = check_allowed_categories(
        df, {"Shipping Mode": ["Standard Class", "First Class"]}
    )
    assert ok is True
    assert errors == []


def test_allowed_categories_fail_for_unknown_values():
    df = make_valid_df(200)
    df.loc[0, "Shipping Mode"] = "Drone"
    ok, errors = check_allowed_categories(
        df, {"Shipping Mode": ["Standard Class", "First Class"]}
    )
    assert ok is False
    assert any("unknown categories" in e for e in errors)


# ---------------------------------------------------------------------------
# run_all_checks (integration)
# ---------------------------------------------------------------------------

def test_run_all_checks_passes_on_valid_data():
    df = make_valid_df(2000)
    config = {
        "expected_columns": VALID_COLUMNS,
        "target_column": "Late_delivery_risk",
        "min_rows": 100,
        "min_positive_rate": 0.40,
        "max_positive_rate": 0.65,
        "null_thresholds": {"Shipping Mode": 0.05},
        "expected_dtypes": {"Late_delivery_risk": "numeric"},
        "value_ranges": {"Late_delivery_risk": {"min": 0, "max": 1}},
        "allowed_categories": {"Shipping Mode": ["Standard Class", "First Class"]},
    }
    report = run_all_checks(df, config)
    assert report.passed is True
    assert all(report.checks.values())
    assert report.errors == []


def test_run_all_checks_fails_on_missing_column():
    df = make_valid_df(2000).drop(columns=["Market"])
    config = {
        "expected_columns": VALID_COLUMNS,
        "target_column": "Late_delivery_risk",
        "min_rows": 100,
        "min_positive_rate": 0.40,
        "max_positive_rate": 0.65,
    }
    report = run_all_checks(df, config)
    assert report.passed is False
    assert report.checks["schema"] is False


def test_validation_report_summary_contains_status():
    report = ValidationReport(passed=True, checks={"schema": True}, errors=[])
    summary = report.summary()
    assert "PASSED" in summary

    report_fail = ValidationReport(passed=False, checks={"schema": False}, errors=["Missing: X"])
    summary_fail = report_fail.summary()
    assert "FAILED" in summary_fail
    assert "Missing: X" in summary_fail
