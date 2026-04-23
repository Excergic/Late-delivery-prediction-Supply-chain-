"""
Tests for core/preprocessing.py.

No ZenML, MLflow, or Evidently imports — pure Python + pytest.
Tests verify deterministic behavior: fixed inputs produce exact expected outputs.
"""

import numpy as np
import pandas as pd

from core.preprocessing import (
    build_preprocessor,
    drop_columns,
    extract_date_features,
    get_column_groups,
    prepare_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_sample_df(n: int = 200) -> pd.DataFrame:
    """Small synthetic DataFrame for testing preprocessing logic."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Type": rng.choice(["DEBIT", "CASH", "TRANSFER"], size=n),
        "Shipping Mode": rng.choice(["Standard Class", "First Class", "Same Day"], size=n),
        "Market": rng.choice(["Europe", "LATAM", "USCA"], size=n),
        "Days for shipment (scheduled)": rng.choice([1, 2, 4], size=n).astype(float),
        "Product Price": rng.uniform(10, 500, size=n),
        "Sales": rng.uniform(50, 1000, size=n),
        "Category Name": rng.choice(["Cleats", "Fishing", "Electronics", "Golf"], size=n),
        "order date (DateOrders)": ["1/15/2018 10:30"] * n,
        "Late_delivery_risk": rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        "Delivery Status": ["COMPLETE"] * n,  # Leakage column — must be dropped
        "Customer Email": ["x@x.com"] * n,    # PII column — must be dropped
    })


SAMPLE_FEATURES_CONFIG = {
    "date_column": "order date (DateOrders)",
    "drop_columns": ["Delivery Status", "Customer Email"],
    "numeric_columns": ["Days for shipment (scheduled)", "Product Price", "Sales"],
    "onehot_columns": ["Type", "Shipping Mode", "Market"],
    "target_enc_columns": ["Category Name"],
    "date_derived_features": ["order_hour", "order_day_of_week", "order_month", "order_is_weekend"],
}


# ---------------------------------------------------------------------------
# extract_date_features
# ---------------------------------------------------------------------------

def test_date_features_are_extracted():
    df = make_sample_df(50)
    result = extract_date_features(df, "order date (DateOrders)")
    assert "order_hour" in result.columns
    assert "order_day_of_week" in result.columns
    assert "order_month" in result.columns
    assert "order_is_weekend" in result.columns


def test_original_date_column_is_dropped():
    df = make_sample_df(50)
    result = extract_date_features(df, "order date (DateOrders)")
    assert "order date (DateOrders)" not in result.columns


def test_date_feature_values_are_in_valid_range():
    df = make_sample_df(50)
    result = extract_date_features(df, "order date (DateOrders)")
    assert result["order_hour"].between(0, 23).all()
    assert result["order_day_of_week"].between(0, 6).all()
    assert result["order_month"].between(1, 12).all()
    assert result["order_is_weekend"].isin([0, 1]).all()


def test_extract_date_does_not_modify_input():
    """extract_date_features must not mutate the input DataFrame (defensive copy)."""
    df = make_sample_df(50)
    original_cols = list(df.columns)
    _ = extract_date_features(df, "order date (DateOrders)")
    assert list(df.columns) == original_cols  # Input unchanged


# ---------------------------------------------------------------------------
# drop_columns
# ---------------------------------------------------------------------------

def test_drop_columns_removes_specified_columns():
    df = make_sample_df(50)
    result = drop_columns(df, ["Delivery Status", "Customer Email"])
    assert "Delivery Status" not in result.columns
    assert "Customer Email" not in result.columns


def test_drop_columns_silently_skips_nonexistent():
    df = make_sample_df(50)
    # Should not raise even if column doesn't exist
    result = drop_columns(df, ["NonExistentColumn", "Delivery Status"])
    assert "Delivery Status" not in result.columns


def test_drop_columns_keeps_remaining_columns():
    df = make_sample_df(50)
    original_count = len(df.columns)
    result = drop_columns(df, ["Delivery Status"])
    assert len(result.columns) == original_count - 1


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

def test_prepare_features_separates_X_and_y():
    df = make_sample_df(200)
    X, y = prepare_features(df, SAMPLE_FEATURES_CONFIG)
    assert "Late_delivery_risk" not in X.columns
    assert len(y) == len(X) == 200


def test_prepare_features_drops_leakage_columns():
    df = make_sample_df(200)
    X, y = prepare_features(df, SAMPLE_FEATURES_CONFIG)
    assert "Delivery Status" not in X.columns
    assert "Customer Email" not in X.columns


def test_prepare_features_adds_date_features():
    df = make_sample_df(200)
    X, y = prepare_features(df, SAMPLE_FEATURES_CONFIG)
    assert "order_hour" in X.columns
    assert "order date (DateOrders)" not in X.columns


# ---------------------------------------------------------------------------
# build_preprocessor + end-to-end
# ---------------------------------------------------------------------------

def test_preprocessor_fits_and_transforms():
    df = make_sample_df(200)
    X, y = prepare_features(df, SAMPLE_FEATURES_CONFIG)
    numeric_cols, onehot_cols, target_enc_cols = get_column_groups(SAMPLE_FEATURES_CONFIG)
    available = set(X.columns)
    numeric_cols = [c for c in numeric_cols if c in available]
    onehot_cols = [c for c in onehot_cols if c in available]
    target_enc_cols = [c for c in target_enc_cols if c in available]

    preprocessor = build_preprocessor(numeric_cols, onehot_cols, target_enc_cols)
    X_transformed = preprocessor.fit_transform(X, y)

    assert X_transformed.shape[0] == 200
    assert X_transformed.shape[1] > 0


def test_transform_produces_same_shape_as_fit_transform():
    """
    Critical: transform() must produce the same number of columns as fit_transform().
    This verifies there's no train-serve skew in output dimensionality.
    """
    df_train = make_sample_df(150)
    df_test = make_sample_df(50)
    # Use different rng to get slightly different data

    X_train, y_train = prepare_features(df_train, SAMPLE_FEATURES_CONFIG)
    X_test, _ = prepare_features(df_test, SAMPLE_FEATURES_CONFIG)

    numeric_cols, onehot_cols, target_enc_cols = get_column_groups(SAMPLE_FEATURES_CONFIG)
    available = set(X_train.columns)
    numeric_cols = [c for c in numeric_cols if c in available]
    onehot_cols = [c for c in onehot_cols if c in available]
    target_enc_cols = [c for c in target_enc_cols if c in available]

    preprocessor = build_preprocessor(numeric_cols, onehot_cols, target_enc_cols)
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)

    assert X_train_t.shape[1] == X_test_t.shape[1], (
        f"Column count mismatch: train={X_train_t.shape[1]}, test={X_test_t.shape[1]}. "
        f"This indicates training-serving skew."
    )


def test_numeric_features_are_scaled_to_zero_mean():
    """
    StandardScaler should produce ~zero mean on the fitted data.
    Tests that the scaler is actually applied, not bypassed.
    """
    df = make_sample_df(500)
    X, y = prepare_features(df, SAMPLE_FEATURES_CONFIG)
    numeric_cols, onehot_cols, target_enc_cols = get_column_groups(SAMPLE_FEATURES_CONFIG)
    available = set(X.columns)
    numeric_cols = [c for c in numeric_cols if c in available]
    onehot_cols = [c for c in onehot_cols if c in available]
    target_enc_cols = [c for c in target_enc_cols if c in available]

    preprocessor = build_preprocessor(numeric_cols, onehot_cols, target_enc_cols)
    X_t = preprocessor.fit_transform(X, y)

    # First few columns are numeric — should have near-zero mean after StandardScaler
    numeric_means = np.abs(X_t[:, :len(numeric_cols)].mean(axis=0))
    assert (numeric_means < 0.1).all(), (
        f"Numeric columns have non-zero mean after StandardScaler: {numeric_means}"
    )


def test_unseen_category_handled_gracefully():
    """
    At serving time, a new category value not seen during training
    must NOT raise an error. OHE uses handle_unknown='ignore' (→ zero vector).
    """
    df_train = make_sample_df(200)
    X_train, y_train = prepare_features(df_train, SAMPLE_FEATURES_CONFIG)

    # Inject a never-before-seen category
    df_test = make_sample_df(20)
    df_test["Type"] = "BITCOIN"  # Not in training data
    X_test, _ = prepare_features(df_test, SAMPLE_FEATURES_CONFIG)

    numeric_cols, onehot_cols, target_enc_cols = get_column_groups(SAMPLE_FEATURES_CONFIG)
    available = set(X_train.columns)
    numeric_cols = [c for c in numeric_cols if c in available]
    onehot_cols = [c for c in onehot_cols if c in available]
    target_enc_cols = [c for c in target_enc_cols if c in available]

    preprocessor = build_preprocessor(numeric_cols, onehot_cols, target_enc_cols)
    preprocessor.fit_transform(X_train, y_train)

    # This must not raise — unseen category → zero vector via handle_unknown='ignore'
    X_test_t = preprocessor.transform(X_test)
    assert X_test_t.shape[0] == 20
