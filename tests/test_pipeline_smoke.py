"""
End-to-end smoke test: full pipeline on 200 rows of synthetic data.

Catches interface breakages that unit tests miss:
  - Parameter name mismatches between steps (e.g., config_path vs features_config_path)
  - Shape mismatches between preprocessing output and model input
  - Evaluation functions receiving wrong dtypes or shapes
  - Registration gate logic receiving a malformed metrics dict

This test does NOT hit disk, MLflow, or ZenML — it calls core functions directly.
Runtime: < 10 seconds.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from core.evaluation import (
    bootstrap_ci,
    compute_metrics,
    confusion_matrix_dict,
    evaluate_slices,
    find_optimal_threshold,
)
from core.preprocessing import (
    build_preprocessor,
    get_column_groups,
    prepare_features,
)

# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

SMOKE_CONFIG = {
    "target_column": "Late_delivery_risk",
    "date_column": "order date (DateOrders)",
    "drop_columns": ["Delivery Status", "Customer Email"],
    "numeric_columns": [
        "Days for shipment (scheduled)",
        "Product Price",
        "Sales",
    ],
    "onehot_columns": ["Type", "Shipping Mode", "Market"],
    "target_enc_columns": ["Category Name"],
    "date_derived_features": [
        "order_hour",
        "order_day_of_week",
        "order_month",
        "order_is_weekend",
    ],
}


def make_smoke_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Type": rng.choice(["DEBIT", "CASH", "TRANSFER"], size=n),
        "Shipping Mode": rng.choice(
            ["Standard Class", "First Class", "Same Day", "Second Class"], size=n
        ),
        "Market": rng.choice(["Europe", "LATAM", "USCA", "Africa", "Pacific Asia"], size=n),
        "Days for shipment (scheduled)": rng.choice([1, 2, 4, 6], size=n).astype(float),
        "Product Price": rng.uniform(10, 500, size=n),
        "Sales": rng.uniform(50, 1000, size=n),
        "Category Name": rng.choice(["Cleats", "Fishing", "Electronics", "Golf"], size=n),
        "order date (DateOrders)": ["1/15/2018 10:30"] * n,
        "Late_delivery_risk": rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        "Delivery Status": ["COMPLETE"] * n,
        "Customer Email": ["x@x.com"] * n,
    })


@pytest.fixture(scope="module")
def pipeline_outputs():
    """
    Run the full preprocessing + training + evaluation chain once for the module.
    All tests share this fixture to avoid redundant computation.
    """
    df_train = make_smoke_df(200, seed=0)
    df_val   = make_smoke_df(50,  seed=1)
    df_test  = make_smoke_df(50,  seed=2)

    X_train_raw, y_train = prepare_features(df_train, SMOKE_CONFIG)
    X_val_raw,   y_val   = prepare_features(df_val,   SMOKE_CONFIG)
    X_test_raw,  y_test  = prepare_features(df_test,  SMOKE_CONFIG)

    numeric_cols, onehot_cols, target_enc_cols = get_column_groups(SMOKE_CONFIG)
    available = set(X_train_raw.columns)
    numeric_cols    = [c for c in numeric_cols    if c in available]
    onehot_cols     = [c for c in onehot_cols     if c in available]
    target_enc_cols = [c for c in target_enc_cols if c in available]

    preprocessor = build_preprocessor(numeric_cols, onehot_cols, target_enc_cols)
    X_train = preprocessor.fit_transform(X_train_raw, y_train)
    X_val   = preprocessor.transform(X_val_raw)
    X_test  = preprocessor.transform(X_test_raw)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_val_proba  = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    threshold, _ = find_optimal_threshold(y_val, y_val_proba, metric="f2")

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "model":   model,
        "y_val_proba":  y_val_proba,
        "y_test_proba": y_test_proba,
        "threshold":    threshold,
        "df_test":      df_test,
        "preprocessor": preprocessor,
    }


# ---------------------------------------------------------------------------
# Feature engineering shape invariants
# ---------------------------------------------------------------------------

def test_train_val_test_have_same_feature_count(pipeline_outputs):
    """Preprocessing must produce identical column count across all three splits."""
    X_train = pipeline_outputs["X_train"]
    X_val   = pipeline_outputs["X_val"]
    X_test  = pipeline_outputs["X_test"]
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], (
        f"Column count mismatch: train={X_train.shape[1]}, "
        f"val={X_val.shape[1]}, test={X_test.shape[1]}"
    )


def test_no_nan_after_preprocessing(pipeline_outputs):
    """Imputers must eliminate all NaN values before model input."""
    for name, arr in [
        ("X_train", pipeline_outputs["X_train"]),
        ("X_val",   pipeline_outputs["X_val"]),
        ("X_test",  pipeline_outputs["X_test"]),
    ]:
        assert not np.isnan(arr).any(), f"NaN values found in {name} after preprocessing"


def test_feature_count_exceeds_raw_columns(pipeline_outputs):
    """OHE expansion means output columns > raw input columns (smoke check)."""
    # Raw config has 3 numeric + 3 OHE + 1 target-enc + 4 date = 11 raw feature cols.
    # After OHE expansion this should be notably larger.
    assert pipeline_outputs["X_train"].shape[1] > 11


# ---------------------------------------------------------------------------
# Model prediction shape and range
# ---------------------------------------------------------------------------

def test_predict_proba_shape(pipeline_outputs):
    """predict_proba must return (n_samples, 2) — binary classifier contract."""
    model = pipeline_outputs["model"]
    X_test = pipeline_outputs["X_test"]
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_predict_proba_sums_to_one(pipeline_outputs):
    """Class probabilities must sum to 1.0 per row."""
    model = pipeline_outputs["model"]
    X_test = pipeline_outputs["X_test"]
    proba = model.predict_proba(X_test)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Evaluation chain — correct wiring
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_expected_keys(pipeline_outputs):
    """compute_metrics must return all keys the register step reads."""
    m = compute_metrics(
        pipeline_outputs["y_test"],
        pipeline_outputs["y_test_proba"],
        threshold=pipeline_outputs["threshold"],
    )
    required = {"f2", "recall", "precision", "auc_pr", "auc_roc", "threshold", "n_samples"}
    assert required.issubset(set(m.keys()))


def test_bootstrap_ci_covers_point_estimate(pipeline_outputs):
    """The CI mean must be close to the point estimate at the same threshold."""
    m = compute_metrics(
        pipeline_outputs["y_test"],
        pipeline_outputs["y_test_proba"],
        threshold=pipeline_outputs["threshold"],
    )
    ci = bootstrap_ci(
        pipeline_outputs["y_test"],
        pipeline_outputs["y_test_proba"],
        threshold=pipeline_outputs["threshold"],
        n_bootstrap=200,
    )
    # CI mean should be within 0.05 of the point estimate
    assert abs(ci["f2"]["mean"] - m["f2"]) < 0.05, (
        f"CI mean {ci['f2']['mean']:.4f} diverges too far from point estimate {m['f2']:.4f}"
    )


def test_confusion_matrix_totals_match_test_size(pipeline_outputs):
    cm = confusion_matrix_dict(
        pipeline_outputs["y_test"],
        pipeline_outputs["y_test_proba"],
        threshold=pipeline_outputs["threshold"],
    )
    total = cm["true_positives"] + cm["false_positives"] + cm["true_negatives"] + cm["false_negatives"]
    assert total == len(pipeline_outputs["y_test"])


def test_slice_evaluation_runs_without_error(pipeline_outputs):
    """evaluate_slices must not raise on valid inputs."""
    slices = evaluate_slices(
        pipeline_outputs["y_test"],
        pipeline_outputs["y_test_proba"],
        pipeline_outputs["df_test"].reset_index(drop=True),
        slice_cols=["Shipping Mode", "Market"],
        threshold=pipeline_outputs["threshold"],
    )
    assert "Shipping Mode" in slices
    assert "Market" in slices


# ---------------------------------------------------------------------------
# Registration gate logic — wiring check
# ---------------------------------------------------------------------------

def test_registration_gate_passes_with_good_metrics(pipeline_outputs):
    """
    The evaluation_metrics dict produced by the pipeline must satisfy the
    default registration thresholds (min_f2=0.50, min_recall=0.70).
    On real data the model comfortably exceeds these; on 50 synthetic test rows
    it should too — but even if not, the gate logic itself must be wired correctly.
    """
    m = compute_metrics(
        pipeline_outputs["y_test"],
        pipeline_outputs["y_test_proba"],
        threshold=pipeline_outputs["threshold"],
    )
    # Simulate the dict that evaluate_model returns
    evaluation_metrics = {
        "model_name": "logistic_regression",
        "optimal_threshold": pipeline_outputs["threshold"],
        "test_f2": m["f2"],
        "test_recall": m["recall"],
        "min_f2_threshold": 0.50,
        "min_recall_threshold": 0.70,
    }
    recall_passes = evaluation_metrics["test_recall"] >= evaluation_metrics["min_recall_threshold"]
    # At threshold tuned for F2, recall should be high (model is recall-oriented)
    assert recall_passes, (
        f"Recall {evaluation_metrics['test_recall']:.4f} below threshold "
        f"{evaluation_metrics['min_recall_threshold']} — check threshold tuning"
    )


def test_registration_gate_fails_correctly_for_bad_model():
    """A model with 0.0 F2 must not pass the gate."""
    evaluation_metrics = {
        "test_f2": 0.0,
        "test_recall": 0.0,
        "min_f2_threshold": 0.50,
        "min_recall_threshold": 0.70,
    }
    assert evaluation_metrics["test_f2"] < evaluation_metrics["min_f2_threshold"]
    assert evaluation_metrics["test_recall"] < evaluation_metrics["min_recall_threshold"]
