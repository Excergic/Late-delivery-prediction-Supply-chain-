"""
Unit tests for core/evaluation.py

Tests cover:
  - f2(): formula correctness
  - compute_metrics(): output shape and value ranges
  - find_optimal_threshold(): returns valid threshold that beats default
  - bootstrap_ci(): output structure, CI ordering, seed reproducibility
  - evaluate_slices(): key slices present, small-slice skipping
  - confusion_matrix_dict(): TP+FP+TN+FN equals N
"""

from __future__ import annotations

import numpy as np
import pytest

from core.evaluation import (
    bootstrap_ci,
    compute_metrics,
    confusion_matrix_dict,
    evaluate_slices,
    f2,
    find_optimal_threshold,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions():
    """Labels and proba where model is perfect."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.85])
    return y_true, y_proba


@pytest.fixture
def random_predictions():
    """Larger synthetic dataset for CI/slice tests."""
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, size=n)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.25, n), 0, 1)
    return y_true, y_proba


# ---------------------------------------------------------------------------
# f2()
# ---------------------------------------------------------------------------

def test_f2_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert f2(y_true, y_pred) == pytest.approx(1.0)


def test_f2_all_false_negative():
    """Predict everything as 0 when truth has 1s → recall = 0 → F2 = 0."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert f2(y_true, y_pred) == pytest.approx(0.0)


def test_f2_weights_recall_over_precision():
    """F2 favours recall. High-recall low-precision should beat high-precision low-recall."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # High recall (3/4), low precision (3/6)
    y_high_recall = np.array([1, 1, 1, 0, 1, 1, 1, 0])
    # High precision (2/2), low recall (2/4)
    y_high_prec   = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    assert f2(y_true, y_high_recall) > f2(y_true, y_high_prec)


# ---------------------------------------------------------------------------
# compute_metrics()
# ---------------------------------------------------------------------------

def test_compute_metrics_keys(perfect_predictions):
    y_true, y_proba = perfect_predictions
    m = compute_metrics(y_true, y_proba)
    expected_keys = {"f2", "recall", "precision", "auc_pr", "auc_roc", "threshold", "n_samples"}
    assert expected_keys == set(m.keys())


def test_compute_metrics_perfect(perfect_predictions):
    y_true, y_proba = perfect_predictions
    m = compute_metrics(y_true, y_proba, threshold=0.5)
    assert m["f2"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)


def test_compute_metrics_ranges(random_predictions):
    y_true, y_proba = random_predictions
    m = compute_metrics(y_true, y_proba, threshold=0.5)
    for k in ["f2", "recall", "precision", "auc_pr", "auc_roc"]:
        assert 0.0 <= m[k] <= 1.0, f"{k} out of [0, 1]: {m[k]}"
    assert m["n_samples"] == len(y_true)
    assert m["threshold"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# find_optimal_threshold()
# ---------------------------------------------------------------------------

def test_find_optimal_threshold_valid_range(random_predictions):
    y_true, y_proba = random_predictions
    t, score = find_optimal_threshold(y_true, y_proba, metric="f2")
    assert 0.1 <= t <= 0.9
    assert 0.0 <= score <= 1.0


def test_find_optimal_threshold_beats_default(random_predictions):
    """The tuned threshold should be at least as good as 0.5."""
    y_true, y_proba = random_predictions
    t, best_score = find_optimal_threshold(y_true, y_proba, metric="f2")
    default_metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert best_score >= default_metrics["f2"] - 1e-6


def test_find_optimal_threshold_recall_metric(random_predictions):
    y_true, y_proba = random_predictions
    t, score = find_optimal_threshold(y_true, y_proba, metric="recall")
    assert 0.1 <= t <= 0.9
    # Recall threshold should push towards lower values (flag more as positive)
    m = compute_metrics(y_true, y_proba, threshold=t)
    assert m["recall"] == pytest.approx(score, abs=1e-4)


# ---------------------------------------------------------------------------
# bootstrap_ci()
# ---------------------------------------------------------------------------

def test_bootstrap_ci_keys(random_predictions):
    y_true, y_proba = random_predictions
    ci = bootstrap_ci(y_true, y_proba, threshold=0.5)
    assert set(ci.keys()) == {"f2", "recall", "precision"}
    for metric_vals in ci.values():
        assert set(metric_vals.keys()) == {"mean", "ci_lower", "ci_upper"}


def test_bootstrap_ci_ordering(random_predictions):
    """ci_lower <= mean <= ci_upper for every metric."""
    y_true, y_proba = random_predictions
    ci = bootstrap_ci(y_true, y_proba, threshold=0.5)
    for metric, vals in ci.items():
        assert vals["ci_lower"] <= vals["mean"] <= vals["ci_upper"], (
            f"{metric}: ci_lower={vals['ci_lower']}, mean={vals['mean']}, ci_upper={vals['ci_upper']}"
        )


def test_bootstrap_ci_reproducible(random_predictions):
    """Same seed → same result."""
    y_true, y_proba = random_predictions
    ci1 = bootstrap_ci(y_true, y_proba, threshold=0.5, seed=99)
    ci2 = bootstrap_ci(y_true, y_proba, threshold=0.5, seed=99)
    for metric in ci1:
        assert ci1[metric]["mean"] == pytest.approx(ci2[metric]["mean"])


def test_bootstrap_ci_different_seeds_differ(random_predictions):
    """Different seeds → different results (probabilistically)."""
    y_true, y_proba = random_predictions
    ci1 = bootstrap_ci(y_true, y_proba, threshold=0.5, seed=1)
    ci2 = bootstrap_ci(y_true, y_proba, threshold=0.5, seed=2)
    # Very unlikely to be equal
    assert ci1["f2"]["mean"] != pytest.approx(ci2["f2"]["mean"])


# ---------------------------------------------------------------------------
# evaluate_slices()
# ---------------------------------------------------------------------------

def test_evaluate_slices_structure(random_predictions):
    import pandas as pd
    y_true, y_proba = random_predictions
    n = len(y_true)
    df = pd.DataFrame({
        "region": np.where(np.arange(n) % 2 == 0, "North", "South"),
    })
    result = evaluate_slices(y_true, y_proba, df, slice_cols=["region"])
    assert "region" in result
    assert "North" in result["region"]
    assert "South" in result["region"]
    for slice_vals in result["region"].values():
        assert set(slice_vals.keys()) == {"f2", "recall", "precision", "n_samples", "positive_rate"}


def test_evaluate_slices_skips_small(random_predictions):
    import pandas as pd
    y_true, y_proba = random_predictions
    n = len(y_true)
    # Make a tiny slice with 3 members — below min_slice_size=50
    categories = ["majority"] * (n - 3) + ["tiny"] * 3
    df = pd.DataFrame({"cat": categories})
    result = evaluate_slices(y_true, y_proba, df, slice_cols=["cat"], min_slice_size=50)
    assert "tiny" not in result.get("cat", {})
    assert "majority" in result["cat"]


def test_evaluate_slices_missing_column(random_predictions):
    """Silently skip columns not present in df."""
    import pandas as pd
    y_true, y_proba = random_predictions
    df = pd.DataFrame({"a": np.ones(len(y_true))})
    result = evaluate_slices(y_true, y_proba, df, slice_cols=["nonexistent"])
    assert "nonexistent" not in result


# ---------------------------------------------------------------------------
# confusion_matrix_dict()
# ---------------------------------------------------------------------------

def test_confusion_matrix_sums_to_n(perfect_predictions):
    y_true, y_proba = perfect_predictions
    cm = confusion_matrix_dict(y_true, y_proba, threshold=0.5)
    total = cm["true_positives"] + cm["false_positives"] + cm["true_negatives"] + cm["false_negatives"]
    assert total == len(y_true)


def test_confusion_matrix_perfect(perfect_predictions):
    y_true, y_proba = perfect_predictions
    cm = confusion_matrix_dict(y_true, y_proba, threshold=0.5)
    assert cm["false_positives"] == 0
    assert cm["false_negatives"] == 0
    assert cm["true_positives"] == int(y_true.sum())
    assert cm["true_negatives"] == int((1 - y_true).sum())
