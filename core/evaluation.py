"""
Model evaluation for the supply chain pipeline.

Pure Python — no ZenML, MLflow, or Evidently imports.

Three evaluation tiers:
  1. Overall metrics at a given threshold (F2, Recall, Precision, AUC-PR, AUC-ROC)
  2. Bootstrap confidence intervals on the test set — because a single number is a guess
  3. Slice-level metrics — because global metrics hide segment failures

Design principle: the test set is touched exactly once. All threshold decisions
happen on the validation set. The test set only sees the frozen threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def f2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F2-score: weights recall twice as heavily as precision.
    Formula: (5 × Precision × Recall) / (4 × Precision + Recall)
    Why F2 for late delivery? Missing a late delivery (FN) is more costly
    than a false alarm (FP). F2 penalises FNs more than F1 does.
    """
    return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the full metric suite at a given classification threshold.

    Threshold converts a probability (0.73) into a binary label (1).
    The default 0.5 is a reasonable starting point; use find_optimal_threshold()
    on the validation set to tune it, then apply the tuned value here.

    Args:
        y_true: True binary labels (0/1).
        y_pred_proba: Model's predicted probability of class 1.
        threshold: Decision threshold. Predictions ≥ threshold → class 1.

    Returns:
        Dict with f2, recall, precision, auc_pr, auc_roc, threshold, n_samples.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc_pr = float(average_precision_score(y_true, y_pred_proba))
    # roc_auc_score is undefined when only one class is present.
    try:
        auc_roc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auc_roc = float("nan")
    return {
        "f2":       round(f2(y_true, y_pred), 4),
        "recall":   round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "auc_pr":   round(auc_pr, 4),
        "auc_roc":  round(auc_roc, 4),
        "threshold": round(threshold, 4),
        "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f2",
    search_range: tuple[float, float] = (0.1, 0.9),
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Sweep classification thresholds and return the one that maximises the metric.

    Call this on the VALIDATION set only. Apply the returned threshold to the
    test set. Never tune the threshold on the test set — it biases the estimate.

    Why tune the threshold?
    The model outputs probabilities. 0.5 is an arbitrary default.
    For late delivery (where recall matters more than precision), we might prefer
    a lower threshold like 0.35 — flag more orders as late, accept more false alarms
    in exchange for catching more true late deliveries.

    Args:
        y_true: Validation set true labels.
        y_pred_proba: Validation set predicted probabilities.
        metric: 'f2' or 'recall'. Default 'f2' — maximises our primary metric.
        search_range: (min_threshold, max_threshold) inclusive.
        step: Granularity of the sweep.

    Returns:
        (best_threshold, best_score)
    """
    thresholds = np.arange(search_range[0], search_range[1] + step / 2, step)
    best_t, best_score = 0.5, -np.inf

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        score = (
            float(recall_score(y_true, y_pred, zero_division=0))
            if metric == "recall"
            else f2(y_true, y_pred)
        )
        if score > best_score:
            best_score = score
            best_t = t

    return round(float(best_t), 4), round(float(best_score), 4)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Bootstrap confidence intervals for F2, Recall, and Precision.

    Why bootstrap CI?
    "F2 = 0.81" is a guess. "F2 = 0.81 (95% CI: 0.79–0.83)" is evidence.
    Bootstrap resamples the test set 1000 times with replacement, computes the
    metric on each resample, and takes the 2.5th–97.5th percentiles as the interval.
    Wider intervals → more uncertainty. Narrow intervals → strong evidence.

    For model comparison: if the F2 CI of model A overlaps with model B's CI,
    you cannot claim A is meaningfully better than B.

    Args:
        y_true: Test set true labels (numpy array).
        y_pred_proba: Test set predicted probabilities.
        threshold: The frozen decision threshold from validation tuning.
        n_bootstrap: Number of bootstrap resamples. 1000 is standard.
        ci: Confidence level (0.95 = 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict: {metric: {mean, ci_lower, ci_upper}} for f2, recall, precision.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = (np.asarray(y_pred_proba) >= threshold).astype(int)

    scores: dict[str, list[float]] = {"f2": [], "recall": [], "precision": []}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        # Skip degenerate samples (all one class) — they produce undefined metrics
        if yt.sum() == 0 or yt.sum() == n:
            continue
        scores["f2"].append(f2(yt, yp))
        scores["recall"].append(float(recall_score(yt, yp, zero_division=0)))
        scores["precision"].append(float(precision_score(yt, yp, zero_division=0)))

    alpha = 1.0 - ci
    result: dict[str, dict[str, float]] = {}
    for metric, values in scores.items():
        arr = np.array(values)
        result[metric] = {
            "mean":      round(float(np.mean(arr)), 4),
            "ci_lower":  round(float(np.percentile(arr, alpha / 2 * 100)), 4),
            "ci_upper":  round(float(np.percentile(arr, (1 - alpha / 2) * 100)), 4),
        }
    return result


# ---------------------------------------------------------------------------
# Slice-level evaluation
# ---------------------------------------------------------------------------

def evaluate_slices(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    df: pd.DataFrame,
    slice_cols: list[str],
    threshold: float = 0.5,
    min_slice_size: int = 50,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute F2 and Recall for each value of each slice column.

    Why slice evaluation?
    Global metrics hide segment failures. Our model might have F2=0.82 overall
    but F2=0.45 for Same Day shipments. That's a product failure — Same Day
    customers pay more and expect reliability. Slice evaluation surfaces this.

    For supply chain, the critical slices are:
    - Shipping Mode: does the model work equally well for each mode?
    - Market: are predictions reliable in Africa vs Europe?
    - Customer Segment: do corporate orders get different accuracy than consumer?

    Args:
        y_true: True labels (must align with df rows).
        y_pred_proba: Predicted probabilities.
        df: DataFrame with original column values (pre-engineering).
        slice_cols: Column names to slice by.
        threshold: Decision threshold.
        min_slice_size: Skip slices with fewer than this many samples.

    Returns:
        {col: {category_value: {f2, recall, precision, n_samples, positive_rate}}}
    """
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    y_true_arr = np.asarray(y_true)

    results: dict[str, dict[str, dict[str, float]]] = {}

    for col in slice_cols:
        if col not in df.columns:
            continue
        results[col] = {}
        for val in sorted(df[col].dropna().unique()):
            mask = (df[col] == val).values
            if mask.sum() < min_slice_size:
                continue
            yt = y_true_arr[mask]
            yp = y_pred[mask]
            results[col][str(val)] = {
                "f2":           round(f2(yt, yp), 4),
                "recall":       round(float(recall_score(yt, yp, zero_division=0)), 4),
                "precision":    round(float(precision_score(yt, yp, zero_division=0)), 4),
                "n_samples":    int(mask.sum()),
                "positive_rate": round(float(yt.mean()), 4),
            }

    return results


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------

def confusion_matrix_dict(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> dict:
    """Return confusion matrix as a flat dict for MLflow logging."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_positives":  int(tp),
        "false_positives": int(fp),
        "true_negatives":  int(tn),
        "false_negatives": int(fn),
    }
