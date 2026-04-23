"""
Model evaluation step: rigorous test-set evaluation with CIs and slice metrics.

The test set is touched exactly once — here. Never during training or threshold tuning.

What this step produces and logs to MLflow:
  1. Overall metrics at default threshold (0.5)
  2. Overall metrics at optimal threshold (tuned on val set — but val set is gone,
     so we use the threshold returned from train_all_models via a config param)
  3. Bootstrap confidence intervals (1000 resamples, 95% CI)
  4. Slice metrics: Shipping Mode, Market, Customer Segment
  5. Confusion matrix counts

Why compute CI on the test set and not the val set?
The test set is the unbiased estimate. Val set metrics are biased upward
because threshold tuning was performed on them. CI on test set is our
honest answer to "how confident are we in this model's performance?"
"""


import json
from typing import Annotated

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import ClassifierMixin
from zenml import step

from core.evaluation import (
    bootstrap_ci,
    compute_metrics,
    confusion_matrix_dict,
    evaluate_slices,
)

EXPERIMENT_NAME = "supply-chain-late-delivery"
SLICE_COLUMNS = ["Shipping Mode", "Market", "Customer Segment"]


@step
def evaluate_model(
    best_model: ClassifierMixin,
    best_model_name: str,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    optimal_threshold: float,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    config_path: str = "configs/training_config.yaml",
) -> Annotated[dict, "evaluation_metrics"]:
    """
    Full evaluation of the best model on the held-out test set.

    Steps:
    1. Use optimal threshold from training step
    2. Compute overall metrics at both default (0.5) and optimal threshold
    3. Bootstrap CI on test set — 1000 resamples, 95% CI
    4. Slice metrics for Shipping Mode, Market, Customer Segment
    5. Log everything to MLflow

    Args:
        best_model: The fitted model from train_all_models.
        best_model_name: Name of the model (e.g., "lightgbm").
        X_val: Validation feature matrix (unused, kept for interface compatibility).
        y_val: Validation labels.
        optimal_threshold: F2-optimal threshold selected on validation during training.
        X_test: Test feature matrix.
        y_test: Test labels.
        test_df: Original test DataFrame with raw column values (for slicing).
        config_path: Path to training_config.yaml.

    Returns:
        evaluation_metrics: Dict with all metrics. Used by register_model step.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    _ = X_val, y_val

    # --- Get predicted probabilities ---
    if hasattr(best_model, "predict_proba"):
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = best_model.predict(X_test).astype(float)

    # --- Test set metrics at default and optimal threshold ---
    metrics_default = compute_metrics(y_test, y_test_proba, threshold=0.5)
    metrics_optimal = compute_metrics(y_test, y_test_proba, threshold=optimal_threshold)

    print(f"\nTest Set Evaluation: {best_model_name}")
    print(f"{'Metric':<15} {'Default (0.5)':>14} {f'Optimal ({optimal_threshold:.2f})':>16}")
    print("-" * 48)
    for k in ["f2", "recall", "precision", "auc_pr", "auc_roc"]:
        print(f"  {k:<13} {metrics_default[k]:>14.4f} {metrics_optimal[k]:>16.4f}")

    # --- Bootstrap CI (on optimal threshold) ---
    print(f"\nBootstrap CI (1000 resamples, 95%, threshold={optimal_threshold:.2f}):")
    ci_results = bootstrap_ci(y_test, y_test_proba, threshold=optimal_threshold)
    for metric, vals in ci_results.items():
        print(
            f"  {metric:<10} {vals['mean']:.4f}  "
            f"(95% CI: {vals['ci_lower']:.4f} – {vals['ci_upper']:.4f})"
        )

    # --- Slice metrics ---
    # Reset test_df index to align with y_test and y_test_proba
    test_df_reset = test_df.reset_index(drop=True)
    slices = evaluate_slices(
        y_test, y_test_proba, test_df_reset,
        slice_cols=SLICE_COLUMNS,
        threshold=optimal_threshold,
    )

    print(f"\nSlice Evaluation (threshold={optimal_threshold:.2f}):")
    for col, col_slices in slices.items():
        print(f"\n  {col}:")
        for val, m in sorted(col_slices.items(), key=lambda x: x[1]["f2"], reverse=True):
            flag = " ⚠" if m["recall"] < 0.75 else ""
            print(
                f"    {str(val):<25} f2={m['f2']:.3f}  "
                f"recall={m['recall']:.3f}  n={m['n_samples']:,}{flag}"
            )

    # --- Confusion matrix ---
    cm = confusion_matrix_dict(y_test, y_test_proba, threshold=optimal_threshold)
    print(f"\nConfusion Matrix (threshold={optimal_threshold:.2f}):")
    print(f"  TP={cm['true_positives']:,}  FP={cm['false_positives']:,}  "
          f"TN={cm['true_negatives']:,}  FN={cm['false_negatives']:,}")

    # --- Log to MLflow ---
    mlflow.set_experiment(EXPERIMENT_NAME)
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"{best_model_name}_test_evaluation", nested=nested):
        mlflow.set_tag("evaluation_type", "test_set")
        mlflow.set_tag("model_name", best_model_name)

        # Log overall metrics
        mlflow.log_metrics({f"test_default_{k}": v for k, v in metrics_default.items()})
        mlflow.log_metrics({f"test_optimal_{k}": v for k, v in metrics_optimal.items()})
        mlflow.log_param("optimal_threshold", optimal_threshold)

        # Log CI metrics
        for metric, vals in ci_results.items():
            mlflow.log_metrics({
                f"test_ci_{metric}_mean":      vals["mean"],
                f"test_ci_{metric}_ci_lower":  vals["ci_lower"],
                f"test_ci_{metric}_ci_upper":  vals["ci_upper"],
            })

        # Log confusion matrix counts
        mlflow.log_metrics(cm)

        # Log slice metrics as a JSON artifact
        mlflow.log_text(json.dumps(slices, indent=2), "slice_metrics.json")

    # --- Assemble result dict ---
    evaluation_metrics = {
        "model_name":           best_model_name,
        "optimal_threshold":    optimal_threshold,
        "test_f2":              metrics_optimal["f2"],
        "test_recall":          metrics_optimal["recall"],
        "test_precision":       metrics_optimal["precision"],
        "test_auc_pr":          metrics_optimal["auc_pr"],
        "test_auc_roc":         metrics_optimal["auc_roc"],
        "test_f2_ci_lower":     ci_results["f2"]["ci_lower"],
        "test_f2_ci_upper":     ci_results["f2"]["ci_upper"],
        "test_recall_ci_lower": ci_results["recall"]["ci_lower"],
        "test_recall_ci_upper": ci_results["recall"]["ci_upper"],
        "n_test_samples":       int(len(y_test)),
        "min_f2_threshold":     config.get("min_f2_to_register", 0.50),
        "min_recall_threshold": config.get("min_recall_to_register", 0.70),
        "slices":               slices,
    }

    return evaluation_metrics
