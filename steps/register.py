"""
Model registration step: gate + register in MLflow model registry.

This step checks evaluation metrics against the thresholds defined in
training_config.yaml. If both F2 and Recall pass, the model is registered
in the MLflow model registry with alias "staging".

Promoting from staging → production is a human decision made in the MLflow UI:
  1. Open MLflow UI: mlflow ui --port 5000
  2. Go to Models → supply-chain-late-delivery
  3. Find the latest version with alias "staging"
  4. Review metrics, click "Transition to → Production"

This human gate is intentional: the first production system should not
auto-promote. After a few production cycles where evaluation gates are proven
trustworthy, you can automate this step.

Rollback: in the MLflow UI, transition the current Production version back to
Staging, and transition the previous version back to Production.
Target: rollback in under 5 minutes.
"""

import logging
from typing import Annotated

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from zenml import step

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "supply-chain-late-delivery"
EXPERIMENT_NAME = "supply-chain-late-delivery"


@step
def register_model(
    best_model: ClassifierMixin,
    preprocessor: ColumnTransformer,
    evaluation_metrics: dict,
) -> Annotated[bool, "was_registered"]:
    """
    Check evaluation gates and register the model in the MLflow model registry.

    Gate logic:
      - test_f2   >= min_f2_to_register   (from training_config.yaml)
      - test_recall >= min_recall_to_register

    If both pass:
      - Single model bundle (preprocessor + model) is logged and registered
        under "supply-chain-late-delivery"
      - Version is set to alias "staging" (ready for human review)
      - Decision threshold is stored with model metadata

    If either fails:
      - Registration is skipped
      - Failure details are logged so you know what to improve

    Args:
        best_model: Fitted model from train_all_models.
        preprocessor: Fitted ColumnTransformer from engineer_features.
        evaluation_metrics: Dict from evaluate_model step.

    Returns:
        was_registered: True if model passed gates and was registered, False otherwise.
    """
    model_name = evaluation_metrics.get("model_name", "unknown")
    test_f2 = evaluation_metrics.get("test_f2", 0.0)
    test_recall = evaluation_metrics.get("test_recall", 0.0)
    min_f2 = evaluation_metrics.get("min_f2_threshold", 0.50)
    min_recall = evaluation_metrics.get("min_recall_threshold", 0.70)
    optimal_threshold = evaluation_metrics.get("optimal_threshold", 0.5)

    f2_passes = test_f2 >= min_f2
    recall_passes = test_recall >= min_recall
    both_pass = f2_passes and recall_passes

    f2_symbol = "≥" if f2_passes else "<"
    f2_status = "✓ PASS" if f2_passes else "✗ FAIL"
    recall_symbol = "≥" if recall_passes else "<"
    recall_status = "✓ PASS" if recall_passes else "✗ FAIL"

    print(f"\nEvaluation Gates for '{model_name}':")
    print(f"  F2:     {test_f2:.4f} {f2_symbol} {min_f2}  {f2_status}")
    print(f"  Recall: {test_recall:.4f} {recall_symbol} {min_recall}  {recall_status}")

    if not both_pass:
        print("\n✗ Model does NOT meet thresholds — skipping registration.")
        print("  To lower thresholds: edit min_f2_to_register / min_recall_to_register in configs/training_config.yaml")
        return False

    # Register in MLflow model registry
    mlflow.set_experiment(EXPERIMENT_NAME)
    nested = mlflow.active_run() is not None

    with mlflow.start_run(run_name=f"{model_name}_registration", nested=nested):
        mlflow.log_metrics({
            "registered_test_f2":      test_f2,
            "registered_test_recall":  test_recall,
            "registered_threshold":    optimal_threshold,
        })
        mlflow.log_params({
            "model_name":        model_name,
            "optimal_threshold": optimal_threshold,
        })
        mlflow.set_tag("stage", "staging")
        mlflow.set_tag("awaiting_human_promotion", "true")

        inference_bundle = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_model),
        ])

        mlflow.sklearn.log_model(
            inference_bundle,
            name="model_bundle",
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={"decision_threshold": optimal_threshold},
        )

    # Set "staging" alias on the newly registered version
    # MJ-1 fix: use search_model_versions() — get_latest_versions(stages=) is deprecated since 2.9.0
    # MJ-4 fix: catch only MlflowException (expected failures), not bare Exception (masks bugs)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if versions:
            version = max(versions, key=lambda v: int(v.version)).version
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="staging",
                version=version,
            )
            print(f"\n✓ Registered '{REGISTERED_MODEL_NAME}' version {version} → alias 'staging'")
            test_f2_ci_lower = float(evaluation_metrics.get("test_f2_ci_lower", 0.0))
            test_f2_ci_upper = float(evaluation_metrics.get("test_f2_ci_upper", 0.0))
            print(
                f"  F2={test_f2:.4f} (CI: {test_f2_ci_lower:.4f}–{test_f2_ci_upper:.4f})"
            )
            print(f"  Recall={test_recall:.4f}  Threshold={optimal_threshold:.2f}")
            print(f"\nNext step: Open MLflow UI → Models → {REGISTERED_MODEL_NAME}")
            print("  Review metrics and promote 'staging' → 'production' when satisfied.")
            print("  Command: mlflow ui --port 5000")
    except MlflowException as e:
        # Expected when using file-based tracking without a server — aliases require
        # a tracking server (mlflow server --host 0.0.0.0 --port 5001).
        # The model IS registered; only the alias could not be set.
        logger.warning("Could not set 'staging' alias on registered model: %s", e)
        print(f"  ⚠ Model registered but alias not set: {e}")
        print("  Set alias manually in the MLflow UI: mlflow ui --port 5000")

    return True
