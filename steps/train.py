"""
Model training step: trains all candidate models and returns the best one.

Each model gets its own MLflow run — compare them side-by-side in the MLflow UI.
The best model (by validation F2-score) is returned as a ZenML artifact.

MLflow logging per model:
  - Hyperparameters from training_config.yaml
  - Validation F2, Recall, Precision, AUC-PR
  - Model artifact (sklearn-compatible, joblib serialised)
  - Tags: algorithm, best_model (if selected)

Training order: Dummy → Logistic Regression → Random Forest → LightGBM [→ XGBoost]
"""


from typing import Annotated

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.base import ClassifierMixin
from zenml import step

from core.evaluation import compute_metrics, find_optimal_threshold
from core.training import build_dummy, build_lightgbm, build_logistic_regression, build_random_forest

EXPERIMENT_NAME = "supply-chain-late-delivery"


def _log_model_run(
    model: ClassifierMixin,
    model_name: str,
    hyperparams: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[float, float, str]:
    """
    Train a model, log it to MLflow, and return (val_f2, best_threshold, run_id).

    Each call creates a self-contained MLflow run with:
      - All hyperparameters
      - Validation metrics (F2, Recall, Precision, AUC-PR)
      - The fitted model artifact
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Use nested=True if ZenML already opened a parent run; False if standalone
    nested = mlflow.active_run() is not None

    with mlflow.start_run(run_name=model_name, nested=nested) as run:
        # Log hyperparameters
        mlflow.log_params({"model": model_name, **hyperparams})

        # Train
        model.fit(X_train, y_train)

        # Evaluate on validation set (threshold tuning happens here)
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_val_proba = model.predict(X_val).astype(float)

        # Find optimal threshold on val set
        best_threshold, _ = find_optimal_threshold(y_val, y_val_proba, metric="f2")
        val_metrics = compute_metrics(y_val, y_val_proba, threshold=best_threshold)
        val_metrics["optimal_threshold"] = best_threshold

        # Log validation metrics with "val_" prefix for clarity
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Log the fitted model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(
            f"  {model_name:<25} val_f2={val_metrics['f2']:.4f}  "
            f"val_recall={val_metrics['recall']:.4f}  "
            f"threshold={best_threshold:.2f}"
        )

    return val_metrics["f2"], best_threshold, run.info.run_id


@step
def train_all_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    config_path: str = "configs/training_config.yaml",
) -> tuple[
    Annotated[ClassifierMixin, "best_model"],
    Annotated[str, "best_model_name"],
    Annotated[float, "best_val_f2"],
    Annotated[float, "best_threshold"],
]:
    """
    Train all candidate models and return the best one by validation F2-score.

    Each model is trained and logged as a separate MLflow run.
    After the step completes, open the MLflow UI to compare all runs:
        mlflow ui --port 5000

    Args:
        X_train: Training feature matrix (from engineer_features step).
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        config_path: Path to training_config.yaml.

    Returns:
        best_model: The fitted model with the highest validation F2-score.
        best_model_name: Name string (e.g., "lightgbm").
        best_val_f2: Validation F2-score of the best model.
        best_threshold: Best F2 threshold selected on validation set.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    random_state = config.get("random_state", 42)

    # Define the training agenda: (name, builder_fn, hyperparams_key)
    # Dummy has no hyperparameters
    agenda = [
        ("dummy",               build_dummy,                {}),
        ("logistic_regression", build_logistic_regression,  config.get("logistic_regression", {})),
        ("random_forest",       build_random_forest,        config.get("random_forest", {})),
        ("lightgbm",            build_lightgbm,             config.get("lightgbm", {})),
    ]

    # Optionally include XGBoost if installed — test the actual library, not the wrapper
    try:
        import xgboost  # noqa: F401

        from core.training import build_xgboost
        agenda.append(("xgboost", build_xgboost, config.get("xgboost", {})))
    except ImportError:
        print("  XGBoost not installed — skipping. Run: pip install xgboost>=2.0.0")

    print(f"\nTraining {len(agenda)} models. Each logged to MLflow experiment '{EXPERIMENT_NAME}'")
    print(f"{'Model':<25} {'Val F2':>8}  {'Val Recall':>10}  {'Threshold':>10}")
    print("-" * 60)

    results: list[tuple[float, float, str, ClassifierMixin, str]] = []

    for model_name, builder_fn, hyperparams in agenda:
        if model_name == "dummy":
            model = builder_fn(random_state=random_state)  # type: ignore[operator]
        else:
            model = builder_fn(hyperparams, random_state=random_state)  # type: ignore[operator]

        val_f2, best_threshold, run_id = _log_model_run(
            model, model_name, hyperparams,
            X_train, y_train, X_val, y_val,
        )
        results.append((val_f2, best_threshold, model_name, model, run_id))

    print("-" * 60)

    # Select best model by validation F2
    best_f2, best_threshold, best_name, best_model, best_run_id = max(results, key=lambda x: x[0])
    print(f"\nBest model: {best_name}  (val_f2={best_f2:.4f})")

    # Tag the winning run in MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("best_model", "true")
        mlflow.set_tag("selected_for_evaluation", "true")

    return best_model, best_name, float(best_f2), float(best_threshold)
