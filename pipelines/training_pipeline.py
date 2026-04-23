"""
Training pipeline: ingests data → validates → splits → engineers features →
trains all models → evaluates on test set → registers the best model.

Usage:
    python pipelines/training_pipeline.py

After the run:
    mlflow ui --port 5000
    # Go to: Models → supply-chain-late-delivery
    # Promote "staging" alias → "production" when satisfied

The human promotion gate (Staging → Production) is intentional. See steps/register.py.

ZenML multi-output note (v0.92.0):
    When a step has multiple Annotated outputs, ZenML returns a list of StepArtifact
    objects (one per output, in declaration order). Python tuple unpacking works
    via list.__iter__ — the clean way to receive multiple outputs in the pipeline body.
    Attribute access (result.name) does NOT work on the returned list.
"""

from zenml import pipeline

from steps.eda import run_eda_step
from steps.engineer_features import engineer_features
from steps.evaluate import evaluate_model
from steps.ingest import ingest_data
from steps.register import register_model
from steps.split import split_data
from steps.train import train_all_models
from steps.validate import validate_data


@pipeline(name="supply_chain_training_pipeline", enable_cache=False)
def training_pipeline(
    data_config_path: str = "configs/data_config.yaml",
    features_config_path: str = "configs/features_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
) -> None:
    """
    End-to-end training pipeline for the supply chain late delivery predictor.

    Steps:
        1. ingest_data       — Load raw CSV
        2. validate_data     — Schema, volume, null, target distribution checks
        3. run_eda_step      — Reproducible EDA diagnostics artifact
        4. split_data        — 80/10/10 stratified split by Late_delivery_risk
        5. engineer_features — Fit preprocessor on train, transform all splits
        6. train_all_models  — Train Dummy, LR, RF, LightGBM; return best by val F2
        7. evaluate_model    — Full test-set evaluation: CI, slices, confusion matrix
        8. register_model    — Gate on F2/Recall thresholds; register in MLflow registry

    enable_cache=False: always re-run. Re-enable caching once the pipeline is stable.
    """
    # Step 1-2: Load and gate on data quality
    raw_df = ingest_data(config_path=data_config_path)
    validated_df = validate_data(df=raw_df, config_path=data_config_path)

    # Step 3: Reproducible EDA snapshot before any split-specific processing
    _ = run_eda_step(df=validated_df)

    # Step 4: Split — stratified 80/10/10
    # ZenML returns a list[StepArtifact] for multi-output steps.
    # Unpack in declaration order: train_df, val_df, test_df.
    train_df, val_df, test_df = split_data(
        df=validated_df, config_path=data_config_path
    )

    # Step 5: Feature engineering — fit on train only
    # Declaration order: X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = engineer_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        features_config_path=features_config_path,
    )

    # Step 6: Train all models, return best by validation F2
    # Declaration order: best_model, best_model_name, best_val_f2, best_threshold
    best_model, best_model_name, best_val_f2, best_threshold = train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config_path=training_config_path,
    )

    # Step 7: Evaluate on test set (touched once, here, never before)
    evaluation_metrics = evaluate_model(
        best_model=best_model,
        best_model_name=best_model_name,
        X_val=X_val,
        y_val=y_val,
        optimal_threshold=best_threshold,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        config_path=training_config_path,
    )

    # Step 8: Gate + register
    register_model(
        best_model=best_model,
        preprocessor=preprocessor,
        evaluation_metrics=evaluation_metrics,
    )


if __name__ == "__main__":
    training_pipeline()
