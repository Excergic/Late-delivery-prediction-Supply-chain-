"""
Feature engineering step: builds, fits, and applies the preprocessing pipeline.

This is the most production-critical step.

The pipeline is fitted on train_df ONLY.
It is then applied (without refitting) to val_df and test_df.

Why? Because val and test represent "future" data the model has not seen.
If we refit the scaler on the full dataset, information from val/test leaks
into training through the scaler statistics. This is data leakage.

The fitted pipeline (preprocessor) is returned as a ZenML artifact.
At serving time, this artifact is loaded and .transform() is called on new orders —
no refitting, no recomputation, identical preprocessing guaranteed.
"""

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from zenml import step

from core.preprocessing import (
    build_preprocessor,
    get_column_groups,
    inject_unknown_categories,
    load_features_config,
    prepare_features,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = str(Path(__file__).parents[1] / "configs" / "features_config.yaml")


@step
def engineer_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features_config_path: str = _DEFAULT_CONFIG_PATH,
) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_val"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_val"],
    Annotated[np.ndarray, "y_test"],
    Annotated[ColumnTransformer, "preprocessor"],
]:
    """
    Apply feature engineering to all three splits.

    Steps:
    1. Extract date features and drop configured columns (applied to all splits)
    2. Build the ColumnTransformer (not yet fitted)
    3. Fit the preprocessor on train_df only — statistics frozen here
    4. Transform all three splits with the frozen preprocessor
    5. Return arrays + the fitted preprocessor as an artifact

    Args:
        train_df: Training split from split_data step.
        val_df: Validation split.
        test_df: Test split.
        features_config_path: Path to features_config.yaml.

    Returns:
        X_train, X_val, X_test: DataFrames of transformed features with column names
        y_train, y_val, y_test: numpy arrays of targets
        preprocessor: Fitted ColumnTransformer — the serving artifact
    """
    config = load_features_config(features_config_path)
    numeric_cols, onehot_cols, target_enc_cols = get_column_groups(config)

    # Apply pre-pipeline steps: date extraction + column drops
    X_train_raw, y_train = prepare_features(train_df, config)
    X_val_raw, y_val = prepare_features(val_df, config)
    X_test_raw, y_test = prepare_features(test_df, config)

    # Only include columns that actually exist in the data after dropping
    # (guards against config listing a column that doesn't exist)
    available = set(X_train_raw.columns)
    numeric_cols = [c for c in numeric_cols if c in available]
    onehot_cols = [c for c in onehot_cols if c in available]
    target_enc_cols = [c for c in target_enc_cols if c in available]

    # Optional train-only unknown injection for target-encoded high-cardinality features.
    X_train_for_fit = inject_unknown_categories(
        X_train_raw,
        target_enc_cols=target_enc_cols,
        unknown_fraction=float(config.get("target_enc_unknown_fraction", 0.05)),
        random_state=int(config.get("random_state", 42)),
    )

    # Build and fit preprocessor on training data ONLY
    preprocessor = build_preprocessor(
        numeric_cols,
        onehot_cols,
        target_enc_cols,
        numeric_scaler=str(config.get("numeric_scaler", "standard")),
    )
    X_train_arr = preprocessor.fit_transform(X_train_for_fit, y_train)

    # Transform val and test with frozen statistics from training
    X_val_arr = preprocessor.transform(X_val_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    # Attach feature names so LightGBM (and any SHAP/importance tooling) can use them.
    # ColumnTransformer.get_feature_names_out() is available in sklearn >= 1.0.
    feature_names = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_val = pd.DataFrame(X_val_arr, columns=feature_names)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names)

    n_features = X_train.shape[1]
    logger.info("Feature engineering complete:")
    logger.info(
        "  Input columns used:  %d numeric, %d one-hot, %d target-enc",
        len(numeric_cols), len(onehot_cols), len(target_enc_cols),
    )
    logger.info("  Output feature count: %d", n_features)
    logger.info("  X_train shape: %s", X_train.shape)
    logger.info("  X_val   shape: %s", X_val.shape)
    logger.info("  X_test  shape: %s", X_test.shape)

    return (
        X_train,
        X_val,
        X_test,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        preprocessor,
    )
