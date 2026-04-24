"""
Feature engineering for the supply chain pipeline.

Pure Python — no ZenML, MLflow, or Evidently imports.

The central design principle: every transformation that happens during training
must happen identically during serving. We achieve this by:
  1. Bundling ALL preprocessing into a single sklearn.Pipeline.
  2. Serializing that pipeline as a versioned artifact.
  3. Loading the frozen artifact at serving time — not recomputing anything.

This file builds the preprocessing pipeline. Fitting happens in the ZenML step,
on the training split only. The fitted pipeline is the artifact that gets registered.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


def load_features_config(config_path: str = "configs/features_config.yaml") -> dict[str, Any]:
    """Load the features configuration YAML."""
    with open(config_path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extract temporal features from an order date column and drop the original.

    Temporal features capture patterns like: orders placed at 11pm may route
    differently, or month captures holiday seasonality in shipping.

    Args:
        df: DataFrame containing the date column.
        date_col: Column name of the raw datetime string.

    Returns:
        DataFrame with original date column replaced by four numeric features.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
    df["order_hour"] = dates.dt.hour.astype("int32")
    df["order_day_of_week"] = dates.dt.dayofweek.astype("int32")  # 0=Mon, 6=Sun
    df["order_month"] = dates.dt.month.astype("int32")
    df["order_is_weekend"] = (dates.dt.dayofweek >= 5).astype("int32")
    df = df.drop(columns=[date_col])
    return df


def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """
    Drop configured columns. Silently skips columns that don't exist.

    Using the config list (not hardcoded) means changing which columns to drop
    only requires a YAML edit — no code change, no git diff in source files.
    """
    existing = [c for c in columns_to_drop if c in df.columns]
    return df.drop(columns=existing)


def build_preprocessor(
    numeric_cols: list[str],
    onehot_cols: list[str],
    target_enc_cols: list[str],
    numeric_scaler: str = "standard",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies three preprocessing paths.

    PATH 1 — Numeric features:
        SimpleImputer(median) → StandardScaler
        Formula: z = (x − mean_train) / std_train
        The mean and std are computed on the training split and FROZEN.
        Serving always uses the training mean/std — never recomputed.

    PATH 2 — Low-cardinality categoricals (OneHot):
        SimpleImputer('UNKNOWN') → OneHotEncoder(handle_unknown='ignore', drop='first')
        Unseen categories at serving time → zero vector (ignored).
        drop='first' removes one dummy per category for linear models.

    PATH 3 — High-cardinality categoricals (Target Encoding):
        SimpleImputer('UNKNOWN') → TargetEncoder(smooth='auto')
        Formula: encoded = (n × category_mean + m × global_mean) / (n + m)
        where n = category count, m = auto-selected smoothing factor.
        Smoothing prevents rare categories (seen once, y=1) from encoding as 1.0.
        Like PATH 1: all encodings are frozen from training.

    The ColumnTransformer's remainder='drop' discards any columns not explicitly
    listed — a safeguard against unexpected columns reaching the model.

    Args:
        numeric_cols: Column names for PATH 1.
        onehot_cols: Column names for PATH 2.
        target_enc_cols: Column names for PATH 3.

    Returns:
        Unfitted ColumnTransformer. Call .fit(X_train, y_train) to train it.
    """
    scaler_name = numeric_scaler.lower().strip()
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    onehot_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )),
    ])

    # TargetEncoder uses y during .fit() to compute P(y=1 | category).
    # sklearn.Pipeline passes y through to all transformers during pipeline.fit(X, y).
    target_enc_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encoder", TargetEncoder(smooth="auto")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("onehot", onehot_pipe, onehot_cols),
            ("target_enc", target_enc_pipe, target_enc_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract feature names from a fitted ColumnTransformer.

    Useful for feature importance analysis and debugging.
    Must be called after .fit().
    """
    return list(preprocessor.get_feature_names_out())


def prepare_features(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply all pre-pipeline transformations: drop columns, extract date features.
    Returns (X, y) ready for the sklearn.Pipeline.

    This function is called identically for training, validation, test, and serving.
    The only difference: at training time, the pipeline is fitted on X_train.
    At serving time, the frozen pipeline is loaded and .transform(X) is called.

    Args:
        df: DataFrame that may still contain leakage columns (they are dropped here).
        config: Contents of features_config.yaml.

    Returns:
        Tuple of (feature DataFrame X, target Series y).
    """
    target_col = config.get("target_column", "Late_delivery_risk")

    # Step 1: Extract date features BEFORE dropping columns
    date_col = config.get("date_column")
    if date_col and date_col in df.columns:
        df = extract_date_features(df, date_col)

    # Step 2: Drop leakage, PII, ID, and redundant columns
    df = drop_columns(df, config.get("drop_columns", []))

    # Step 3: Separate features from target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    return X, y


def get_column_groups(config: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Read the three feature group lists from features_config.yaml.

    Returns (numeric_cols, onehot_cols, target_enc_cols).
    Date-derived features are added to numeric_cols automatically.
    """
    numeric_cols = list(config.get("numeric_columns", []))
    onehot_cols = list(config.get("onehot_columns", []))
    target_enc_cols = list(config.get("target_enc_columns", []))

    # Date-derived features are numeric after extraction
    date_features = list(config.get("date_derived_features", []))
    numeric_cols = numeric_cols + date_features

    return numeric_cols, onehot_cols, target_enc_cols


def inject_unknown_categories(
    df: pd.DataFrame,
    target_enc_cols: list[str],
    unknown_fraction: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train-only category perturbation to teach behavior for unseen categories.
    """
    if unknown_fraction <= 0:
        return df
    df_copy = df.copy()
    sample_n = int(round(len(df_copy) * unknown_fraction))
    if sample_n <= 0:
        return df_copy
    sampled_idx = df_copy.sample(n=sample_n, random_state=random_state).index
    for col in target_enc_cols:
        if col in df_copy.columns:
            df_copy.loc[sampled_idx, col] = "UNKNOWN"
    return df_copy
