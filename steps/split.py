"""
Data splitting step: stratified 80/10/10 train/val/test split.

Why stratify?
Our target is 54.8% positive. Without stratification, random chance could give
train=56% positive and test=51% positive. Every metric computed on the test set
would reflect a different class balance than what the model trained on — making
comparisons misleading. Stratification locks the positive rate at ~54.8% in all
three splits.

Why split BEFORE feature engineering?
If we fit the StandardScaler on the full dataset (all 180K rows), the training
features have been scaled using statistics that include test set information.
The model indirectly "sees" the test set during training. This is data leakage.
Split first. Fit on train only. Transform train, val, test separately.
"""


from typing import Annotated

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from zenml import step


@step
def split_data(
    df: pd.DataFrame,
    config_path: str = "configs/data_config.yaml",
) -> tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "val_df"],
    Annotated[pd.DataFrame, "test_df"],
]:
    """
    Stratified train/val/test split.

    Split fractions and random state come from data_config.yaml.
    Default: 80% train, 10% val, 10% test, seed=42.

    Stratification column: Late_delivery_risk (the target).

    Args:
        df: Validated raw DataFrame from validate_data step.
        config_path: Path to data_config.yaml.

    Returns:
        Three DataFrames: train_df, val_df, test_df.
        All still contain the target column — it's separated in engineer_features.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    target_col = config.get("target_column", "Late_delivery_risk")
    train_frac = config.get("train_frac", 0.80)
    val_frac = config.get("val_frac", 0.10)
    test_frac = config.get("test_frac", 0.10)
    random_state = config.get("random_state", 42)

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got {total:.6f}. "
            f"Check configs/data_config.yaml."
        )

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        stratify=df[target_col],
        random_state=random_state,
    )

    # Second split: val vs test (equal halves of temp)
    # val_frac relative to (val + test) portion = val_frac / (1 - train_frac)
    val_relative = val_frac / (1.0 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative),
        stratify=temp_df[target_col],
        random_state=random_state,
    )

    print("Split complete:")
    print(f"  Train: {len(train_df):>7,} rows  ({train_df[target_col].mean():.1%} late)")
    print(f"  Val:   {len(val_df):>7,} rows  ({val_df[target_col].mean():.1%} late)")
    print(f"  Test:  {len(test_df):>7,} rows  ({test_df[target_col].mean():.1%} late)")

    return train_df, val_df, test_df
