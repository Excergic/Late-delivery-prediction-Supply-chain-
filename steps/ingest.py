"""
Data ingestion step: reads the raw CSV and returns a DataFrame.

This step is intentionally minimal — single responsibility.
All validation happens in the separate validate step.
All feature engineering happens in the separate engineer_features step.

Why separate ingestion from validation?
If we combined them, a validation failure would appear as an "ingestion error"
in the pipeline logs. Separated, the ZenML UI shows exactly which check failed.
"""


from pathlib import Path

import pandas as pd
import yaml
from zenml import step


@step
def ingest_data(config_path: str = "configs/data_config.yaml") -> pd.DataFrame:
    """
    Load the raw supply chain dataset from CSV.

    Args:
        config_path: Path to data_config.yaml, relative to project root.

    Returns:
        Raw DataFrame with all original columns intact.
        Shape: (180_519, 53) for the full DataCo dataset.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = Path(config["data_path"])
    encoding = config.get("encoding", "utf-8")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Make sure the CSV is at {data_path.resolve()}"
        )

    df = pd.read_csv(data_path, encoding=encoding)

    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {data_path}")
    return df
