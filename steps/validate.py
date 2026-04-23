"""
Data validation step: runs quality checks before any processing begins.

Fail fast. If this step raises, the pipeline stops here.
No downstream step sees bad data. No corrupted model gets trained.
"""


import pandas as pd
import yaml
from zenml import step

from core.validation import run_all_checks


@step
def validate_data(
    df: pd.DataFrame,
    config_path: str = "configs/data_config.yaml",
) -> pd.DataFrame:
    """
    Run all data quality checks against the raw DataFrame.

    Checks:
    - Schema: all 53 expected columns present
    - Target integrity: binary, non-null, distribution within 40-65%
    - Volume: at least 10,000 rows
    - Null rates: critical columns within configured thresholds

    Args:
        df: Raw DataFrame from ingest_data step.
        config_path: Path to data_config.yaml.

    Returns:
        The same DataFrame, unchanged. Validation is a gate, not a transform.

    Raises:
        ValueError: If any validation check fails, with a full error report.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    report = run_all_checks(df, config)
    print(report.summary())

    if not report.passed:
        raise ValueError(
            f"Data validation FAILED. Fix the issues above before training.\n"
            f"Failed checks: {[k for k, v in report.checks.items() if not v]}"
        )

    return df
