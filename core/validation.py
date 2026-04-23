"""
Data quality checks for the supply chain pipeline.

Pure Python — no ZenML, MLflow, or Evidently imports.
All functions accept a DataFrame and return (passed: bool, errors: list[str]).
The top-level `run_all_checks` composes them into a ValidationReport.

Design principle: fail fast and fail loudly.
Bad data that silently reaches training produces confident but wrong predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    """Result of running all data quality checks."""
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    quality_metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation {status}"]
        for check, ok in self.checks.items():
            icon = "✓" if ok else "✗"
            lines.append(f"  {icon} {check}")
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.quality_metrics:
            lines.append("Quality metrics:")
            for name, value in self.quality_metrics.items():
                lines.append(f"  - {name}: {value:.4f}")
        return "\n".join(lines)


def check_schema(df: pd.DataFrame, expected_columns: list[str]) -> tuple[bool, list[str]]:
    """
    Verify all expected columns are present.

    Why: if a column is renamed upstream (e.g., 'Shipping Mode' → 'ShippingMode'),
    the pipeline uses null for that feature silently. Schema validation catches this
    immediately with a descriptive error instead of a corrupted model.
    """
    missing = set(expected_columns) - set(df.columns)
    if missing:
        return False, [f"Missing columns: {sorted(missing)}"]
    return True, []


def check_target(
    df: pd.DataFrame,
    target_col: str,
    min_positive_rate: float = 0.40,
    max_positive_rate: float = 0.65,
) -> tuple[bool, list[str]]:
    """
    Verify the target column is binary, non-null, and within expected distribution.

    The distribution check guards against labeling errors:
    - If positive_rate → 0%: all orders labeled on-time (labeling failed)
    - If positive_rate → 100%: all orders labeled late (same)
    Both would produce a model that looks good but learned nothing real.
    """
    errors: list[str] = []

    if target_col not in df.columns:
        return False, [f"Target column '{target_col}' not found"]

    null_count = int(df[target_col].isnull().sum())
    if null_count > 0:
        errors.append(f"Target '{target_col}' has {null_count} null values — must be zero")

    unique_vals = set(df[target_col].dropna().unique().tolist())
    if not unique_vals.issubset({0, 1}):
        errors.append(f"Target must be binary (0/1). Found values: {unique_vals}")

    if len(df) > 0:
        positive_rate = float(df[target_col].mean())
        if not (min_positive_rate <= positive_rate <= max_positive_rate):
            errors.append(
                f"Label distribution {positive_rate:.1%} outside expected range "
                f"[{min_positive_rate:.0%}, {max_positive_rate:.0%}]. "
                f"Possible labeling error or wrong dataset."
            )

    return len(errors) == 0, errors


def check_volume(df: pd.DataFrame, min_rows: int = 10_000) -> tuple[bool, list[str]]:
    """
    Verify the dataset has at least min_rows rows.

    Why: a partial CSV read, a failed database export, or an accidental filter
    can silently produce a tiny dataset. The model trains on it without complaint.
    Minimum row count catches this before training wastes compute.
    """
    if len(df) < min_rows:
        return False, [
            f"Dataset has {len(df):,} rows, expected ≥ {min_rows:,}. "
            f"Possible incomplete data pull."
        ]
    return True, []


def check_null_rates(
    df: pd.DataFrame, thresholds: dict[str, float]
) -> tuple[bool, list[str]]:
    """
    Verify null rates don't exceed configured thresholds for critical columns.

    A column that was 1% null during training but is now 40% null indicates
    an upstream data pipeline failure — not a genuine change in the world.
    Catching this before training prevents corrupted feature distributions.
    """
    errors: list[str] = []
    for col, max_null_rate in thresholds.items():
        if col not in df.columns:
            continue
        null_rate = float(df[col].isnull().mean())
        if null_rate > max_null_rate:
            errors.append(
                f"Column '{col}' null rate {null_rate:.1%} exceeds threshold "
                f"{max_null_rate:.1%}. Upstream data pipeline may have failed."
            )
    return len(errors) == 0, errors


def check_dtypes(df: pd.DataFrame, expected_dtypes: dict[str, str]) -> tuple[bool, list[str]]:
    """Check declared dtype families for critical columns."""
    errors: list[str] = []
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            continue
        series = df[col]
        if expected == "numeric" and not pd.api.types.is_numeric_dtype(series):
            errors.append(f"Column '{col}' expected numeric but found {series.dtype}")
        elif expected == "string" and not (
            pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        ):
            errors.append(f"Column '{col}' expected string-like but found {series.dtype}")
    return len(errors) == 0, errors


def check_value_ranges(df: pd.DataFrame, ranges: dict[str, dict[str, float]]) -> tuple[bool, list[str]]:
    """Check numeric columns stay within configured min/max bounds."""
    errors: list[str] = []
    for col, bounds in ranges.items():
        if col not in df.columns:
            continue
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if "min" in bounds and (numeric_series < bounds["min"]).any():
            errors.append(f"Column '{col}' has values below min={bounds['min']}")
        if "max" in bounds and (numeric_series > bounds["max"]).any():
            errors.append(f"Column '{col}' has values above max={bounds['max']}")
    return len(errors) == 0, errors


def check_allowed_categories(
    df: pd.DataFrame, allowed_categories: dict[str, list[str]]
) -> tuple[bool, list[str]]:
    """Check categorical columns against approved category vocabularies."""
    errors: list[str] = []
    for col, allowed in allowed_categories.items():
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str).unique().tolist())
        disallowed = values - set(allowed)
        if disallowed:
            sample = sorted(disallowed)[:10]
            errors.append(f"Column '{col}' contains unknown categories: {sample}")
    return len(errors) == 0, errors


def compute_quality_metrics(df: pd.DataFrame, target_col: str) -> dict[str, float]:
    """
    Lightweight batch quality metrics for observability.
    Completeness is the non-null ratio across all cells.
    """
    total_cells = max(len(df) * max(len(df.columns), 1), 1)
    completeness = 1.0 - (df.isnull().sum().sum() / total_cells)
    target_positive_rate = float(df[target_col].mean()) if target_col in df.columns and len(df) else 0.0
    return {
        "completeness": float(completeness),
        "volume_rows": float(len(df)),
        "target_positive_rate": target_positive_rate,
    }


def run_all_checks(df: pd.DataFrame, config: dict) -> ValidationReport:
    """
    Run all validation checks defined in config.

    Returns a ValidationReport. If .passed is False, the caller should
    raise an exception rather than continuing with corrupted data.

    Args:
        df: Raw DataFrame loaded from the data source.
        config: Contents of data_config.yaml.

    Returns:
        ValidationReport with .passed, .checks, and .errors.
    """
    checks: dict[str, bool] = {}
    all_errors: list[str] = []

    ok, errs = check_schema(df, config.get("expected_columns", []))
    checks["schema"] = ok
    all_errors.extend(errs)

    ok, errs = check_target(
        df,
        config.get("target_column", "Late_delivery_risk"),
        min_positive_rate=config.get("min_positive_rate", 0.40),
        max_positive_rate=config.get("max_positive_rate", 0.65),
    )
    checks["target_integrity"] = ok
    all_errors.extend(errs)

    ok, errs = check_volume(df, min_rows=config.get("min_rows", 10_000))
    checks["volume"] = ok
    all_errors.extend(errs)

    ok, errs = check_null_rates(df, config.get("null_thresholds", {}))
    checks["null_rates"] = ok
    all_errors.extend(errs)

    ok, errs = check_dtypes(df, config.get("expected_dtypes", {}))
    checks["dtypes"] = ok
    all_errors.extend(errs)

    ok, errs = check_value_ranges(df, config.get("value_ranges", {}))
    checks["value_ranges"] = ok
    all_errors.extend(errs)

    ok, errs = check_allowed_categories(df, config.get("allowed_categories", {}))
    checks["allowed_categories"] = ok
    all_errors.extend(errs)

    target_col = config.get("target_column", "Late_delivery_risk")
    quality_metrics = compute_quality_metrics(df, target_col)

    return ValidationReport(
        passed=all(checks.values()),
        checks=checks,
        errors=all_errors,
        quality_metrics=quality_metrics,
    )
