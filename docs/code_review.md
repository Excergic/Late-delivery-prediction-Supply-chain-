# Code Review: Feature Engineering Changes

**Reviewed**: `core/preprocessing.py`, `steps/engineer_features.py`, `tests/test_preprocessing.py`, `configs/features_config.yaml`
**Date**: 2026-04-24
**Reviewer**: MLOps Code Review Co-Pilot

---

## Critical Findings (0)

No data leakage, no training-serving skew, no security issues, no silent failures on the hot path.

---

## Major Findings (3)

### MJ-1: Relative config path in ZenML step will fail in remote orchestrators
**File**: `steps/engineer_features.py:40`
**Issue**: `features_config_path: str = "configs/features_config.yaml"` — when ZenML runs steps remotely (Kubernetes, Vertex AI, etc.), CWD is not guaranteed. This will silently work locally but raise `FileNotFoundError` in cloud runs.
**Fix**: Resolve default relative to the repo root using `Path(__file__).parents[1] / "configs/features_config.yaml"`.
**Status**: Fixed.

### MJ-2: Missing test coverage for several code branches
**File**: `tests/test_preprocessing.py`
**Issue**: `get_column_groups`, `load_features_config`, `inject_unknown_categories(fraction=0)`, `robust` scaler, and the "date column absent" path in `prepare_features` are all untested. Any regression in these branches will be invisible.
**Fix**: Added targeted tests for each branch.
**Status**: Fixed.

### MJ-3: `print()` in ZenML step
**File**: `steps/engineer_features.py:115-121`
**Issue**: `print()` produces unstructured output that cannot be filtered by log level, redirected to log aggregators, or silenced in tests.
**Fix**: Replaced with `logging.getLogger(__name__)` and `logger.info(...)`.
**Status**: Fixed.

---

## Minor Findings (2)

### MN-1: Incomplete type annotation on `config` parameter
**Files**: `core/preprocessing.py:166`, `core/preprocessing.py:200`
`config: dict` should be `config: dict[str, Any]` for consistency with `load_features_config` return type.
**Status**: Open — fix when next touching these functions.

### MN-2: Misleading comment in test
**File**: `tests/test_preprocessing.py:165`
`# Use different rng to get slightly different data` is not true — `make_sample_df` re-seeds with 42 each call. Deleted.
**Status**: Fixed.

---

## Positive Patterns

- **Leakage prevention is explicit and enforced**: Module docstring, function docstrings, and YAML comments all state the rule — fit on train, transform val/test. The step implements this correctly.
- **Config-driven column management**: Adding or moving a column only requires a YAML edit. Reason codes (`LEAKAGE`, `PII`, `ID`) in the config are production-grade documentation.
- **Immutability**: `extract_date_features` uses `df.copy()` and has a test asserting the input is not mutated.
- **`inject_unknown_categories` correctly excluded from the pipeline artifact**: Runs before `fit_transform` but is not a pipeline stage — serving never re-injects unknowns.
- **`remainder='drop'`**: Silent drop of unexpected serving-time columns is the right default for a production system.

---

## Summary

| Severity | Count | Status |
|---|---|---|
| Critical | 0 | — |
| Major | 3 | All fixed |
| Minor | 2 | MN-2 fixed; MN-1 open |

**Verdict: PASS WITH CONDITIONS → PASS (after fixes)**
