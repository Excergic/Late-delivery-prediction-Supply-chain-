"""
Golden-set parity checks for production hardening.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def parity_report(
    actual_scores: np.ndarray,
    expected_scores: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Return parity diagnostics between actual and expected scores."""
    if len(actual_scores) != len(expected_scores):
        raise ValueError("actual_scores and expected_scores must have equal length")

    deltas = np.abs(actual_scores - expected_scores)
    max_abs_delta = float(np.max(deltas)) if len(deltas) else 0.0
    within_tolerance = bool(np.all(deltas <= tolerance))

    return {
        "n_samples": int(len(actual_scores)),
        "tolerance": float(tolerance),
        "max_abs_delta": max_abs_delta,
        "within_tolerance": within_tolerance,
    }
