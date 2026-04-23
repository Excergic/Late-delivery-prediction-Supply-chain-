"""
Unit tests for parity checks.
"""

from __future__ import annotations

import numpy as np

from core.parity import parity_report


def test_parity_report_within_tolerance_true() -> None:
    actual = np.array([0.1, 0.2, 0.3], dtype=float)
    expected = np.array([0.1000001, 0.1999999, 0.3000001], dtype=float)
    report = parity_report(actual, expected, tolerance=1e-3)
    assert report["within_tolerance"] is True
    assert report["n_samples"] == 3


def test_parity_report_detects_violation() -> None:
    actual = np.array([0.1, 0.2, 0.9], dtype=float)
    expected = np.array([0.1, 0.2, 0.3], dtype=float)
    report = parity_report(actual, expected, tolerance=1e-4)
    assert report["within_tolerance"] is False
    assert report["max_abs_delta"] > 0.0
