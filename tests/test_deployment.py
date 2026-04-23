"""
Unit tests for deployment shadow comparison helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.deployment import build_shadow_dataframe, summarize_shadow_comparison


def test_summarize_shadow_comparison_metrics() -> None:
    production_scores = np.array([0.1, 0.6, 0.8, 0.2], dtype=float)
    candidate_scores = np.array([0.2, 0.4, 0.9, 0.7], dtype=float)
    summary = summarize_shadow_comparison(
        production_scores=production_scores,
        candidate_scores=candidate_scores,
        decision_threshold=0.5,
    )
    assert summary["n_samples"] == 4
    assert 0.0 <= summary["disagreement_rate"] <= 1.0
    assert "positive_rate_shift" in summary


def test_build_shadow_dataframe_shape_and_columns() -> None:
    order_ids = pd.Series([11, 22, 33])
    production_scores = np.array([0.1, 0.9, 0.2], dtype=float)
    candidate_scores = np.array([0.2, 0.8, 0.8], dtype=float)
    df = build_shadow_dataframe(
        order_ids=order_ids,
        production_scores=production_scores,
        candidate_scores=candidate_scores,
        decision_threshold=0.5,
    )
    assert len(df) == 3
    assert set(
        [
            "order_id",
            "production_score",
            "candidate_score",
            "production_label",
            "candidate_label",
            "label_disagreement",
            "score_delta",
        ]
    ).issubset(df.columns)
