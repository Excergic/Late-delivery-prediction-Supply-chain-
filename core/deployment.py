"""
Deployment safety helpers for shadow comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_shadow_comparison(
    production_scores: np.ndarray,
    candidate_scores: np.ndarray,
    decision_threshold: float,
) -> dict[str, Any]:
    """
    Compute safety metrics for production vs candidate shadow predictions.
    """
    if len(production_scores) != len(candidate_scores):
        raise ValueError("production_scores and candidate_scores must have equal length")

    prod_labels = (production_scores >= decision_threshold).astype(int)
    cand_labels = (candidate_scores >= decision_threshold).astype(int)

    disagreement_rate = float(np.mean(prod_labels != cand_labels))
    mean_score_delta = float(np.mean(candidate_scores - production_scores))
    prod_positive_rate = float(np.mean(prod_labels))
    cand_positive_rate = float(np.mean(cand_labels))

    return {
        "n_samples": int(len(production_scores)),
        "decision_threshold": float(decision_threshold),
        "disagreement_rate": disagreement_rate,
        "mean_score_delta": mean_score_delta,
        "production_positive_rate": prod_positive_rate,
        "candidate_positive_rate": cand_positive_rate,
        "positive_rate_shift": float(cand_positive_rate - prod_positive_rate),
    }


def build_shadow_dataframe(
    order_ids: pd.Series,
    production_scores: np.ndarray,
    candidate_scores: np.ndarray,
    decision_threshold: float,
) -> pd.DataFrame:
    """Return row-level shadow comparison outputs."""
    prod_labels = (production_scores >= decision_threshold).astype(int)
    cand_labels = (candidate_scores >= decision_threshold).astype(int)

    return pd.DataFrame(
        {
            "order_id": order_ids.values,
            "production_score": production_scores,
            "candidate_score": candidate_scores,
            "production_label": prod_labels,
            "candidate_label": cand_labels,
            "label_disagreement": (prod_labels != cand_labels).astype(int),
            "score_delta": candidate_scores - production_scores,
        }
    )
