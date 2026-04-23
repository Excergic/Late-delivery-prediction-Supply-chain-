"""
Model builder functions for the supply chain pipeline.

Pure Python — no ZenML or MLflow imports.

Each function returns an unfitted sklearn-compatible estimator.
All hyperparameters come from training_config.yaml — never hardcoded.

Training order and rationale:
  1. DummyClassifier   — floor: "predict all orders as late" gives ~F2=0.69 for free
  2. LogisticRegression — interpretable baseline: reveals linear signal in features
  3. RandomForest       — ensemble without boosting: shows non-linear signal
  4. LightGBM           — main candidate: best tabular ML at this scale
  5. XGBoost            — optional: try if LightGBM underperforms (not installed by default)

If your LR F2 is within 5% of LightGBM, prefer LR — simpler, faster, interpretable.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_dummy(random_state: int = 42) -> ClassifierMixin:
    """
    Majority-class classifier — the absolute performance floor.

    Predicts the most frequent class for every input.
    For our data (54.8% late), this means flagging every order as late.

    F2 for a majority-class predictor:
      precision = 0.548 (all our "late" predictions that are correct)
      recall    = 1.000 (we catch every actual late order — we flag everything)
      F2 = (5 × 0.548 × 1.0) / (4 × 0.548 + 1.0) ≈ 0.688

    Any real model must beat F2 ≈ 0.69 to justify its complexity.
    """
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def build_logistic_regression(config: dict, random_state: int = 42) -> ClassifierMixin:
    """
    Logistic regression — interpretable linear baseline.

    Coefficient sign tells you direction: positive = more late, negative = less late.
    After training, you can inspect the largest coefficients to understand which
    features most strongly predict late delivery.

    C (inverse regularization): lower C = stronger L2 penalty = smaller weights.
    At C=1.0 with StandardScaler'd features, LR converges reliably on 144K rows.
    """
    # Replace None values (from YAML null) with sklearn defaults
    params = {k: v for k, v in config.items() if v is not None}
    return LogisticRegression(**params, random_state=random_state)


def build_random_forest(config: dict, random_state: int = 42) -> ClassifierMixin:
    """
    Random Forest — ensemble of decorrelated decision trees.

    Shows how much the ensemble effect alone helps, before gradient boosting.
    Each tree is fit on a bootstrap sample with a random feature subset.
    Aggregates by majority vote — naturally handles non-linear interactions.

    min_samples_leaf=5 prevents overfitting on tiny leaf nodes.
    n_jobs=-1 uses all CPU cores — important for 200 trees × 144K rows.
    """
    params = {k: v for k, v in config.items() if v is not None}
    return RandomForestClassifier(**params, random_state=random_state)


def build_lightgbm(config: dict, random_state: int = 42) -> ClassifierMixin:
    """
    LightGBM — gradient boosted trees, leaf-wise growth.

    The best general-purpose model for tabular data at this scale.
    Builds trees sequentially: each tree corrects the errors of the previous ensemble.

    Key hyperparameters:
      learning_rate=0.05: small steps → better generalization, needs more trees
      n_estimators=500:   500 trees at 0.05 learning rate is a standard starting point
      num_leaves=63:      controls tree complexity (≈ 2^6 - 1)
      min_child_samples=20: min examples per leaf, prevents overfitting

    LightGBM is 10-100x faster than XGBoost on many datasets due to histogram binning.
    """
    from lightgbm import LGBMClassifier  # Import here so failure is visible
    params = {k: v for k, v in config.items() if v is not None}
    return LGBMClassifier(**params, random_state=random_state)


def build_xgboost(config: dict, random_state: int = 42) -> ClassifierMixin:
    """
    XGBoost — gradient boosted trees, level-wise growth.

    Optional: only run if lightgbm underperforms or is unavailable.
    Level-wise growth is more regularised than LightGBM's leaf-wise approach,
    which can be an advantage on noisy datasets.

    Raises ImportError if xgboost is not installed.
    """
    from xgboost import XGBClassifier  # Import here — optional dependency
    params = {k: v for k, v in config.items() if v is not None}
    # XGBoost uses 'seed' instead of 'random_state'
    params.pop("random_state", None)
    return XGBClassifier(**params, seed=random_state, eval_metric="logloss")


BUILDERS = {
    "dummy":               build_dummy,
    "logistic_regression": build_logistic_regression,
    "random_forest":       build_random_forest,
    "lightgbm":            build_lightgbm,
    "xgboost":             build_xgboost,
}
