# Problem Statement: Supply Chain Late Delivery Risk Prediction

## Business Context
At order placement, the logistics team cannot reliably identify which orders
will arrive late. Currently, flagging is manual and experience-based no
formal metric is tracked. 

### The goal 

predict late delivery risk at the moment an order is placed, enabling proactive intervention (expedited shipping, customer notification, priority routing) before the failure happens.

## ML Formulation
- **Problem type**: Binary classification
- **Target variable**: `Late_delivery_risk` (0 = on time, 1 = late)
- **Primary metric**: F2-score — weights recall twice as heavily as precision
  because missing a late delivery (false negative) is more costly than a
  false alarm (false positive).
- **Guardrail metrics**: Recall ≥ 0.80, Precision tracked, AUC-PR curve
- **Current baseline**: Manual / experience-based. No formal metric.
  Our first model becomes the formal baseline.

## Data Summary
- **Rows**: 180,519
- **Features**: 53 columns (post-cleaning: ~35 usable)
- **Label availability**: Yes — `Late_delivery_risk` is pre-labeled
- **Class distribution**: 54.8% late (1), 45.2% on time (0) — near-balanced
- **Known leakage columns** (must exclude):
  - `Delivery Status` — IS the target; encodes late/on-time directly
  - `Days for shipping (real)` — actual days, known only after delivery
  - `shipping date (DateOrders)` — actual ship date, known only after the fact
- **Known useless columns** (drop):
  - `Product Description` — 100% null
  - `Order Zipcode` — 86% null
- **Features to investigate**: `Order Status` (may reflect post-order state — verify before use)

## Constraints
- **Latency**: Prediction at order placement — batch acceptable initially,
  real-time API as stretch goal
- **Interpretability**: Not formally required (open for discussion)
- **Regulatory**: None identified

## Framework
- **Orchestration**: ZenML
- **Experiment tracking**: MLflow (via ZenML integration)

## Success Criteria
The model is in production when:
1. F2-score ≥ 0.75 on held-out test set (to be revised after first model run)
2. Recall ≥ 0.80 — we catch at least 80% of truly late deliveries
3. Pipeline is reproducible — same data + same code = same result
4. Predictions are served on new orders without manual intervention
