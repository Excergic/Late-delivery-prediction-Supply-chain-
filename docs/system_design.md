# System Design: Supply Chain Late Delivery — Data & Prediction Flow

**Scope**: How real order data enters the inference pipeline and how predictions are
delivered to consumers (operations dashboard + automated logistics trigger).
**Date**: 2026-04-24
**ML pipeline internals**: See `architecture.md` — this document covers the system around that pipeline.

---

## Requirements

### Functional

- Inference pipeline reads new, unscored orders from the order management database.
- Pipeline scores each order with the current production model (from MLflow registry).
- Predictions are written back to the database with full audit metadata.
- Operations team queries a dashboard view of high-risk orders.
- Logistics system polls for high-risk orders and triggers automated actions (expedite, notify).

### Non-Functional

| Requirement | Target |
|---|---|
| Prediction freshness | ≤ 1 hour from order placement |
| Availability | Pipeline must self-heal from missed/failed runs |
| Durability | No prediction silently lost — idempotent writes with conflict detection |
| Consistency | Eventually consistent: predictions visible within next batch run |
| Audit trail | Every prediction linked to the model version and ZenML run that produced it |
| Rollback | Bad model version isolated by `model_version` column — filter change is instant |

---

## Estimation

| Metric | Calculation | Result |
|---|---|---|
| Orders/day | 180,519 dataset rows ÷ ~365 days | ~500 orders/day |
| Orders/hour (hourly batch) | 500 ÷ 24 | ~21 orders per batch |
| Inference time | sklearn `transform()` on 21 rows at <1ms/row | < 100ms per run |
| Annual prediction storage | 4 cols × 100 bytes × 500/day × 365 days | ~73 MB/year |
| DB read per batch | 21 rows × 53 cols × ~100 bytes | ~110 KB per query |

**Implication**: This system is tiny by infrastructure standards. A single scheduled pipeline
writing to one database table is the correct architecture. Queues, distributed compute,
caching layers, and horizontal scaling are all over-engineering for this volume.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Order Management Database               │
│                                                          │
│  ┌──────────────────┐    ┌───────────────────────────┐  │
│  │   orders table   │    │    predictions table      │  │
│  │  (existing)      │    │  order_id (FK)            │  │
│  │  order_id  PK    │    │  late_risk_score FLOAT    │  │
│  │  order_date      │    │  predicted_late SMALLINT  │  │
│  │  shipping_mode   │    │  scored_at TIMESTAMP      │  │
│  │  ...53 cols      │    │  model_version TEXT       │  │
│  └──────┬───────────┘    │  pipeline_run_id TEXT     │  │
│         │                └──────────────┬────────────┘  │
└─────────┼──────────────────────────────┼───────────────┘
          │  SELECT unscored orders       │  INSERT predictions
          │  (NOT IN predictions)         │  ON CONFLICT DO NOTHING
          ▼                              │
┌──────────────────────────────────────────────────────────┐
│              Inference Pipeline (ZenML, hourly)          │
│                                                          │
│  load_new_orders()       ← queries DB for unscored rows  │
│  validate_input_data()   ← schema + null checks          │
│  load_production_model() ← MLflow registry "Production"  │
│  apply_pipeline()        ← frozen sklearn artifact       │
│  generate_predictions()  ← late_risk_score, predicted_late│
│  write_predictions()     ─────────────────────────────────┘
└────────────────────┬─────────────────────────────────────
                     │  run logs, model version, row counts
                     ▼
          ┌─────────────────────┐    ┌─────────────────┐
          │  MLflow + ZenML UI  │    │  Slack Alerter  │
          │  (audit trail,      │    │  (pipeline fail,│
          │   run history)      │    │   backlog alert)│
          └─────────────────────┘    └─────────────────┘

┌─────────────────────────────────────────────────────┐
│  Operations Dashboard (reads predictions table)     │
│  SELECT o.*, p.late_risk_score, p.predicted_late    │
│  FROM orders o JOIN predictions p USING (order_id)  │
│  WHERE p.predicted_late = 1                         │
│    AND p.model_version = 'current_production'       │
│  ORDER BY p.late_risk_score DESC                    │
└─────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  Automated Trigger (logistics system polls)      │
│  SELECT * FROM predictions                       │
│  WHERE late_risk_score > 0.8                     │
│    AND scored_at > now() - interval '1h'         │
│  → logistics system takes action (expedite, etc.)│
└──────────────────────────────────────────────────┘
```

### Component Summary

| Component | Purpose | Failure mode |
|---|---|---|
| `orders` table | Source of truth for new orders | Upstream failure; pipeline detects empty result and alerts |
| `predictions` table | Append-only prediction log with audit metadata | Write failure → idempotent retry on next run |
| Inference pipeline | Hourly ZenML job: read → score → write | Self-healing; next run picks up unscored orders |
| MLflow registry | Source of production model artifact | Pipeline fails fast and alerts if registry unreachable |
| Operations dashboard | Query layer over predictions table | Stale data (max 1 hour old); no serving-path dependency |
| Slack alerter | Pipeline failure and backlog notifications | Non-critical; failure doesn't affect predictions |

---

## Deep-Dive: New Order Detection

**Problem**: How does the pipeline identify which orders have not yet been scored?

### Option A — Timestamp watermark
```sql
SELECT * FROM orders WHERE order_date > :last_run_timestamp
```
Fragile: requires external state (last_run_timestamp). If a run fails, the watermark
is stale and orders are skipped until the watermark is manually corrected.

### Option B — Absence check (chosen)
```sql
SELECT * FROM orders
WHERE order_id NOT IN (
    SELECT order_id FROM predictions
    WHERE model_version = :current_version
)
```
Self-healing: any orders missed by a failed run are automatically picked up on the
next scheduled run. No external state required. Naturally supports re-scoring when
a new model version is promoted — the new version simply hasn't scored anything yet.

### Option C — Status column on orders table
```sql
SELECT * FROM orders WHERE scored_at IS NULL
```
Requires ALTER TABLE on the orders table, which is often owned by another team
and a live production table. Avoided.

**Verdict**: Option B. Self-healing, no schema changes to orders, supports model versioning.

**Index required**:
```sql
CREATE INDEX idx_predictions_order_version ON predictions (order_id, model_version);
```

---

## Deep-Dive: Predictions Table Schema and Write Safety

### Schema

```sql
CREATE TABLE predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    order_id        BIGINT NOT NULL REFERENCES orders(order_id),
    late_risk_score FLOAT  NOT NULL CHECK (late_risk_score BETWEEN 0.0 AND 1.0),
    predicted_late  SMALLINT NOT NULL CHECK (predicted_late IN (0, 1)),
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT NOT NULL,        -- MLflow model version tag
    pipeline_run_id TEXT NOT NULL,        -- ZenML run ID for audit trail
    UNIQUE (order_id, model_version)      -- idempotency constraint
);

CREATE INDEX idx_predictions_dashboard ON predictions (predicted_late, scored_at DESC);
CREATE INDEX idx_predictions_lookup    ON predictions (order_id, model_version);
```

### Idempotent Write Pattern

The inference pipeline must be safe to re-run. If it crashes mid-batch, the next
run completes the batch without double-writing.

```sql
INSERT INTO predictions
    (order_id, late_risk_score, predicted_late, scored_at, model_version, pipeline_run_id)
VALUES
    (:order_id, :score, :label, now(), :version, :run_id)
ON CONFLICT (order_id, model_version) DO NOTHING;
```

`DO NOTHING` on conflict means the pipeline is safe to re-run at any time.
The `pipeline_run_id` links every prediction back to the ZenML run ID — every
prediction is fully auditable.

### Model Rollback via model_version

When a bad model is promoted and then rolled back in MLflow:
1. Promote the previous model version back to "Production" in MLflow UI.
2. The next inference run loads the restored version.
3. The dashboard filters by `model_version = 'current_production'` (updated to restored version).
4. Bad predictions are not deleted — they remain with their original `model_version` tag,
   invisible to the dashboard but auditable in the predictions table.

**Target rollback time: < 5 minutes** (same as architecture.md).

---

## Tradeoffs

| Decision | Chosen | Alternative | Why | When alternative wins |
|---|---|---|---|---|
| Predictions storage | Separate `predictions` table | Columns on `orders` table | Full audit trail; supports shadow comparison; no ALTER TABLE on live table | If orders table is fully owned and model history never needed |
| New order detection | Absence check (NOT IN predictions) | Timestamp watermark | Self-healing; no external state to manage | Never at this scale — absence check with index is fast |
| Trigger transport | Logistics system polls predictions table | Webhook / event on each prediction | Zero new infrastructure; polling is trivially cheap at 500 orders/day | Sub-second notification required OR >10K orders/day |
| Scheduling | ZenML cron schedule (hourly) | Event-driven on order INSERT | Batch is simpler, debuggable, appropriate at this volume | >10K orders/day or predictions required within seconds of placement |
| Write pattern | `ON CONFLICT DO NOTHING` | Check-then-insert | Atomic, no race condition, single round-trip | Never — check-then-insert has a TOCTOU race |

---

## Operational Concerns

### Monitoring

**Health signals to track (ZenML + custom logging):**

| Signal | Query / Metric | Alert Threshold |
|---|---|---|
| Orders being scored | `count(predictions WHERE scored_at > now() - 2h)` | Alert if 0 for 2+ consecutive hours |
| Prediction backlog | `count(orders) - count(distinct order_id in predictions)` | Alert if backlog > 500 (1 day of orders) |
| Pipeline run duration | ZenML step duration log | Alert if > 5× rolling average |
| Score distribution | `avg(late_risk_score)` per hourly batch | Alert if shifts > 10% week-over-week (prediction drift) |

The score distribution alert catches model drift before ground truth labels arrive —
a sudden shift in `avg(late_risk_score)` is an early warning that input data has changed.

### Failure Modes

| Failure | Effect | Recovery |
|---|---|---|
| Pipeline crash mid-batch | Partial batch; some orders unscored | Self-healing: next hourly run completes the batch via absence check |
| MLflow registry unreachable | Cannot load production model | Fail fast with clear error; Slack alert; previous predictions remain valid |
| Orders DB read fails | No input data | Fail fast; Slack alert; retry on next schedule |
| Orders DB write fails | Predictions not persisted | Idempotent: re-run writes the batch cleanly |
| Bad model promoted to Production | High false positive/negative rate | Rollback via MLflow UI; bad predictions isolated by model_version |

No circuit breakers required at this scale. Standard retry (3× with exponential
backoff, max 30s) on DB connections is sufficient.

### Deployment

No new services to deploy. Two changes to existing infrastructure:

1. **Create `predictions` table** — DDL migration, backward-compatible, non-breaking.
2. **Add `write_predictions` step to inference pipeline** — replaces current
   "store to CSV" step with a DB write using the schema above.
3. **Configure ZenML schedule** — hourly cron trigger on the inference pipeline.

Rollout order:
1. Run DDL migration (no downtime).
2. Deploy updated inference pipeline (shadow run: write to predictions table AND CSV output, verify parity).
3. Switch dashboard to query predictions table.
4. Remove CSV output step.

### Cost Estimation

| Item | Estimate |
|---|---|
| Compute (inference pipeline) | Negligible — runs in < 1 minute/hour on existing hardware |
| DB storage (predictions table) | ~73 MB/year → essentially free |
| DB queries (read + write per run) | 24 runs/day × 2 queries × ~110 KB = ~5 MB/day |
| MLflow artifact load (per run) | One model artifact pull from local registry per run |

**Total incremental cost**: Near zero. No new infrastructure required.

---

## What Is Not Designed Here

| Item | Location |
|---|---|
| ML pipeline internals (training, evaluation, feature engineering, drift) | `architecture.md` |
| Real-time API serving (stretch goal) | Deferred — design when order volume requires sub-minute scoring |
| Automated retraining trigger | `architecture.md` (deferred to after ≥3 production cycles) |
| Dashboard UI implementation | Tool choice deferred — any tool that can query SQL works (Metabase, Grafana, custom) |
