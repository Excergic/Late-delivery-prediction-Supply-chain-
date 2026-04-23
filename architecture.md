# Architecture: Supply Chain Late Delivery Risk Prediction

## MLOps Pipeline Overview

**A production ML system is not a model — it is a system of pipelines.**

### Ten Production Stages

| Stage | Purpose | Failure When Missing |
|---|---|---|
| 1. Data Ingestion | Read raw CSV, load orders into pipeline | Ad-hoc pulls differ between train and serve |
| 2. Data Validation | Schema checks, null gates, distribution checks | Garbage data flows silently into training |
| 3. Feature Engineering | Drop leakage, encode, scale, extract date features | Training-serving skew — most common silent bug |
| 4. Model Training | Fit models, track experiments in MLflow | Cannot reproduce results or compare runs |
| 5. Model Evaluation | Measure F2, Recall, slice performance | Deploying models worse than current system |
| 6. Model Registry | Version models with metadata + promotion status | No rollback path, no audit trail |
| 7. Deployment | Batch inference pipeline, score new orders | Inconsistent predictions, reckless updates |
| 8. Monitoring | Track F2 and recall over time in production | Model degrades for weeks, nobody notices |
| 9. Drift Detection | Compare current vs training distributions | Stale model, no early warning |
| 10. Retraining Trigger | Decide when and how to retrain | Always reactive, always late |

### Five Pipelines

```
Training Pipeline      → stages 1–6: data → features → train → evaluate → register
Inference Pipeline     → load model → apply transformations → predict → store results
Drift Detection        → compare current feature distributions vs training baseline
Monitoring Pipeline    → compute rolling F2/recall on labeled data, fire Slack alerts
Retraining Pipeline    → training pipeline + human-approval promotion gate
```

### Maturity Target

| Level | Status |
|---|---|
| Level 0 — Manual | Starting point (notebooks) |
| Level 1 — Pipeline Automation | **Build now** (reproducible pipelines, versioned models) |
| Level 2 — CI/CD for ML | **Designed-in** (code structure enables it; add CI later) |
| Level 3 — Full Automation | Deferred — earn it after evaluation gates are proven |

---

## Data Plan

**Source:** `/data/DataCoSupplyChainDataset.csv`
**Size:** 180,519 rows × 53 columns
**Compliance:** Customer PII already masked (XXXXXXXXX)
**Versioning:** ZenML artifact store versions the dataset automatically on every pipeline run

**Validation gates (fail fast — hard stop if any fails):**
- All 53 expected columns present
- `Late_delivery_risk` is binary (0/1 only, no nulls)
- Row count ≥ 10,000
- Label distribution between 40%–65% positive (guards against labeling errors)
- Null rate on critical features below defined thresholds

**Train/Val/Test split:** 80/10/10 stratified by `Late_delivery_risk`
**Rationale:** Near-balanced classes (~55/45), stratification ensures split proportions match overall distribution.

---

## Feature Engineering Plan

### Columns to Drop (23 total)

| Reason | Columns |
|---|---|
| **Leakage** (4) | `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)`, `Order Status` |
| **PII** (5) | `Customer Email`, `Customer Fname`, `Customer Lname`, `Customer Password`, `Customer Street` |
| **100% null** (1) | `Product Description` |
| **86% null** (1) | `Order Zipcode` |
| **Pure IDs** (6) | `Order Id`, `Customer Id`, `Order Customer Id`, `Order Item Id`, `Product Card Id`, `Order Item Cardprod Id` |
| **Irrelevant** (1) | `Product Image` |
| **Redundant** (3) | `Category Id`, `Department Id`, `Product Category Id` |
| **Too high cardinality** (2) | `Order City` (3,597), `Customer City` (563) |
| **Redundant geography** (1) | `Order State` (captured by `Order Region`) |

### Remaining Features (~30)

**Numeric → `SimpleImputer(median)` → `StandardScaler`:**
`Days for shipment (scheduled)` ⭐, `Benefit per order`, `Sales per customer`,
`Order Item Discount`, `Order Item Discount Rate`, `Order Item Product Price`,
`Order Item Profit Ratio`, `Order Item Quantity`, `Sales`, `Order Item Total`,
`Order Profit Per Order`, `Product Price`, `Latitude`, `Longitude`, `Product Status`

**Low-cardinality categorical → `SimpleImputer('UNKNOWN')` → `OneHotEncoder`:**
`Type` (4), `Customer Segment` (3), `Shipping Mode` (4) ⭐, `Market` (5),
`Department Name` (11), `Customer Country` (2), `Customer State` (46)

**High-cardinality categorical → `SimpleImputer('UNKNOWN')` → `TargetEncoder (out-of-fold)`:**
`Category Name` (50), `Order Region` (23), `Product Name` (118), `Order Country` (164)

**Date-derived features (from `order date (DateOrders)`):**
`order_hour`, `order_day_of_week`, `order_month`, `order_is_weekend`

### Preprocessing Pipeline (prevents train-serve skew)

```python
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), NUMERIC_COLS),
        ('onehot', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), ONEHOT_COLS),
        ('target', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('encoder', TargetEncoder())
        ]), TARGET_ENC_COLS),
    ])),
    ('model', <model>)
])
```

**Critical principle:** The entire pipeline (imputer stats, scaler mean/variance, encoder mappings) is saved as a single ZenML artifact. At serving time, the same frozen artifact is loaded — zero chance of training-serving skew.

**Feature store:** Not needed. sklearn.Pipeline handles train-serve parity. Feature stores are for multi-team feature reuse — not applicable here.

---

## Training & Evaluation Plan

### Training Order

| Step | Model | Purpose |
|---|---|---|
| 1 | **Dummy Classifier** (majority class) | Absolute floor — anything below this is broken |
| 2 | **Logistic Regression** | Interpretable baseline — reveals linear signal |
| 3 | **Random Forest** | Ensemble baseline — shows non-linear without gradient boosting |
| 4 | **LightGBM** | Main candidate — best tabular model for this scale |
| 5 | **XGBoost** | Optional comparison if LightGBM underperforms |

### Experiment Tracking (MLflow via ZenML)

Every run logs:
- All hyperparameters
- F2-score, Recall, Precision, AUC-PR, AUC-ROC (on val + test sets)
- Git commit hash
- Dataset artifact version (ZenML artifact ID)
- Training duration
- Full sklearn.Pipeline artifact (model + all preprocessors)

### Evaluation Strategy

- **Primary metric:** F2-score (weights recall 2× over precision)
- **Guardrail:** Recall ≥ 0.80
- **Tracked:** Precision, AUC-PR, AUC-ROC
- **Slice evaluation:** F2 and Recall broken down by `Shipping Mode` and `Market`
  — a model with 0.85 overall recall but 0.45 recall for Same Day shipments is a product failure
- **Baseline comparison:** Every model compared against Dummy Classifier
- **Hyperparameter tuning:** Default parameters first; grid search over 10–15 configs if needed
- **Test set rule:** Touch the test set exactly once — final evaluation only, never during development

### Why Not Neural Networks?

180K rows, ~30 features. Gradient boosting consistently outperforms neural networks on tabular
data at this scale. Neural networks become competitive above ~10M rows or with high-dimensional
embeddings. Neither applies here.

---

## Deployment Plan

**Serving mode:** Batch inference pipeline
**Schedule:** Configurable (hourly or daily based on order volume)
**Input:** New orders since last run
**Output:** `order_id | late_delivery_risk_score (0-1) | predicted_late (0/1) | scored_at`

### Model Promotion Workflow

```
Training run completes
     → Evaluation gates pass (F2 > threshold, Recall ≥ 0.75)
     → Model automatically lands in MLflow "Staging"
     → Human review: compare metrics vs current Production in MLflow UI
     → Human approves → promoted to "Production"
     → Previous "Production" → "Archived" (retained 90 days)
```

### Model Update Strategy

- **First deployment:** Direct (no existing model to compare against)
- **Subsequent updates:** Shadow comparison — run new + old model on same batch, compare F2/recall
  before human promotion decision. No extra infrastructure since it's batch.

### Rollback Plan

Promote the archived previous version back to "Production" via MLflow UI.
**Target: rollback in under 5 minutes.** This is designed before deployment, not improvised during incidents.

**Rollback triggers:**
- F2-score drops >5% vs previous production model
- Recall drops below 0.75
- Prediction distribution collapses (>20% shift in scored-late rate)
- Golden input test failure (fixed test cases with known expected outputs)

---

## Monitoring & Drift Plan

### Data Drift Detection (Evidently)

**Statistical tests:**
- Numeric features: Kolmogorov-Smirnov test (p-value threshold: 0.05)
- Categorical features: Chi-squared test (p-value threshold: 0.05)

**Priority features to watch:**
- `Shipping Mode` distribution — carrier mix changes alter late delivery patterns
- `Order Region` distribution — new markets, model has never seen them
- `Days for shipment (scheduled)` — carrier SLA changes shift this distribution
- `Market` distribution — regional demand shifts

### Prediction Drift

Track distribution of predicted probabilities over time.
Alert if "% orders flagged as late" shifts by >10% week-over-week.
This often signals upstream data change before ground truth labels arrive.

### Performance Monitoring

Ground truth (`Late_delivery_risk` actuals) arrives after delivery completes — days after order placement.

Once actuals arrive, compute:
- Rolling 30-day F2-score and Recall on scored orders
- Compare against production model's baseline metrics

### Retraining Triggers

| Trigger | Threshold | Action |
|---|---|---|
| Performance drop | F2-score < 0.70 on 30-day window | Slack alert + manual retrain decision |
| Multi-feature drift | ≥ 3 key features show significant drift | Slack alert + investigation |
| Scheduled | Quarterly (every 3 months) | Retrain regardless of drift |

**Tool:** Evidently AI (ZenML integration via data validator component)
**Alerting:** ZenML Slack alerter → configured Slack channel

---

## Versioning & Governance

### Model Registry (MLflow)

Three stages: `Staging` → `Production` → `Archived`

**Per-version metadata (audit trail):**
- ZenML artifact ID of training data
- Git commit hash of pipeline code
- Hyperparameters (all, logged to MLflow)
- Val + test metrics (F2, Recall, Precision, AUC-PR)
- Training timestamp
- Who promoted to Production + when
- What triggered archival

### Configuration Versioning

All hyperparameters in `configs/` as YAML files, committed to git alongside code.
Every pipeline run references a specific config file — config is versioned with code.

### Governance

**Human-in-the-loop promotion:** No model reaches Production without explicit human approval.
**Rationale:** First production system. Fully automated promotion is earned after evaluation gates are proven trustworthy over multiple iterations.

---

## ZenML Stack Specification

**ZenML Deployment Mode:** Local OSS Server (`zenml login --local`)
Gives: ZenML dashboard, Model Control Plane, Model stage transitions
Upgrade path: OSS Server on Docker → Pro SaaS when team grows or cloud is needed

| Component | Choice | Why |
|---|---|---|
| **Orchestrator** | Local | Solo dev, macOS, no containerization overhead needed |
| **Artifact Store** | Local | 180K rows, local filesystem is sufficient |
| **Experiment Tracker** | MLflow (local) | Industry standard, best ZenML integration, comparison UI |
| **Data Validator** | Evidently | Drift detection + quality reports, clean ZenML integration |
| **Model Registry** | MLflow | Staging/Production/Archived transitions, human promotion via UI |
| **Alerter** | Slack | Pipeline failures, drift alerts, retraining notifications |
| **Model Deployer** | Not included | Batch serving doesn't need a model endpoint |
| **Container Registry** | Not included | Local orchestrator doesn't containerize steps |
| **Step Operator** | Not included | CPU training, no GPU or distributed compute needed |
| **Feature Store** | Not included | sklearn.Pipeline handles train-serve parity |
| **Service Connector** | Not included | Local stack, no cloud credentials to manage |

### Deferred Components (when to add)

| Component | Add When |
|---|---|
| Container Registry + K8s Orchestrator | Moving to cloud or CI/CD pipeline |
| Model Deployer (BentoML) | Switching to real-time API serving |
| Service Connector (AWS/GCP) | Artifact store moves to S3/GCS |
| Image Builder | Containerizing pipeline steps |
| Automated Retraining Pipeline | Evaluation gates proven trustworthy over ≥3 production cycles |

---

## Pipeline Decomposition

### Training Pipeline (steps 1–6)

```
ingest_data
    → validate_data         [Evidently schema + distribution checks]
    → split_data            [80/10/10 stratified]
    → engineer_features     [build sklearn.Pipeline, fit on train only]
    → train_model           [one step per algorithm: LR, RF, LightGBM, XGBoost]
    → evaluate_model        [F2, Recall, slice metrics on val + test]
    → register_model        [log to MLflow, tag as Staging]
```

### Inference Pipeline

```
load_new_orders
    → validate_input_data   [schema check, null check]
    → load_production_model [from MLflow registry, "Production" stage]
    → apply_pipeline        [load frozen sklearn.Pipeline artifact]
    → generate_predictions  [late_delivery_risk_score, predicted_late]
    → store_predictions     [CSV or database output]
```

### Drift Detection Pipeline

```
load_reference_profile      [training set feature distributions, saved as artifact]
    → load_current_window   [last N orders since last detection run]
    → compute_drift_report  [Evidently: KS + Chi-squared per feature]
    → evaluate_thresholds   [flag features with p-value < 0.05]
    → alert_if_needed       [Slack: "3 features drifted significantly"]
```

### Monitoring Pipeline (runs after ground truth arrives)

```
load_scored_orders          [orders with predictions]
    → join_ground_truth     [actual Late_delivery_risk values]
    → compute_metrics       [rolling 30-day F2, Recall, Precision]
    → compare_vs_baseline   [alert if drop > threshold]
    → update_dashboard      [MLflow metrics log]
    → alert_if_needed       [Slack: "Recall dropped to 0.72"]
```

---

## Project Structure

```
supply-chain-late-delivery/
├── data/
│   └── DataCoSupplyChainDataset.csv
├── configs/
│   ├── data_config.yaml          # column lists, validation thresholds
│   ├── features_config.yaml      # feature groups, encoding choices
│   ├── training_config.yaml      # hyperparameters per model
│   └── deployment_config.yaml    # batch schedule, output path
├── core/                         # Pure Python — no framework imports
│   ├── __init__.py
│   ├── preprocessing.py          # Feature engineering, pipeline building
│   ├── validation.py             # Schema checks, distribution checks
│   └── evaluation.py             # Metric computation, slice evaluation
├── steps/                        # ZenML steps — thin wrappers over core/
│   ├── ingest.py
│   ├── validate.py
│   ├── split.py
│   ├── engineer_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── register.py
│   ├── drift_detect.py
│   └── inference.py
├── pipelines/
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   └── drift_pipeline.py
├── tests/
│   ├── test_preprocessing.py     # No ZenML needed — imports from core/
│   ├── test_validation.py
│   └── test_evaluation.py
├── problem_statement.md
├── architecture.md               # This file
├── pyproject.toml
└── README.md
```

**Critical design principle — `core/` module:**
All pure business logic lives in `core/` with no framework imports.
Steps in `steps/` are thin wrappers that call `core/` functions.
Tests import from `core/` only — no ZenML needed in the test environment.
This makes framework migration cheap and keeps tests fast.

---

## MVP Scope

**What we build first (Phase 3, Implementation):**
1. Project setup + ZenML stack configuration
2. Data loading + validation step
3. EDA + feature understanding
4. Feature engineering pipeline (sklearn.Pipeline)
5. Training pipeline: Logistic Regression baseline + LightGBM
6. Evaluation + MLflow tracking
7. Batch inference pipeline
8. Drift detection pipeline (Evidently)
9. Monitoring setup + Slack alerts

**Success criteria for MVP:**
- F2-score ≥ 0.75 on held-out test set
- Recall ≥ 0.80
- Full training pipeline runs with one command
- Any past experiment is reproducible from MLflow run ID
- Rollback is a single command (< 5 minutes)

---

## Deferred Items

| Item | Why Deferred | When to Add |
|---|---|---|
| Real-time API serving | Batch is sufficient for MVP | When order volume requires sub-minute scoring |
| Automated retraining | Need to trust evaluation gates first | After ≥3 production cycles with stable metrics |
| Cloud infrastructure | Local is sufficient for solo dev | When team grows or data scale demands it |
| CI/CD pipeline | Level 2 maturity | After Level 1 is proven stable |
| Cross-validation | Holdout sufficient at 180K rows | If model performance is unstable across runs |
| XGBoost | LightGBM usually sufficient | Only if LightGBM underperforms significantly |
