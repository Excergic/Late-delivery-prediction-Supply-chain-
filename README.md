# Supply Chain Late Delivery Risk Prediction

A production-grade MLOps system that predicts late delivery risk **at the moment an order is placed**, enabling proactive intervention — expedited shipping, customer notification, priority routing — before the failure happens.

## Problem

The logistics team cannot reliably identify which orders will arrive late. Current flagging is manual and experience-based with no formal metric tracked. This project replaces that with a binary classifier trained on 180,519 historical orders.

**ML formulation:**
- Task: Binary classification (`Late_delivery_risk`: 0 = on time, 1 = late)
- Primary metric: **F2-score** — recall weighted 2× over precision (missing a late delivery costs more than a false alarm)
- Guardrails: Recall ≥ 0.80, Precision tracked, AUC-PR curve
- Class distribution: 54.8% late / 45.2% on time (near-balanced)

**Success criteria:**
1. F2-score ≥ 0.75 on held-out test set
2. Recall ≥ 0.80 — catch at least 80% of truly late deliveries
3. Pipeline is reproducible — same data + same code = same result
4. Predictions served on new orders without manual intervention

## Architecture

Five ZenML pipelines cover the full ML lifecycle:

```
Training Pipeline   → ingest → validate → EDA → split → features → train → evaluate → register
Inference Pipeline  → load model → preprocess → predict → store results
Drift Pipeline      → compare current feature distributions vs training baseline
Monitoring Pipeline → compute rolling F2/recall on labeled data, fire alerts
Shadow Pipeline     → run production + candidate model on same batch, compare before promotion
```

**Stack:**
| Component | Choice |
|---|---|
| Orchestration | ZenML (local) |
| Experiment tracking | MLflow |
| Model registry | MLflow (Staging → Production → Archived) |
| Drift detection | Evidently |
| Preprocessing | sklearn.Pipeline (prevents training-serving skew) |
| Models | Dummy → Logistic Regression → Random Forest → LightGBM → XGBoost |

## Project Structure

```
.
├── data/
│   └── DataCoSupplyChainDataset.csv     # Source dataset (180,519 rows × 53 cols)
├── configs/
│   ├── data_config.yaml                 # Column lists, validation thresholds
│   ├── features_config.yaml             # Feature groups, encoding choices
│   ├── training_config.yaml             # Hyperparameters per model
│   └── deployment_config.yaml           # Batch schedule, drift thresholds, output paths
├── core/                                # Pure Python — no framework imports
│   ├── preprocessing.py                 # Feature engineering, pipeline building
│   ├── validation.py                    # Schema checks, distribution checks
│   ├── evaluation.py                    # Metric computation, slice evaluation
│   ├── training.py                      # Model training logic
│   ├── drift.py                         # Drift detection (KS, Chi-squared, PSI)
│   ├── monitoring.py                    # Rolling F2/recall computation
│   ├── deployment.py                    # Model loading from MLflow registry
│   ├── parity.py                        # Golden-set score comparison
│   └── eda.py                           # EDA diagnostics
├── steps/                               # ZenML steps — thin wrappers over core/
│   ├── ingest.py
│   ├── validate.py
│   ├── eda.py
│   ├── split.py
│   ├── engineer_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── register.py
│   ├── inference.py
│   ├── drift_detect.py
│   ├── monitor.py
│   └── shadow_compare.py
├── pipelines/
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   ├── drift_pipeline.py
│   ├── monitoring_pipeline.py
│   └── shadow_pipeline.py
├── scripts/
│   ├── golden_parity_check.py           # Verify model scores match expected outputs
│   └── rollback_production.py           # Reassign production alias to a prior version
├── tests/                               # Unit tests — import from core/ only, no ZenML
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   ├── test_evaluation.py
│   ├── test_drift.py
│   ├── test_monitoring.py
│   ├── test_deployment.py
│   ├── test_parity.py
│   ├── test_eda.py
│   └── test_pipeline_smoke.py
├── docs/
│   ├── system_design.md
│   ├── code_review.md
│   ├── ship_checklist.md
│   ├── runbooks/
│   │   ├── incident_response.md
│   │   └── rollback_runbook.md
│   └── checklists/
│       └── anti-slop-checklist.md
├── problem_statement.md
├── architecture.md
├── pyproject.toml
└── run_training.py                      # Shortcut entry point for the training pipeline
```

## Setup

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the repository

```bash
git clone <repo-url>
cd Late-delivery-prediction-Supply-chain-
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

This installs the project in editable mode with all runtime dependencies (ZenML, MLflow, Evidently, scikit-learn, LightGBM, XGBoost, pandas, numpy) plus dev tools (ruff, mypy, pre-commit).

### 4. Install pre-commit hooks

```bash
pre-commit install
```

Hooks run `pytest --strict-markers -x`, `ruff check`, and `mypy` before every commit.

### 5. Initialize ZenML

```bash
zenml login --local
zenml init
```

This starts a local ZenML server and initializes the artifact store. Open the ZenML dashboard at `http://127.0.0.1:8237` to browse pipeline runs and artifacts.

### 6. Add the dataset

Download the DataCo Supply Chain dataset and place it at:

```
data/DataCoSupplyChainDataset.csv
```

The file must be encoded in `latin-1`. The validation step checks for all 53 expected columns before training proceeds.

---

## Running the Pipelines

### Training

Trains Dummy Classifier → Logistic Regression → Random Forest → LightGBM, selects the best model by validation F2-score, evaluates on the held-out test set, and registers the model in MLflow.

```bash
python run_training.py
```

Or invoke the pipeline directly:

```bash
python pipelines/training_pipeline.py
```

After the run, open MLflow UI to review metrics and promote a model to production:

```bash
mlflow ui --port 5000
# Open http://127.0.0.1:5000
# Navigate: Models → supply-chain-late-delivery → promote "staging" → "production"
```

### Batch Inference

Scores new orders using the model tagged with a given alias (`staging` by default; change to `production` after promotion):

```bash
python pipelines/inference_pipeline.py
```

Output is written to `data/predictions.csv` with columns:
`order_id | late_delivery_risk_score | predicted_late | scored_at`

To score against a specific alias:

```bash
python -c "
from pipelines.inference_pipeline import inference_pipeline
inference_pipeline(model_alias='production')
"
```

### Drift Detection

Compares the current order window against the training reference distribution using Kolmogorov-Smirnov (numeric) and Chi-squared (categorical) tests. Alerts if ≥ 3 key features drift significantly (p-value < 0.05).

```bash
python pipelines/drift_pipeline.py
```

Requires two CSV files defined in `configs/deployment_config.yaml`:
- `data/reference_window.csv` — feature distributions from training time
- `data/current_window.csv` — recent orders to check

### Monitoring

Computes rolling 30-day F2-score and recall after ground-truth labels arrive (i.e., after deliveries complete). Alerts if performance drops below thresholds.

```bash
python pipelines/monitoring_pipeline.py
```

Requires:
- `data/predictions.csv` — previous inference output
- `data/labeled_predictions.csv` — same rows with `Late_delivery_risk` actuals joined in

### Shadow Comparison

Runs both the production and staging model on the same input batch and compares their predictions before any promotion decision. Flags if disagreement rate exceeds 10%.

```bash
python pipelines/shadow_pipeline.py
```

---

## Utility Scripts

### Golden Parity Check

Verifies that model scores on a fixed reference set match expected values within tolerance. Use after any code or preprocessing change to confirm serving logic is unchanged.

```bash
python scripts/golden_parity_check.py \
  --input data/golden_parity_input.csv \
  --tolerance 1e-6 \
  --output data/golden_parity_report.json
```

Input CSV must have columns `actual_score` and `expected_score`. Optionally include `actual_label` and `expected_label` for label match rate.

### Rollback Production

Reassigns the `production` alias in MLflow to any prior version. Target rollback time: under 5 minutes.

```bash
python scripts/rollback_production.py \
  --model-name supply-chain-late-delivery \
  --to-version 12
```

---

## Quality Gates

All changes must pass before merging:

```bash
pytest --strict-markers -x       # Full test suite, fail fast
python -m ruff check .           # Linting
python -m mypy .                 # Type checking
```

These run automatically as pre-commit hooks. Never skip them with `--no-verify`.

---

## Configuration

All pipeline behaviour is controlled by YAML files in `configs/` — no magic constants in source code.

| File | Controls |
|---|---|
| [configs/data_config.yaml](configs/data_config.yaml) | Data path, expected columns, null thresholds, validation gates, split fractions |
| [configs/features_config.yaml](configs/features_config.yaml) | Feature groups, encoding strategy per column |
| [configs/training_config.yaml](configs/training_config.yaml) | Hyperparameters per model, F2/recall registration thresholds |
| [configs/deployment_config.yaml](configs/deployment_config.yaml) | Inference I/O paths, drift thresholds, monitoring thresholds, shadow comparison settings |

---

## Model Promotion Workflow

```
Training run completes
  → Evaluation gates pass (F2 > threshold, Recall ≥ 0.75)
  → Model lands in MLflow "Staging"
  → Human reviews metrics vs current Production in MLflow UI
  → Human approves → promoted to "Production"
  → Previous "Production" → "Archived" (retained 90 days)
```

No model reaches Production without explicit human approval. Fully automated promotion is deferred until evaluation gates are proven trustworthy over multiple iterations.

**Rollback triggers:**
- F2-score drops > 5% vs previous production model
- Recall drops below 0.75
- Prediction distribution collapses (> 20% shift in scored-late rate)
- Golden parity check fails on fixed reference inputs

---

## Key Design Decisions

**`core/` is framework-free.** All business logic (preprocessing, evaluation, drift, monitoring) lives in `core/` with zero ZenML or MLflow imports. Steps in `steps/` are thin wrappers. Tests import only from `core/` — fast, no ZenML environment needed.

**Single pipeline artifact.** The entire sklearn.Pipeline (imputer stats, scaler parameters, encoder mappings, model weights) is saved as one ZenML artifact. At serving time the same frozen artifact is loaded — no chance of training-serving skew.

**Test set touched once.** The test set is used exclusively in the `evaluate_model` step, never during development or hyperparameter search.

**Leakage columns are blocked at config level.** `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)`, and `Order Status` are listed in `features_config.yaml` as drop columns and removed before any fit call.

---

## Further Reading

- [problem_statement.md](problem_statement.md) — Business context, ML formulation, success criteria
- [architecture.md](architecture.md) — Full pipeline design, feature plan, deployment plan, monitoring plan
- [docs/system_design.md](docs/system_design.md) — System design document
- [docs/runbooks/rollback_runbook.md](docs/runbooks/rollback_runbook.md) — Step-by-step rollback procedure
- [docs/runbooks/incident_response.md](docs/runbooks/incident_response.md) — Incident response playbook
- [docs/ship_checklist.md](docs/ship_checklist.md) — Pre-ship checklist
