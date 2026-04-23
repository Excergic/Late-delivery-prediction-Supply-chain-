# Rollback Runbook (Target: < 5 minutes)

Use this runbook when production model behavior degrades.

## Triggers

- F2 drops by >5% vs current baseline
- Recall drops below guardrail
- Prediction distribution collapses
- Golden input parity test fails

## Pre-checks (1 minute)

1. Confirm scoring pipeline is running.
2. Confirm issue is model-behavior, not infrastructure outage.
3. Identify last known-good model version in MLflow.

## Execute Rollback (1-2 minutes)

```bash
python scripts/rollback_production.py \
  --model-name supply-chain-late-delivery \
  --to-version <LAST_GOOD_VERSION>
```

## Verify (1-2 minutes)

1. Run shadow/spot check on a small batch.
2. Confirm prediction distribution returns to expected range.
3. Confirm no critical alert persists.

## Aftercare

- Open incident record with timeline.
- Capture root cause hypothesis.
- Decide next action: threshold adjustment, recalibration, or retraining.
