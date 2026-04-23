# Incident Response Runbook

Use this when production behavior degrades.

## Triage Sequence

1. **Infrastructure health**
   - Confirm scoring job is running.
   - Confirm input data availability and output writes.
2. **Drift status**
   - Run drift pipeline and inspect drifted features.
3. **Golden-set parity**
   - Run:
     - `python scripts/golden_parity_check.py --input <golden_csv>`
4. **Timeline annotation**
   - When did issue begin?
   - What changed (model alias/config/data source)?
5. **Action decision**
   - Roll back immediately if guardrails fail:
     - `python scripts/rollback_production.py --to-version <last_good>`
   - Otherwise choose threshold adjustment/recalibration/retraining.

## Post-Incident

- Record user impact window and estimated business impact.
- Document root cause and detection gap.
- Add at least one preventive action item with owner and due date.
