# Ship Checklist

- [ ] Training pipeline runs end-to-end without errors.
- [ ] Inference pipeline runs and writes predictions.
- [ ] Drift pipeline runs and generates drift report.
- [ ] Monitoring pipeline runs and generates monitoring report.
- [ ] Shadow comparison runs for staging vs production.
- [ ] Rollback command tested on a non-production alias.
- [ ] Golden parity check passes on approved golden set.
- [ ] Quality gates pass:
  - `bash scripts/quality_gate.sh`
- [ ] Deployment config reviewed for target environment.
- [ ] Runbooks updated (`rollback_runbook.md`, `incident_response.md`).
