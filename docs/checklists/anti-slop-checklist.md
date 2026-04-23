# Anti-Slop Checklist

Use this checklist before opening a PR or asking for final review.

## Spec Quality

- [ ] Task scope is explicit (what changes, what does not).
- [ ] Exact files/symbols are identified.
- [ ] Acceptance criteria are concrete and testable.
- [ ] Ambiguities are resolved before implementation.

## RPI Discipline

- [ ] Research output exists in `docs/research/`.
- [ ] Plan output exists in `docs/plans/`.
- [ ] Implementation followed approved plan, or plan was updated explicitly.
- [ ] Context compaction was used for long tasks.

## ML Safety

- [ ] No leakage introduced (`fit` only on train split).
- [ ] Training/serving transformation parity is preserved.
- [ ] Threshold provenance is explicit and versioned.
- [ ] Artifact contracts remain deployable and reproducible.

## Quality Gates

- [ ] `pytest --strict-markers -x` passes.
- [ ] `python -m ruff check .` passes.
- [ ] `python -m mypy .` passes.
- [ ] No gate bypass (`--no-verify`) used.

## Security and Hygiene

- [ ] No secrets or credentials included.
- [ ] No force push used.
- [ ] Rollback path is defined for behavior-changing work.
