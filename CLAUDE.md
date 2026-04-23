# Supply Chain Agent Playbook

This file defines the default operating protocol for coding agents in this repository.

## Core Rule

Use **RPI** for any non-trivial change:
- **Research** (read-only)
- **Plan** (design-only)
- **Implement** (execution-only)

Do not merge phases in one long context.

## Task Classifier

Use full RPI when:
- touching multiple files/modules
- fixing non-obvious bugs
- changing training/serving behavior
- modifying metrics, thresholds, or registration logic

Skip full RPI only for small, explicit edits (single-file typo/config tweak).

## Research Prompt Template

```text
Goal: Understand <task>.
Mode: Read-only. No file edits.
Deliverable: docs/research/<task>.md

Include:
1) Relevant files and symbols
2) Current behavior and constraints
3) Risks (leakage, skew, reproducibility, gating)
4) Open questions
5) Exact file paths for proposed edits
```

## Plan Prompt Template

```text
Goal: Produce an implementation plan for docs/research/<task>.md.
Mode: Design-only. No source edits.
Deliverable: docs/plans/<task>.md

Include:
1) Ordered edit steps by file
2) Function signatures and interface changes
3) Test plan (unit + smoke)
4) Rollback plan
5) Explicit acceptance criteria
```

## Implement Prompt Template

```text
Goal: Execute docs/plans/<task>.md exactly.
Mode: Implement-only.

Rules:
1) Follow the plan step-by-step
2) If reality diverges, stop and request re-plan
3) Run gates before finalizing:
   - pytest --strict-markers -x
   - python -m ruff check .
   - python -m mypy .
```

## Context Management

- One agent = one task.
- Compact every 3-5 significant operations into:
  - `docs/research/<task>-checkpoint.md`
- Restart with a fresh agent after compaction.

Restart immediately if the agent:
- repeats steps
- contradicts prior decisions
- loses track of changed files
- proposes already-failed approaches

## Hard Blocks

- Never force push.
- Never bypass hooks (`--no-verify`).
- Never commit secrets (`.env`, tokens, credentials).
- Never change training logic without corresponding tests.
- Never change serving path without train/serve parity check.

## Quality Gates Policy

All changes must pass:
- `pytest --strict-markers -x`
- `python -m ruff check .`
- `python -m mypy .`

No partial passes, no "known failures", no skipped gate in final handoff.

## ML-Specific Safety Checks

For any preprocessing/training/inference change, explicitly verify:
- no leakage (`fit` only on train split)
- same transform logic in training and inference
- threshold provenance is explicit and versioned
- model artifact is deployable as one bundle
- evaluation is robust to edge cases (e.g., single-class slices)
