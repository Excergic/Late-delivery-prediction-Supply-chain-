# Plan Artifacts

Use this folder for **Plan-phase** outputs only (design before coding).

## Naming

- `<task>.md`

## Required Structure

```markdown
# Plan: <task>

## Objective
- What will be changed and why

## Inputs
- Link to `docs/research/<task>.md`

## Edit Plan (Ordered)
1. `path/to/file.py` — exact change
2. `path/to/file.py` — exact change

## Interface Changes
- Function signatures, configs, or artifact contracts that change

## Test Plan
- Unit tests to add/update
- Integration/smoke checks
- Exact gate commands

## Rollback Plan
- How to revert safely if behavior regresses

## Acceptance Criteria
- Concrete pass/fail outcomes
```

## Rules

- No source edits during planning.
- Include exact file paths and expected behavior changes.
- Define verification before implementation starts.
