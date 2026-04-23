# Plan: <task>

## Objective
- <what changes and why>

## Inputs
- `docs/research/<task>.md`

## Edit Plan (Ordered)
1. `<path>` — <change>
2. `<path>` — <change>

## Interface Changes
- <signature/config/artifact contract changes>

## Test Plan
- Unit: <tests>
- Integration/Smoke: <tests>
- Gates:
  - `pytest --strict-markers -x`
  - `python -m ruff check .`
  - `python -m mypy .`

## Rollback Plan
- <safe revert strategy>

## Acceptance Criteria
- <criterion 1>
- <criterion 2>
