# Research Artifacts

Use this folder for **Research-phase** outputs only (read-only investigation).

## Naming

- `<task>.md`
- `<task>-checkpoint.md` for compaction snapshots

## Required Structure

```markdown
# Research: <task>

## Scope
- What question this research answers

## Code Locations
- `path/to/file.py` — symbol and role

## Current Behavior
- What happens today (facts only)

## Constraints
- Data, API, model, infra, or runtime constraints

## Risks
- Leakage/skew/reproducibility/security/regression risks

## Open Questions
- Unknowns to resolve before planning

## Candidate Edit Surface
- Exact files likely to change in implementation
```

## Rules

- No code changes during research.
- Use exact file paths.
- Distinguish facts from assumptions.
