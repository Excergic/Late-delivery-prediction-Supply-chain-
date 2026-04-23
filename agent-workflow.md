# Agent Workflow: Supply Chain Late Delivery

## RPI Protocol
- Research phase: use a read-only agent to explore code paths and write findings to `docs/research/<task>.md`.
- Plan phase: use a fresh agent to convert research into concrete file-level edits and test plan in `docs/plans/<task>.md`.
- Implement phase: use a fresh agent to execute only the approved plan; stop and re-plan on unexpected behavior.

## Quality Gates
- Testing: `pytest --strict-markers -x` (all tests must pass, no known failures).
- Linting: `ruff` configured in `pyproject.toml` with strict rule set and first-party import grouping.
- Type checking: `mypy` configured in `pyproject.toml` with `disallow_untyped_defs`.
- Pre-commit: `.pre-commit-config.yaml` runs tests, lint, and typecheck before commit.
- One-command local gate runner: `bash scripts/quality_gate.sh`.

## Agent Isolation
- Branching strategy: work on feature branches only, never directly on `main`.
- Worktree usage: required for parallel agent work.
- Setup commands:
  - `git worktree add ../agent-<task> -b <branch-name>`
  - `git worktree remove ../agent-<task>`
- Hard blocks:
  - No `git push --force`
  - No `--no-verify`
  - No committing secrets (`.env`, keys, credentials)
- Traceability: include agent context in commit body (task scope + gate results).

## Anti-Slop Checklist
- [ ] All tests passing (100%, no skips)
- [ ] Strict linting enabled (zero warnings)
- [ ] Type checking passing
- [ ] Pre-commit hooks configured
- [ ] No force pushes allowed
- [ ] Agent cannot commit without gates
- [ ] Plans reviewed before implementation
- [ ] Research docs reference exact file paths
- [ ] Checklist reviewed in `docs/checklists/anti-slop-checklist.md`
