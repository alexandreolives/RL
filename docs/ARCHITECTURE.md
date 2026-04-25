# Repository Architecture

This file defines the intended boundaries of the repository.

## Layers

1. `src/`
- Reusable model code and core primitives.
- Must not depend on `experiments/`.

2. `eval/`
- Evaluation/training utilities built on top of `src/`.
- Can be imported by experiment runners.

3. `experiments/`
- Step-oriented orchestration, scripts, and result notes.
- Should call `eval/` and `src/`, but avoid re-implementing core logic.

4. `scripts/`
- Stable entrypoint wrappers for users.
- Should remain thin and delegate to `experiments/.../scripts/`.

## Step contracts

Step outputs should be explicit and machine-discoverable:
- write metrics to `artifacts/<step>/<run_id>/metrics.json`
- write checkpoints to `artifacts/<step>/<run_id>/model.pt`
- include config/runtime fields in metrics (`seed`, `device`, `steps`, etc.)

See:
- `experiments/contracts/README.md`

## Naming conventions

- New public launchers should use neutral names (`run_*.sh`) without
  environment-specific suffixes.
- Legacy `*_wsl2.sh` scripts are kept for backward compatibility and map to the
  same execution path.

## Documentation ownership

- Canonical result summary for Step 1: `experiments/step1_engram_core/notes/PAPER_COMPARE.md`
- Canonical run protocol for Step 2: `experiments/step2_engram_lejepa_eval/notes/DETERMINISTIC_PROXY_RUNBOOK.md`
- Canonical status tracker: `experiments/ROADMAP.md`
