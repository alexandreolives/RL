# Experiment Contracts

This folder documents lightweight contracts between steps.

## Run ID

Recommended pattern:
- `<step>_<variant>_seed<seed>_<timestamp>`

## Output layout

Each run should write under:
- `artifacts/<step>/<run_id>/`

Expected files:
- `metrics.json` (required)
- `manifest.json` (recommended M0 run/config snapshot)
- `model.pt` (optional, for training runs)
- extra reports (`*.json`, `*.txt`) as needed

## Minimal `metrics.json` fields

- `variant`
- `seed`
- `device`
- `train_steps` and/or `eval_steps`
- primary quality metrics (task-dependent)
- runtime metric (`train_time_sec` or latency)

The reusable `RunRecord` and `write_run_manifest` helpers live in
`src/models/atoms/experiment.py`. Manifests use `schema_version: 1` and keep
the serialized modular configuration alongside run metadata.

## Notes

- Keep field names stable over time where possible.
- If schema changes are required, add `schema_version`.
