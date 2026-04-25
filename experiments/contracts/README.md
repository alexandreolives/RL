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
- `model.pt` (optional, for training runs)
- extra reports (`*.json`, `*.txt`) as needed

## Minimal `metrics.json` fields

- `variant`
- `seed`
- `device`
- `train_steps` and/or `eval_steps`
- primary quality metrics (task-dependent)
- runtime metric (`train_time_sec` or latency)

## Notes

- Keep field names stable over time where possible.
- If schema changes are required, add `schema_version`.
