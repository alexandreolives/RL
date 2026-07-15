# Deterministic Proxy Runbook

This runbook defines the current local "stable comparison" protocol for:
- `baseline`
- `engram`
- `engram_noconv`

It is a local proxy benchmark, not a paper-faithful reproduction.

## Goals

- keep variant comparison fair (same sampled training/eval examples)
- keep runs reproducible across seeds
- run efficiently on WSL2 Docker with 3 GPUs

## Training protocol

- dataset: `wikitext-2-raw-v1`
- byte-level input
- `seq_len=256`, `batch_size=8`
- `train_steps=400`, `eval_steps=64`
- recommended flags:
  - `--input-mode byte`
  - `--no-byte-patching`
  - `--byte-patch-size 1`
- per-seed deterministic batch plan:
  - save with `--batch-plan-out`
  - reload with `--batch-plan-in` when needed

## Evaluation protocol

Script:
- `eval/transformer/paper_tasks_compare.py`

Tasks:
- `arc_challenge`
- `arc_easy`
- `hellaswag`
- `mmlu`
- `piqa` (may be skipped depending on runtime dataset availability)
- `winogrande`
- `openbookqa`

Default eval args:
- `--limit 512`
- `--max-len 2048`

## Key outputs

- train outputs:
  - `artifacts/text_lm_compare_det/`
- deterministic plans:
  - `artifacts/text_lm_compare_det/plan_seed*.json`
- eval outputs:
  - `artifacts/paper_tasks_compare_det_engram/seed*.json`
  - `artifacts/paper_tasks_compare_det_noconv/seed*.json`

## Operational notes (WSL2 Docker)

- strict per-job GPU isolation:
  - use one container per seed
  - set `CUDA_VISIBLE_DEVICES=<gpu_id>` per container
- avoid mixing unrelated heavy jobs while collecting comparison results
- if GPU assignment is ambiguous, verify inside each container:
  - `python -c 'import torch; print(torch.cuda.device_count())'`

## Current decision

Until the paper-aligned multi-seed run is complete, `engram_noconv` remains the
preferred default based only on the historical proxy experiment.

## Note on JEPA/LeJEPA in this setup

The archived run used the lightweight objective now named `lejepa_proxy`; it
did not implement SIGReg or true pre-encoder views. Adding that proxy on top of
`engram_noconv` was approximately neutral. These numbers do not describe the
current paper-aligned `lejepa` mode.

Practical implication:
- keep `engram_noconv` as default for now.
- run a new deterministic multi-seed comparison with `--jepa-mode lejepa`
  before deciding whether the paper-aligned objective helps text.
