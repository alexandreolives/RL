# Engram Brick

This file defines how Engram is used as a reusable project brick.

## Why this brick exists

The Engram work in this repo is now stable enough to serve as a reusable
component for later world-model / RL experiments:

- deterministic memory-lookup path
- controlled variant switches
- reproducible WSL2 GPU docker workflow
- reproducible internal long-context task harness

This is a systems and method brick, not a claim of full paper reproduction.

## Brick scope

Implementation scope:
- model and variants in `src/models/` (`engram`, `engram_noconv`, etc.)
- training/eval harness in `eval/transformer/`
- WSL2 GPU docker stack in `docker/`
- reproducible variant runner in
  `scripts/run_engram_variant_grid.sh`

Out of scope for this brick:
- full paper benchmark harnesses (`MMLU`, `BBH`, `DROP`, `HumanEval`, etc.)
- host-memory offload + deterministic prefetch system from paper-scale setup
- paper-scale model sizes and full pretraining recipe

## Default recommendation

For internal long-context synthetic tasks, use:
- variant: `engram_noconv`
- backend: `flash`
- input mode: `symbolic`
- train cache size: `64` (or sweep)

Why:
- strongest and most stable results on internal synthetic tasks
- good throughput behavior in local setup

For proxy academic tasks (`arc*`, `hellaswag`, `mmlu`, `openbookqa`), current
best practical default is also `engram_noconv` (9-seed deterministic campaign).

## Repro commands

Single run:

```bash
bash scripts/run_long_context_compare_ddp.sh
```

Full variant grid:

```bash
bash scripts/run_engram_variant_grid.sh
```

## Current reliability status

- reliable for internal comparative research and ablations
- not yet valid for claiming paper-level benchmark reproduction

See:
- `experiments/step1_engram_core/notes/PAPER_COMPARE.md`
- `experiments/step1_engram_core/notes/VARIANT_BENCHMARK.md`
- `experiments/step1_engram_core/notes/PAPER_SUITE_RESULTS_2026-04-22.md`
