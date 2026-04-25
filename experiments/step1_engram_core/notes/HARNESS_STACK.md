# Paper Eval Harness Stack

There is no single library that cleanly covers the full Engram paper stack.

Recommended stack:

- `lm-evaluation-harness`:
  - best first choice for standard academic benchmarks
  - use for `MMLU`, `BBH`, `DROP`, `GSM8K`, `MATH`, `ARC`, `HellaSwag`,
    `PIQA`, `WinoGrande`, `TriviaQA`, `RACE`, etc.
- `RULER`:
  - official long-context benchmark harness
  - use for paper-style long-context stress testing
- `LongPPL`:
  - official implementation for LongPPL metric
  - use for long-context language modeling signal beyond plain PPL

Official sources:
- `lm-evaluation-harness`: https://github.com/EleutherAI/lm-evaluation-harness
- `RULER`: https://github.com/NVIDIA/RULER
- `LongPPL`: https://github.com/PKU-ML/LongPPL

## Why this stack

`lm-evaluation-harness` is the easiest standard benchmark entry point:
- broad task coverage
- standard prompts
- local model support
- custom wrapper support
- OpenAI-compatible local server support

But it does not replace:
- `RULER` for the official long-context benchmark family
- `LongPPL` for the dedicated long-context metric

## Benchmark mapping

Use `lm-evaluation-harness` first:
- `MMLU`
- `MMLU-Pro` if task variant exists in your installed version
- `CMMLU`
- `ARC-Challenge`
- `BBH`
- `DROP`
- `HumanEval` / `MBPP` if supported cleanly in your setup
- `GSM8K`
- `MATH`

Use dedicated harnesses:
- `RULER` -> `RULER`
- `LongPPL` -> `LongPPL`

## Practical recommendation

Execution order:
1. wire `lm-evaluation-harness`
2. wire `RULER`
3. wire `LongPPL`

This is the fastest path to a defensible paper-gap report.

## One-command setup (local or remote)

Use:

```bash
docker build -f docker/Dockerfile.wsl2.gpu.eval -t rl-engram:gpu-eval .
./scripts/setup_eval_harnesses.sh
```

Default harness location:
- `$HOME/RL/harnesses`

## Current status

As of 2026-04-21:
- harness stack is runnable in Docker without any venv
- `lm-eval` smoke run completed successfully (`arc_challenge`, `hellaswag`, `mmlu`)
- next required step is a local model wrapper so `lm-eval` can call our
  `TransformerMolecule` variants directly
