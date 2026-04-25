# Engram Bench vs Paper

This note tracks what is reproduced vs not reproduced from the local Engram
paper references in `papers/`.

## Verdict

Current status:
- we have a solid local Engram research brick for internal ablations
- we do **not** have paper-faithful reproduction yet
- we cannot claim benchmark parity or improvement vs paper claims

## Paper claims to check

From local paper notes (`papers/notes/ENGRAM_PAPER_COMPARISON.md`), the main
measurable claims include:

- low inference overhead with large offloaded memory (`~2-3%`)
- benchmark gains on `MMLU`, `BBH`, `DROP`, `HumanEval`, `GSM8K`, `MATH`, etc.
- long-context gains on paper tasks (`NIAH`, variable tracking, long-context suites)

## What we have reproduced

Reproduced locally:
- stable WSL2 GPU docker workflow
- deterministic Engram variant harness
- internal synthetic long-context gains with controlled train-cache setup
- paper-harness runtime stack wired in Docker (`lm-eval`, `RULER`, `LongPPL` repos cloned)

Internal synthetic results (tiny, symbolic, cache=64, seeds 0/1):
- `passkey`: baseline `0.0586`, engram `0.9922`
- `multi_query`: baseline `0.0039`, engram `0.5547`
- `variable_tracking`: baseline `0.0469`, engram `0.9922`

Updated true-DDP run (3x GPU, train task `passkey`, cache=64, seeds 0/1):
- `passkey`: baseline `0.9063`, engram `1.0000`
- `multi_query`: baseline `0.0039`, engram `0.2344`
- `variable_tracking`: baseline `0.0039`, engram `0.0430`

Interpretation:
- the "Engram > baseline" direction remains true under corrected distributed
  training, but absolute values depend strongly on training protocol and train
  task selection.

These are meaningful internal gains, but not paper benchmark reproduction.

Paper-benchmark harness smoke run (`lm-eval`, 2026-04-21):
- runtime: `rl-engram:gpu-eval` (Docker, no venv)
- model: `EleutherAI/pythia-70m` (sanity backend model, not our Engram model)
- tasks: `arc_challenge`, `hellaswag`, `mmlu`
- limit: `32` (test-only, not final metrics)
- results:
  - `arc_challenge`: `acc=0.1250`, `acc_norm=0.0938`
  - `hellaswag`: `acc=0.3750`, `acc_norm=0.4062`
  - `mmlu`: `acc=0.2330`
- output file:
  - `~/RL/harnesses/results/lm_eval_20260421_115451.json`

Interpretation:
- harness execution is validated end-to-end in Docker.
- these numbers are **not** Engram-vs-baseline results, they only validate the
  benchmark runner path.

## What is not reproduced

Not reproduced yet:
- official paper benchmark suite wiring on our own model wrapper
  (`MMLU`, `BBH`, `DROP`, `HumanEval`, etc.)
- long-context eval suites from paper context (`LongPPL`, `RULER`)
- host-memory offload + deterministic prefetch system used in paper-scale setup
- full paper-scale training recipe and model scale

## Reproduced vs improved

Claim check:
- reproduced paper benchmark metrics: **no**
- improved vs paper benchmark metrics: **no claim possible**
- improved vs our previous local baseline on internal tasks: **yes**

Why comparison with paper is still invalid today:
- current `lm-eval` run used a reference HF model (`pythia-70m`), not
  `TransformerMolecule` + Engram variants.
- `--limit 32` is a smoke configuration and cannot be used for final benchmark
  claims.

Operational scripts now in repo:
- paper suite runner:
  - `scripts/run_paper_suite.sh`
- baseline-vs-engram scorecard:
  - `scripts/paper_scorecard.py`

First local-model baseline-vs-engram proxy benchmark (2026-04-21):
- model family: local `TransformerMolecule` checkpoints
- checkpoint recipe: `wikitext-2-raw-v1`, tiny setup, same train budget
- benchmark script: `eval/transformer/paper_tasks_compare.py`
- tasks: `arc_challenge`, `hellaswag`, `mmlu`
- limit: `32` per task
- results:
  - baseline:
    - `arc_challenge`: `0.1875`
    - `hellaswag`: `0.2188`
    - `mmlu`: `0.1250`
  - engram:
    - `arc_challenge`: `0.2188`
    - `hellaswag`: `0.1875`
    - `mmlu`: `0.0625`
  - delta (`engram - baseline`):
    - `arc_challenge`: `+0.0313`
    - `hellaswag`: `-0.0313`
    - `mmlu`: `-0.0625`

Interpretation:
- this is the first real baseline-vs-engram comparison on local checkpoints.
- result is mixed and currently does not support a "paper-like gain everywhere"
  claim.
- this remains a proxy benchmark, not a final paper-faithful reproduction.

Extended proxy run (2026-04-21, stronger checkpoint recipe):
- checkpoint training:
  - dataset: `wikitext-2-raw-v1`
  - seq len: `256`
  - batch: `8`
  - train steps: `400`
  - eval steps: `64`
  - seeds: `0, 1`
- paper-task compare:
  - tasks: `arc_challenge`, `hellaswag`, `mmlu`
  - limit: `128` per task

Per-seed deltas (`engram - baseline`):
- seed 0:
  - `arc_challenge`: `-0.0234`
  - `hellaswag`: `-0.0156`
  - `mmlu`: `-0.0391`
- seed 1:
  - `arc_challenge`: `0.0000`
  - `hellaswag`: `+0.0156`
  - `mmlu`: `-0.0156`

Mean across seeds:
- `arc_challenge`: `-0.0117`
- `hellaswag`: `0.0000`
- `mmlu`: `-0.0273`

Current implication:
- with this stronger proxy protocol, Engram is still not showing a consistent
  paper-like gain on these academic tasks.

Extended task coverage run (2026-04-21, limit=128, seeds `0/1/2`, parallel 1 GPU/seed):
- tasks: `arc_challenge`, `arc_easy`, `hellaswag`, `mmlu`, `winogrande`,
  `openbookqa` (`piqa` skipped in current datasets runtime)
- per-seed deltas (`engram - baseline`):
  - seed 0: `+0.0469`, `-0.0312`, `+0.0703`, `-0.0156`, `0.0000`, `+0.0625`
  - seed 1: `+0.0078`, `+0.0156`, `+0.0156`, `-0.0156`, `0.0000`, `+0.0625`
  - seed 2: `+0.0391`, `-0.0391`, `+0.0391`, `-0.0156`, `0.0000`, `+0.0234`
- mean delta across seeds:
  - `arc_challenge`: `+0.0312`
  - `arc_easy`: `-0.0182`
  - `hellaswag`: `+0.0417`
  - `mmlu`: `-0.0156`
  - `winogrande`: `0.0000`
  - `openbookqa`: `+0.0495`

Current implication:
- direction is now positive on several tasks, but still mixed.
- this still does not satisfy paper-faithful reproduction claims.

## Practical default

For local synthetic long-context tasks, current recommended variant is:
- `engram_noconv`

Rationale:
- best or tied-best across current internal tasks
- strong throughput behavior in local setup

See detailed variant table in:
- `experiments/step1_engram_core/notes/VARIANT_BENCHMARK.md`

## Deterministic 9-seed update (2026-04-22)

We added a stricter protocol to reduce train-sampling noise:
- shared deterministic batch plans per seed
- same training recipe per variant
- variants compared: `baseline`, `engram`, `engram_noconv`
- seeds: `0..8`
- proxy eval tasks:
  - `arc_challenge`, `arc_easy`, `hellaswag`, `mmlu`, `winogrande`,
    `openbookqa`

Implementation note:
- `eval/transformer/train_text_lm_compare.py` now supports
  - `--batch-plan-out`
  - `--batch-plan-in`
- this allows exact train/eval sample reuse between variants.

Result summary (delta accuracy vs baseline, mean over 9 seeds):

- `engram`
  - `arc_challenge`: `-0.0078`
  - `arc_easy`: `+0.0072`
  - `hellaswag`: `-0.0126`
  - `mmlu`: `-0.0215`
  - `winogrande`: `0.0000`
  - `openbookqa`: `+0.0420`

- `engram_noconv`
  - `arc_challenge`: `+0.0015`
  - `arc_easy`: `-0.0028`
  - `hellaswag`: `+0.0224`
  - `mmlu`: `-0.0213`
  - `winogrande`: `0.0000`
  - `openbookqa`: `+0.0542`

Current implication:
- `engram_noconv` remains the strongest practical default in this local proxy.
- both variants still underperform baseline on `mmlu` in this setup.

Artifacts:
- training outputs:
  - `~/RL/engram/artifacts/text_lm_compare_det/`
- deterministic plans:
  - `~/RL/engram/artifacts/text_lm_compare_det/plan_seed*.json`
- eval outputs:
  - `~/RL/engram/artifacts/paper_tasks_compare_det_engram/seed*.json`
  - `~/RL/engram/artifacts/paper_tasks_compare_det_noconv/seed*.json`

LeJEPA extension has been moved to Step 2 docs:
- `../../step2_engram_lejepa_eval/notes/LEJEPA_RESULTS_2026-04-22.md`
