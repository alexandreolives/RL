# RL Compute-Efficient Research

This repository is a research lab notebook in code.

The core intention is to explore how far we can push useful intelligence under
strict compute constraints, with a practical emphasis on:

- memory-efficient architectures,
- stable representation learning,
- long-context and multimodal behavior,
- reproducible comparison protocols.

## Project intent

This is not a polished product model repository and not a single-paper
reproduction repo.

It is a step-by-step research program with explicit goals:

- build reusable architecture bricks,
- validate them with deterministic A/B protocols,
- keep only changes that survive controlled comparisons,
- progressively move from proxy tasks to harder multimodal settings.

In short: **scientific iteration under constrained compute**, not benchmark
theater.

## What is implemented today

- Step-based experiment organization in `experiments/`.
- Engram core brick and deterministic multi-seed proxy evaluations.
- LeJEPA-on-text extension and ablation results.
- A configurable Transformer stack split into reusable `atoms` and assembled
  `molecules`, with symbolic-token and byte-first input modes.
- Experimental DeepSeek-V4-inspired variants (`v1` through `v5`) combining
  sliding, compressed sparse, and heavily compressed attention; sparse/hash
  MoE; mHC residual streams; partial/scaled RoPE; and per-layer dynamic caches.
- Synthetic long-context probes and train/eval comparisons for passkey,
  multi-query retrieval, and variable tracking.
- OCR-like multimodal foundation (in progress), designed to be adapted to the
  existing Engram stack.

Main tracker: [experiments/ROADMAP.md](experiments/ROADMAP.md).

## Current status and limitations

- Several results are explicitly marked as proxy/smoke and should not be
  interpreted as paper-faithful reproduction.
- Some launchers are optimized for Docker-based workflows; platform notes are
  provided separately.
- Multimodal OCR-like work is active and not final.

See:
- [experiments/step1_engram_core/notes/PAPER_COMPARE.md](experiments/step1_engram_core/notes/PAPER_COMPARE.md)
- [experiments/step3_ocr_like/README.md](experiments/step3_ocr_like/README.md)

## Papers that motivated this project

The current research direction was shaped by four paper families:

- Engram-style memory-augmented transformers (long-context efficiency and
  low-overhead memory lookup).
- JEPA/LeJEPA world-model and representation-learning line (latent prediction
  and anti-collapse objectives).
- Modular token-stream world models for RL (Simulus / M3 direction).
- OCR/document understanding systems motivating the multimodal branch.

Repository references:
- Engram reading/comparison notes:
  [papers/notes/ENGRAM_PAPER_COMPARISON.md](papers/notes/ENGRAM_PAPER_COMPARISON.md)
- JEPA + Simulus paper pack:
  [papers/targets/jepa_simulus/README.md](papers/targets/jepa_simulus/README.md)
- Information-compression plan derived from those readings:
  [papers/notes/INFO_COMPRESSION_PLAN.md](papers/notes/INFO_COMPRESSION_PLAN.md)
- Additional bycloud-sourced references and recap:
  [papers/notes/BYCLOUD_ML_RECAP.md](papers/notes/BYCLOUD_ML_RECAP.md)

## Quick start

Requirements:

- Python `>=3.12`
- PyTorch (CUDA optional but recommended for non-trivial runs)

With [uv](https://docs.astral.sh/uv/) (recommended; `uv.lock` is committed):

```bash
uv sync
```

Or with a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Sanity check:

```bash
PYTHONPATH=src:. .venv/bin/python -c "import torch; from models.example import build_variant; model = build_variant('baseline'); print(torch.__version__, sum(p.numel() for p in model.parameters()))"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` and set `$env:PYTHONPATH = "src;."` before
running the Python commands.

## Build a model variant

```python
from models.example import build_variant

model = build_variant("v5", attention_backend="auto")
```

`build_variant` also exposes the controlled comparison variants such as
`baseline`, `engram`, `engram_noconv`, `engram_layerhash`, `dsa`, `mhc`, and
`full`. For lower-level control, construct a `TransformerConfig` from
`models.atoms.config` and pass it to `TransformerMolecule`.

## Run the lightweight evaluations

Start with a small CPU-safe comparison:

```bash
PYTHONPATH=src:. .venv/bin/python eval/transformer/long_context_accuracy.py \
  --device cpu --seq-len 256 --batch 2 --steps 2 \
  --variants baseline engram_noconv
```

Other entry points include:

```bash
PYTHONPATH=src:. .venv/bin/python eval/transformer/ablation.py
PYTHONPATH=src:. .venv/bin/python eval/transformer/train_long_context_compare.py --help
```

These are synthetic research probes. See
[eval/transformer/README.md](eval/transformer/README.md) and the experiment
notes before interpreting their metrics.

## Repository map

- `src/models/`: core model components.
- `src/models/atoms/`: attention, cache, embedding, Engram, MLP/MoE, norm,
  residual, and RoPE primitives.
- `src/models/molecules/`: configurable model assemblies.
- `eval/transformer/`: local evaluation and training utilities.
- `experiments/step1_engram_core/`: baseline/engram/engram_noconv experiments.
- `experiments/step2_engram_lejepa_eval/`: LeJEPA extension.
- `experiments/step3_ocr_like/`: OCR-like multimodal stack.
- `docker/`: optional container runtime notes and Dockerfiles.
- `docs/`: publishing and onboarding notes.
- `docs/ARCHITECTURE.md`: repository boundaries and contracts.

## How to navigate results

- Step 1 overview: [experiments/step1_engram_core/README.md](experiments/step1_engram_core/README.md)
- Step 1 trace: [experiments/step1_engram_core/notes/ENGRAM_STEP_TRACE.md](experiments/step1_engram_core/notes/ENGRAM_STEP_TRACE.md)
- Step 2 results: [experiments/step2_engram_lejepa_eval/notes/LEJEPA_RESULTS_2026-04-22.md](experiments/step2_engram_lejepa_eval/notes/LEJEPA_RESULTS_2026-04-22.md)
- Step 3 overview: [experiments/step3_ocr_like/README.md](experiments/step3_ocr_like/README.md)

## Runtime notes

Container-based launchers are available and optional:

- [docker/README.md](docker/README.md)

The long-term target is portable execution on standard Linux/macOS/Windows
Python setups, with Docker as an implementation convenience rather than a hard
requirement.
