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
- OCR-like multimodal foundation (in progress), designed to be adapted to the
  existing Engram stack.

Main tracker:
- [experiments/ROADMAP.md](experiments/ROADMAP.md)

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

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Repository map

- `src/models/`: core model components.
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
