# LeJEPA Remote GPU Validation (2026-07-15)

## Scope

This run validates that the paper-aligned LeJEPA implementation executes and
backpropagates correctly on a CUDA GPU. It is a technical smoke test, not a
quality benchmark and not evidence that LeJEPA improves language-model
accuracy.

Tested commit:
- `d3a0407` — `feat(transformer): add paper-aligned LeJEPA objective`

The archived multi-seed numbers in `LEJEPA_RESULTS_2026-04-22.md` remain
results for `lejepa_proxy`, not for this implementation.

## Remote environment

- host: `DESKTOP-AETC9GV`, reached through the `wsl2` SSH alias
- isolated checkout: `/home/alexandre/RL-d3a0407`
- container image: `faster-qwen3-tts-openai:latest`
- GPU selected: NVIDIA GeForce RTX 3060 Ti, 8 GiB
- PyTorch: `2.6.0+cu124`
- Transformers: `5.13.1`
- Datasets: `5.0.0`
- Pytest: `9.1.1`

The image's bundled Transformers 4.57.3 was not sufficient for the repository's
DeepSeek-V4 adapter. The pinned project dependencies were therefore installed
under `.remote-deps` in the isolated checkout. The pre-existing remote checkout
at `/home/alexandre/RL` was not modified because it contains historical
uncommitted and untracked files.

## Test suite

Executed remotely inside the CUDA-capable container:

```bash
python -m pytest -q tests/test_lejepa.py tests/test_deepseek_v4_v6.py
```

Result:

```text
9 passed in 52.22s
```

This covers deterministic pre-encoder views, SIGReg behavior, gradients
through every view, the integrated text-LM LeJEPA path, and the DeepSeek-V4 v6
regression suite.

## CUDA backward smoke test

Configuration:
- model: `baseline`, size `tiny`, byte input
- device: `cuda:0`
- batch size: `8`
- input length: `65` bytes (`64` LM input positions)
- LeJEPA global views: `2`
- mask ratio: `0.4`
- LeJEPA lambda: `0.1`
- SIGReg slices: `64`
- auxiliary loss weight in the hybrid LM objective: `0.05`
- deterministic view/SIGReg seed: `7`

Observed values:

| Metric | Value |
| --- | ---: |
| LM loss | `5.760946` |
| LeJEPA loss | `0.428028` |
| Total loss | `5.782347` |
| Byte-embedding gradient finite | `true` |
| Byte-embedding gradient norm | `0.026119` |
| Peak allocated CUDA memory | `119.9 MiB` |

## Verdict

The implementation passes its complete targeted test suite remotely and
produces finite non-zero encoder gradients during a real CUDA backward pass.
The subsequent deterministic nine-seed comparison is now complete and is
documented in `LEJEPA_REAL_RESULTS_2026-07-15.md`. No model-quality conclusion
should be drawn from this smoke test alone.
