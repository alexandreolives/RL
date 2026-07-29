# M1 — Minimal multimodal agent

This step provides a deterministic cue-and-recall Gymnasium environment and a
small multimodal actor-critic baseline. It is a smoke environment for testing
representation and planning components, not a realistic benchmark.

`agents.ModularMultimodalAgent` composes the optional Engram latent adapter,
action-conditioned JEPA predictor, shared Loop Transformer and spectral loop
behind independent flags. Each block remains testable in isolation.

Run the CPU smoke training:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/step6_multimodal_agent/scripts/run_m1_smoke.py
```

Stateful GRU versus Loop comparison:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/step6_multimodal_agent/scripts/compare_stateful_memory.py
```
