# M1 — Minimal multimodal agent

This step provides a deterministic cue-and-recall Gymnasium environment and a
small multimodal actor-critic baseline. It is a smoke environment for testing
representation and planning components, not a realistic benchmark.

`MultimodalParityEnv` adds a harder streamed-sequence task in which the agent
must predict the parity of 8–32 multimodal cues at the final step.

`agents.ModularMultimodalAgent` composes the optional Engram latent adapter,
action-conditioned JEPA predictor, shared Loop Transformer and spectral loop
behind independent flags. Each block remains testable in isolation.

Current validation status (2026-07-29): the implementation and CPU test suite
pass (`53 passed`). On the streamed parity task at sequence length 16, the
three-seed mean is GRU **1.00** versus stateful Loop **0.517** (see
`notes/M1_PARITY.md`). This is a useful negative control: the current Loop
core is functional but is not yet a replacement for the trained GRU. Longer
training, a stronger recurrent state update and matched compute budgets remain
required before claiming an architectural gain.

Run the complete M1 checks from the repository root:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q
```

Run the CPU smoke training:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/step6_multimodal_agent/scripts/run_m1_smoke.py
```

Stateful GRU versus Loop comparison:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/step6_multimodal_agent/scripts/compare_stateful_memory.py
```

Parité séquentielle :

```bash
PYTHONPATH=src:. .venv/bin/python experiments/step6_multimodal_agent/scripts/compare_stateful_memory.py --task parity --sequence-length 16
```

Halting calibration smoke:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/step6_multimodal_agent/scripts/measure_halting.py
```
