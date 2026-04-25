# Transformer Eval

This folder contains lightweight local evaluations for the current byte-first
transformer stack.

These scripts are not a faithful reproduction of the Engram paper benchmark
suite. They are intended to validate that the model variants run end-to-end and
to provide early signals on latency and representation behavior before wiring
real tasks.

## Scripts

- `ablation.py`: compares a few architecture variants and reports parameter
  count plus rough forward-pass speed.
- `needle.py`: inserts a synthetic byte pattern in the middle of the sequence
  and probes the logits at the matching patch position.
- `retrieval.py`: repeats a synthetic byte pattern at multiple positions and
  reports simple vector statistics at the related patch locations.
- `common.py`: shared model/batch helpers.

## Run

```bash
PYTHONPATH=src:. .venv/bin/python eval/transformer/ablation.py
PYTHONPATH=src:. .venv/bin/python eval/transformer/needle.py
PYTHONPATH=src:. .venv/bin/python eval/transformer/retrieval.py
```

## Current limits

- The probes are synthetic and do not measure task accuracy.
- The numbers are noisy on CPU and should not be treated as final throughput.
- `mHC`, `DSA`, and `Engram` are still approximations of the papers, not
  faithful reproductions.
- The paper-level evaluations still need real datasets and benchmark harnesses.
- The current default `engram` path uses adaptive convolution:
  convolution is enabled on shorter sequences and disabled on longer ones.

## Related docs

- `../../experiments/ROADMAP.md`: global step tracking.
- `../../experiments/step1_engram_core/notes/`: step-1 consolidated docs/results.
- `../../experiments/step2_engram_lejepa_eval/notes/`: step-2 LeJEPA notes/results.
