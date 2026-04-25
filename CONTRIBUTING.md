# Contributing

Thanks for contributing.

## Scope

This repository is research-first. Good contributions include:
- reproducible experiments,
- bug fixes with clear behavioral impact,
- documentation improvements that reduce ambiguity,
- benchmarks with deterministic settings and explicit seeds.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Contribution rules

- Keep changes scoped and minimal.
- Do not mix unrelated refactors with experiment changes.
- Preserve step-based organization in `experiments/`.
- If a result claim changes, update the related `notes/*.md`.
- Avoid committing large artifacts, logs, and checkpoints.

## Reproducibility expectations

For new experiments, document:
- exact command,
- seed(s),
- dataset source,
- hardware/runtime context,
- output path (`metrics.json`, score files).

## Pull request checklist

- [ ] Code runs end-to-end for the modified path.
- [ ] Docs updated (`README` + relevant step notes).
- [ ] No large binary artifacts committed.
- [ ] Claims in docs match generated results.
