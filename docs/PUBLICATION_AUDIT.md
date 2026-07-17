# Publication audit

This document records the checks required before publishing the repository.

## Scope

The repository contains research implementations and proxy experiments, not
official DeepSeek, Engram, mHC, Attention Residuals, JEPA, or LeJEPA releases.
Results must be described as exploratory unless independently reproduced with
the protocols in `experiments/`.

## Local-only data

Generated checkpoints, logs, caches, and campaign outputs are ignored by Git.
Do not publish them accidentally. The repository does not require SSH hosts,
GPU MAC addresses, private paths, or local Docker images to run its tests.

## Reproducibility

Before release, run:

```bash
PYTHONPATH=src:. pytest -q
git status --short
git ls-files | rg '(^|/)(\.env|.*\.(pem|key))$'
```

Record the commit, Python/PyTorch versions, model-size, sequence length,
seeds, and input mode with every reported result. Clearly distinguish
`byte`, `symbolic`, and any tokenizer-backed experiment.

## Content and licensing

Source code is covered by the repository license. Papers, subtitles, and
video metadata are reference material; verify redistribution rights before
including them in a public release. Prefer links and short notes when rights
are unclear.

## Release bar

- all tests pass;
- no credentials or machine-specific paths are tracked;
- results include configuration and limitations;
- large artifacts are stored outside Git (or via a documented release asset);
- a fresh checkout can run the quick-start commands.
