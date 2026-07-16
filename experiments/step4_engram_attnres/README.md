# Step 4 — Engram × Attention Residuals

This experiment tests whether Full Attention Residuals improve the reuse of
representations injected by Engram.

## Controlled design

The complete 2×2 ablation is:

| Standard residual | Full AttnRes |
| --- | --- |
| `baseline` | `attnres` |
| `engram_noconv` | `engram_noconv_attnres` |

All four arms use the same tiny backbone, sample plan, optimization budget and
seed. AttnRes pseudo-queries are initialized to zero as specified in
arXiv:2603.15031. The construction tests also verify that every shared backbone
parameter is bit-identical between paired variants at initialization.

Engram outputs are treated as distinct depth-wise sources. This is the explicit
experimental extension being tested; the Attention Residuals paper itself does
not evaluate Engram.

## Campaign

The remote campaign reuses the archived `baseline` and `engram_noconv`
checkpoints and deterministic batch plans, then trains only the two new arms:

```bash
experiments/step4_engram_attnres/scripts/run_attnres_campaign.sh
```

Defaults:

- seeds `0..8`;
- three GPUs, one seed per GPU;
- WikiText-2 byte LM, no byte patching;
- 400 training steps and 64 LM evaluation steps;
- downstream comparison limit 512.

The decisive interaction on a loss metric is:

```text
(Engram+AttnRes - Engram) - (AttnRes - baseline)
```

A negative value means the combination improves loss beyond the sum of the two
individual architectural effects. Accuracy uses the opposite sign convention.
