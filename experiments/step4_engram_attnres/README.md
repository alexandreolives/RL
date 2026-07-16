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

Completed original-design results:

[notes/ATTNRES_RESULTS_2026-07-16.md](notes/ATTNRES_RESULTS_2026-07-16.md)

Completed fused-v1 results:

[notes/ATTNRES_V1_RESULTS_2026-07-16.md](notes/ATTNRES_V1_RESULTS_2026-07-16.md)

Completed gated-bypass v2 results:

[notes/ATTNRES_V2_RESULTS_2026-07-16.md](notes/ATTNRES_V2_RESULTS_2026-07-16.md)

## Corrective v1

`engram_noconv_attnres_v1` keeps attention and MLP outputs as the AttnRes depth
sources, but fuses each Engram output additively into the current attention
source. Engram therefore no longer competes as an independent softmax slot.
The original `engram_noconv_attnres` remains available as the reproducible
separate-source design.

Run the paired v1 campaign after the original campaign has released the GPUs:

```bash
experiments/step4_engram_attnres/scripts/run_attnres_v1_campaign.sh
```

## Corrective v2

`engram_noconv_attnres_v2` keeps Engram outside the depth-wise softmax. AttnRes
routes only embedding, attention, and MLP sources; Engram updates accumulate in
a separately gated additive bypass applied after every AttnRes aggregation and
after the final aggregation.

The campaign defaults to the nine-seed LM phase only:

```bash
experiments/step4_engram_attnres/scripts/run_attnres_v2_campaign.sh
```

Run `PHASE=eval` only if the paired LM comparison removes the regression versus
`engram_noconv`.

The completed LM campaign did not meet that criterion, so no downstream v2
evaluation was launched.

## Corrective v3

`engram_noconv_attnres_v3` uses the same external bypass with a bounded
sigmoid gate initialized to 0.1. It can be launched with the existing campaign
script by setting `VARIANT` and `OUT_ROOT`:

```bash
VARIANT=engram_noconv_attnres_v3 OUT_ROOT=artifacts/attnres_engram_v3_multiseed \
  experiments/step4_engram_attnres/scripts/run_attnres_v2_campaign.sh
```

The completed v3 LM result is recorded in
[notes/ATTNRES_V3_RESULTS_2026-07-16.md](notes/ATTNRES_V3_RESULTS_2026-07-16.md).
It improves v2 but remains below standalone Engram, so downstream evaluation
was not launched.

The native v6 mHC `M=4` + Engram + AttnRes screening is recorded in
[notes/V6_MHC_ENGRAM_ATTRES_SCREEN_2026-07-17.md](notes/V6_MHC_ENGRAM_ATTRES_SCREEN_2026-07-17.md).
