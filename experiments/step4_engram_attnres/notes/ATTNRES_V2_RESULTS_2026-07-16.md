# Gated Engram bypass × Full AttnRes v2 results (2026-07-16)

## Verdict

The v2 bypass does not remove the language-model regression versus Engram.
It is significantly worse on paired loss and wins only one of nine seeds.
The downstream phase was therefore not run, following the campaign's
pre-registered stopping criterion.

## What v2 tested

`engram_noconv_attnres_v2` separates the two paths completely:

- AttnRes routes only the embedding, attention, and MLP deltas;
- Engram updates accumulate in a per-channel gated additive bypass;
- the bypass is added after every depth-wise aggregation and after the final
  aggregation, without consuming softmax probability mass.

The gates are initialized to one to preserve a strong Engram path and avoid
consuming random-number-generator state before paired initialization.

## Protocol

The LM protocol is identical to v0 and v1: nine paired seeds, the archived
sample plans, WikiText-2 raw bytes, sequence length 256, batch size 8, 400
training steps, and 64 evaluation steps. Three RTX 3060 Ti GPUs ran one seed
each. Datasets were forced offline and read from the populated cache.

## Language-model results

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| `engram_noconv` | **1.83785** | **6.28313** |
| `engram_noconv_attnres_v1` | 1.85408 | 6.38612 |
| `engram_noconv_attnres_v2` | 1.85988 | 6.42341 |
| `engram_noconv_attnres` (v0) | 1.86130 | 6.43236 |

Paired effects; negative is better:

| Comparison | Loss delta (95% CI) | PPL delta (95% CI) | Seed wins |
| --- | ---: | ---: | ---: |
| v2 − Engram | **+0.02203 ± 0.01014** | **+0.14028 ± 0.06468** | 1/9 |
| v2 − v1 | +0.00580 ± 0.01148 | +0.03729 ± 0.07339 | 3/9 |
| v2 − v0 | −0.00141 ± 0.01074 | −0.00895 ± 0.06890 | 6/9 |

V2 is effectively back at v0 performance. Its small mean improvement over v0
is inconclusive, while its regression against Engram is conclusive.

## Gate inspection

After training, the mean per-channel gates are approximately:

| Engram insertion | Mean gate across seeds |
| --- | ---: |
| first insertion | 0.993 |
| second insertion | 0.998 |

Individual channel values remain close to one. The optimizer therefore leaves
the cumulative bypass almost fully open. This indicates that removing Engram
from the depth softmax is not sufficient: the scale and placement of the
persistent path must also be controlled.

## Consequence

No downstream evaluation was launched. A future version should not reuse this
unit-initialized cumulative bypass unchanged. A better-controlled experiment
would use a bounded or zero/low-initialized residual gate, preferably with a
short gate-scale sweep before another nine-seed campaign.

## Artifacts

Remote root:

`/home/alexandre/RL-d3a0407/artifacts/attnres_engram_v2_multiseed/`

- checkpoints and LM metrics: `train/`;
- per-seed logs: `logs/`;
- campaign log: `artifacts/attnres_engram_v2_campaign.log`.
