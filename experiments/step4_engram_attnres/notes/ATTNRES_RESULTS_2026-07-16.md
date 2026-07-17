# Engram × Full AttnRes Results (2026-07-16)

## Verdict

Full Attention Residuals improve the standard-residual baseline in this tiny
protocol, in the same direction as the Attention Residuals paper. The original
Engram combination, however, is antagonistic: making every Engram output an
independent depth-wise softmax source removes the standalone AttnRes benefit and
slightly degrades Engram.

The separate-source `engram_noconv_attnres` design is therefore rejected. The
next experiment is the fused corrective variant
`engram_noconv_attnres_v1`, where Engram remains an additive injection inside
the current attention source.

## Protocol

- variants: `baseline`, `attnres`, `engram_noconv`,
  `engram_noconv_attnres`;
- seeds: `0..8`, paired by deterministic archived sample plans;
- tiny byte-level model, no byte patching;
- WikiText-2 raw, sequence length 256, batch size 8;
- 400 training steps and 64 LM evaluation steps;
- downstream limit 512, max length 2048;
- downstream tasks: ARC-Challenge, ARC-Easy, HellaSwag, MMLU, Winogrande and
  OpenBookQA;
- hardware: three NVIDIA GeForce RTX 3060 Ti GPUs, one seed job per GPU;
- 95% intervals use the paired nine-seed mean ± Student-t half-width
  (`t(8)=2.306`).

The historical `baseline` and `engram_noconv` checkpoints were reused. The two
new arms use the same archived plans and seed-specific initialization. Unit
tests verify that shared backbone tensors are bit-identical at initialization;
the zero AttnRes pseudo-queries do not consume RNG state.

## Language-model results

Mean absolute metrics:

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| `baseline` | 2.17088 | 8.76661 |
| `attnres` | 2.12403 | 8.36561 |
| `engram_noconv` | **1.83785** | **6.28313** |
| `engram_noconv_attnres` | 1.86130 | 6.43236 |

Paired effects; negative is better:

| Comparison | Loss delta (95% CI) | PPL delta (95% CI) | Seed wins |
| --- | ---: | ---: | ---: |
| AttnRes − baseline | **−0.04684 ± 0.01612** | **−0.40099 ± 0.13771** | 9/9 |
| Engram − baseline | **−0.33303 ± 0.00975** | **−2.48347 ± 0.08342** | 9/9 |
| combined − Engram | **+0.02345 ± 0.00412** | **+0.14923 ± 0.02692** | 0/9 |

The loss interaction

```text
(combined - Engram) - (AttnRes - baseline)
```

is `+0.07029 ± 0.01875`, positive on 9/9 seeds. On loss, this is clear
antagonism rather than synergy.

## Downstream results

Accuracy deltas in percentage points:

| Task | AttnRes − baseline | combined − Engram |
| --- | ---: | ---: |
| ARC-Challenge | +0.595 ± 1.923 | −0.706 ± 1.048 |
| ARC-Easy | **+1.649 ± 1.040** | −0.260 ± 0.637 |
| HellaSwag | +0.629 ± 1.551 | −0.065 ± 0.463 |
| MMLU | −0.260 ± 0.833 | +0.260 ± 1.322 |
| Winogrande | 0.000 ± 0.000 | 0.000 ± 0.000 |
| OpenBookQA | +0.556 ± 1.140 | −0.511 ± 1.275 |
| Unweighted task aggregate | **+0.528 ± 0.297** | −0.214 ± 0.436 |

AttnRes beats the baseline aggregate on 9/9 seeds. The combined model beats
Engram on only 4/9 seeds; its aggregate confidence interval crosses zero.

The downstream interaction is `−0.742 ± 0.342` percentage point and is negative
on 9/9 seeds. This shows that the modest standalone AttnRes downstream gain is
not preserved once Engram is introduced as separate softmax sources.

## Interpretation

These results do not contradict the Attention Residuals paper:

- AttnRes alone improves both LM loss and the downstream aggregate locally;
- the paper does not test Engram;
- this model has only four Transformer blocks, so depth dilution is much weaker
  than at the scales targeted by the paper.

The failed combination gives each Engram output its own slot in the depth-wise
softmax. This both increases competition for probability mass and changes the
number of sources only in Engram layers. The most plausible correction is to
keep Engram as a strong additive/gated injection and let AttnRes route only the
attention/MLP deltas. That is exactly the `v1` fused design prepared for the
next campaign.

Follow-up: the fused v1 campaign is also complete. It improves v0 on LM loss
but remains worse than Engram and does not improve downstream results:
[ATTNRES_V1_RESULTS_2026-07-16.md](ATTNRES_V1_RESULTS_2026-07-16.md).

## Artifacts

Remote root:

`<repo>/artifacts/attnres_engram_multiseed/`

- training checkpoints and metrics: `train/`;
- downstream JSON files: `eval/`;
- per-seed logs: `logs/`;
- campaign log: `artifacts/attnres_engram_campaign.log`.
