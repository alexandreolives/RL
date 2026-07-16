# Fused Engram × Full AttnRes v1 Results (2026-07-16)

## Verdict

The fused v1 correction materially improves the original separate-source
combination on language modeling, but it does not recover the performance of
Engram alone. Downstream, v1 is statistically indistinguishable from v0 and
remains slightly below Engram on the unweighted task aggregate.

`engram_noconv_attnres_v1` is therefore rejected as the preferred Engram
variant. It is useful evidence that reducing softmax competition helps, but a
future v2 should give Engram a true bypass outside depth-wise softmax rather
than merging it into a source that is still softmax-weighted later.

## What v1 changes

The original combination appends each Engram output as an independent AttnRes
source. v1 instead adds the Engram output to the current attention delta and
stores the fused delta as one source. On the four-block tiny model, the final
AttnRes history therefore contains nine sources instead of eleven:

- one token-embedding source;
- four attention sources, including fused Engram injections where configured;
- four MLP sources.

The old `engram_noconv_attnres` implementation remains available for exact
reproduction.

## Protocol

The protocol is identical to the original campaign:

- paired seeds `0..8` and the same archived sample plans;
- WikiText-2 raw byte LM, no byte patching;
- sequence length 256, batch size 8;
- 400 training steps and 64 LM evaluation steps;
- six downstream tasks, limit 512 and max length 2048;
- three RTX 3060 Ti GPUs;
- paired 95% Student-t intervals (`t(8)=2.306`).

Datasets were forced offline after the first launch encountered Hugging Face
`504` retries. All data came from the previously populated local cache. The
failed pre-training launch produced no checkpoints or metrics and was replaced
by a clean nine-seed restart.

## Language-model results

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| `engram_noconv` | **1.83785** | **6.28313** |
| `engram_noconv_attnres` (v0) | 1.86130 | 6.43236 |
| `engram_noconv_attnres_v1` | 1.85408 | 6.38612 |

Paired effects; negative is better:

| Comparison | Loss delta (95% CI) | PPL delta (95% CI) | Seed wins |
| --- | ---: | ---: | ---: |
| v1 − Engram | **+0.01623 ± 0.00373** | **+0.10299 ± 0.02410** | 0/9 |
| v1 − v0 | **−0.00721 ± 0.00259** | **−0.04624 ± 0.01662** | 9/9 |

The fused correction removes about 31% of the v0 loss regression:

```text
1 - (0.01623 / 0.02345) ≈ 0.31
```

However, the v1 loss interaction remains antagonistic at
`+0.06308 ± 0.01806`, positive on 9/9 seeds.

## Downstream results

Accuracy deltas for v1 relative to Engram, in percentage points:

| Task | v1 − Engram |
| --- | ---: |
| ARC-Challenge | −0.892 ± 0.987 |
| ARC-Easy | −0.477 ± 0.907 |
| HellaSwag | −0.174 ± 0.749 |
| MMLU | +0.369 ± 1.025 |
| Winogrande | 0.000 ± 0.000 |
| OpenBookQA | −0.333 ± 1.415 |
| Unweighted task aggregate | −0.251 ± 0.390 |

The aggregate is positive on 2/9 seeds and negative on 7/9. Its confidence
interval crosses zero, so the direct v1-versus-Engram downstream difference is
not conclusive.

Relative to v0, the v1 aggregate changes by `−0.038 ± 0.241` percentage point
(4/9 positive, 5/9 negative). The fused design does not produce a measurable
downstream improvement over the separate-source design.

The downstream interaction relative to standalone AttnRes is
`−0.779 ± 0.289` percentage point and negative on 9/9 seeds. The standalone
AttnRes downstream benefit is still lost in the presence of Engram.

## Conclusion and next design

Both tested combinations fail for the same broad reason: Engram information is
eventually multiplied by depth-wise softmax weights. v1 reduces competition by
removing independent Engram slots, but it does not provide a persistent unit or
gated bypass.

A justified v2 would keep two explicit paths:

1. AttnRes routes only embedding, attention and MLP deltas;
2. Engram contributes through a separately gated additive bypass after each
   AttnRes aggregation, without consuming softmax probability mass.

That v2 should be compared first against `engram_noconv`, not only against the
standard baseline. The stopping criterion is strict: it must remove the LM
regression before another full downstream campaign is worthwhile.

## Artifacts

Remote root:

`/home/alexandre/RL-d3a0407/artifacts/attnres_engram_v1_multiseed/`

- checkpoints and LM metrics: `train/`;
- downstream JSON files: `eval/`;
- per-seed logs: `logs/`;
- campaign log: `artifacts/attnres_engram_v1_campaign.log`.
