# Bounded Engram bypass × Full AttnRes v3 results (2026-07-16)

## Verdict

V3 improves the failed v2 bypass, but still loses decisively to Engram alone.
It wins 0/9 paired seeds against `engram_noconv`, so the downstream phase was
not launched.

## Change from v2

V3 keeps Engram outside the AttnRes depth softmax, but replaces the unbounded
per-channel gain with a sigmoid-bounded gate initialized at `0.1`.

## Results

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| `engram_noconv` | **1.83785** | **6.28313** |
| `engram_noconv_attnres_v2` | 1.85988 | 6.42341 |
| `engram_noconv_attnres_v3` | 1.85108 | 6.36692 |

Paired effects; negative is better:

| Comparison | Loss delta (95% CI) | PPL delta (95% CI) | Seed wins |
| --- | ---: | ---: | ---: |
| v3 − Engram | **+0.01323 ± 0.00503** | **+0.08379 ± 0.03197** | 0/9 |
| v3 − v2 | −0.00880 ± 0.01093 | −0.05649 ± 0.06975 | 6/9 |

The low initial scale removes about 40% of v2's LM regression, but does not
recover the standalone Engram result.

## Gate behavior

The learned mean gates remain between `0.1010` and `0.1014` across all seeds
and both insertion layers. The short training budget therefore does not learn
to increase the bypass, suggesting that the useful scale is not simply a
single fixed coefficient applied to a cumulative Engram state.

## Consequence

No downstream evaluation was run. The external cumulative-bypass family
(v2/v3) is rejected as the preferred combination under this protocol. Further
work should test a local, normalized Engram injection or a non-cumulative gate
before spending another nine-seed campaign.

## Artifacts

`<repo>/artifacts/attnres_engram_v3_multiseed/`
