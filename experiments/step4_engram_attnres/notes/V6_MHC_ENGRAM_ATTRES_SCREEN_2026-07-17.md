# Native v6 mHC + Engram + AttnRes screening (2026-07-17)

This screening tests the architecture described in the Bycloud Attention
Residuals video: four parallel mHC streams (`M=4`) combined with depth-wise
Attention Residuals. The native v6-compatible backbone also enables Engram.

## Protocol

- paired variants: `v6_engram` and `v6_engram_attnres`;
- seeds `0..8`, three RTX 3060 Ti GPUs;
- tiny model, byte input, batch size 2;
- 200 training steps and 32 evaluation steps;
- no byte patching, offline cached WikiText-2.

## Results

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| `v6_engram` | 2.29209 | 9.90285 |
| `v6_engram_attnres` | **2.20629** | **9.08287** |

Paired deltas (AttnRes − Engram; negative is better):

| Metric | Delta (95% CI) | Seed wins |
| --- | ---: | ---: |
| Loss | **−0.08580 ± 0.02931** | 9/9 |
| Perplexity | **−0.81997 ± 0.29377** | 9/9 |

This is a positive result for the Bycloud-style mHC + AttnRes composition in
this short native-v6 screening. It is not directly comparable to the earlier
tiny non-mHC Engram × AttnRes campaigns because the backbone and stream layout
are different. A longer run is warranted before downstream evaluation.

## Artifacts

`/home/alexandre/RL-d3a0407/artifacts/v6_mhc_engram_screen2/`
