# Paper-aligned LeJEPA Multi-seed Results (2026-07-15)

## Verdict

The paper-aligned LeJEPA objective is technically stable, but it does not
improve this text-byte setup under the tested hyperparameters.

Across nine paired seeds, `engram_noconv + lejepa` remains better than the
baseline by `+0.006651` accuracy points on average across the six available
tasks. However, it is `-0.002345` behind plain `engram_noconv` and `-0.001821`
behind the historical `lejepa_proxy`. Neither paired difference is
statistically conclusive with nine seeds.

Keep `engram_noconv` as the preferred default for this protocol.

## Protocol

- variants compared: baseline, `engram_noconv`, historical `lejepa_proxy`, and
  paper-aligned `engram_noconv + lejepa`
- seeds: `0` through `8`, paired through the archived deterministic batch plans
- dataset: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`
- byte-level input, no byte patching, sequence length `256`, batch size `8`
- training: `400` steps; LM evaluation: `64` steps
- paper-task evaluation: limit `512`, maximum length `2048`
- LeJEPA loss weight: `0.05`; mask ratio: `0.4`; lambda: `0.1`
- two global views, `256` SIGReg slices, `17` knots, `t_max=5.0`

The available tasks were ARC Challenge, ARC Easy, HellaSwag, MMLU,
Winogrande, and OpenBookQA. PIQA was unavailable in the runtime dataset stack
and was skipped consistently. The baseline outputs reproduced the archived
baseline exactly for every seed and available task.

## Task results

The table reports mean accuracy delta against the same per-seed baseline.
`Real - no JEPA` is the paired difference between paper-aligned LeJEPA and
plain `engram_noconv`.

| Task | `engram_noconv` | `lejepa_proxy` | real `lejepa` | Real - no JEPA |
| --- | ---: | ---: | ---: | ---: |
| ARC Challenge | `+0.001486` | `-0.000372` | `-0.005574` | `-0.007061` |
| ARC Easy | `-0.002821` | `-0.002604` | `-0.006944` | `-0.004123` |
| HellaSwag | `+0.022352` | `+0.020399` | `+0.018880` | `-0.003472` |
| MMLU | `-0.021267` | `-0.021701` | `-0.018012` | `+0.003255` |
| Winogrande | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| OpenBookQA | `+0.054222` | `+0.055111` | `+0.051556` | `-0.002667` |

Real LeJEPA is worse than plain `engram_noconv` on four tasks, better on MMLU,
and tied on Winogrande.

## Aggregate comparison

For each seed, the aggregate is the unweighted mean accuracy delta over the
six available tasks. The `+-` values below are population standard deviations
across seeds.

| Variant or paired contrast | Mean | Std. dev. |
| --- | ---: | ---: |
| `engram_noconv` vs baseline | `+0.008995` | `0.003899` |
| historical `lejepa_proxy` vs baseline | `+0.008472` | `0.003623` |
| real `lejepa` vs baseline | `+0.006651` | `0.004120` |
| real `lejepa` minus `engram_noconv` | `-0.002345` | `0.003565` |
| real `lejepa` minus `lejepa_proxy` | `-0.001821` | `0.003946` |

The paired real-LeJEPA comparison against `engram_noconv` has three seed wins
and six losses. An approximate two-sided 95% paired t interval is
`[-0.005251, +0.000562]`. Against `lejepa_proxy`, the corresponding interval is
`[-0.005039, +0.001396]`. Both include zero, so the observed regressions should
not be presented as statistically established effects.

## Language-model metrics

| Variant | Eval LM loss | Eval perplexity |
| --- | ---: | ---: |
| `engram_noconv` | `1.837851 +- 0.005996` | `6.283133 +- 0.037657` |
| historical `lejepa_proxy` | `1.837976 +- 0.006100` | `6.283926 +- 0.038328` |
| real `lejepa` | `1.856488 +- 0.007366` | `6.401387 +- 0.047132` |

Real LeJEPA increases perplexity by about `1.88%` relative to
`engram_noconv`. Its evaluation LeJEPA loss is stable at
`0.323331 +- 0.000214`. The proxy's auxiliary loss has a different definition
and scale and must not be compared numerically with the real LeJEPA loss.

## Interpretation

The implementation and optimization path work as intended, and the result is
reproducible. Under this configuration, however, the auxiliary representation
objective trades away some next-byte modeling quality without producing a
task-accuracy gain over the simpler control. A future sweep could test a lower
LeJEPA loss weight or different lambda/view/SIGReg settings, but the current
evidence does not justify enabling it by default.

## Execution record

- hardware: three NVIDIA GeForce RTX 3060 Ti GPUs, one seed job per GPU
- training wall time: approximately `9m35s`
- paper-task evaluation wall time: approximately `24m00s`
- total campaign wall time: approximately `33m35s`
- remote checkout: `<repo>`
- campaign log: `artifacts/lejepa_real_multiseed/campaign.log`
- training outputs: `artifacts/lejepa_real_multiseed/train/`
- evaluation outputs: `artifacts/lejepa_real_multiseed/eval/`

The historical proxy results remain documented separately in
`LEJEPA_RESULTS_2026-04-22.md`.
