# LeJEPA Results (2026-04-22)

Scope:
- comparison on 9 deterministic seeds (`0..8`)
- same train plans as step 1
- compared models:
  - `baseline`
  - `engram_noconv`
  - `engram_noconv + lejepa`

Eval tasks:
- `arc_challenge`, `arc_easy`, `hellaswag`, `mmlu`, `winogrande`, `openbookqa`

## Mean delta vs baseline

`engram_noconv`:
- `arc_challenge`: `+0.001486`
- `arc_easy`: `-0.002821`
- `hellaswag`: `+0.022352`
- `mmlu`: `-0.021267`
- `winogrande`: `0.000000`
- `openbookqa`: `+0.054222`

`engram_noconv + lejepa`:
- `arc_challenge`: `-0.000372`
- `arc_easy`: `-0.002604`
- `hellaswag`: `+0.020399`
- `mmlu`: `-0.021701`
- `winogrande`: `0.000000`
- `openbookqa`: `+0.055111`

## Difference (`lejepa - noconv`)

- `arc_challenge`: `-0.001858`
- `arc_easy`: `+0.000217`
- `hellaswag`: `-0.001953`
- `mmlu`: `-0.000434`
- `winogrande`: `0.000000`
- `openbookqa`: `+0.000889`

## Seed-level mean delta (6 tasks)

- `engram_noconv`: `+0.008995 ± 0.003899`
- `engram_noconv + lejepa`: `+0.008472 ± 0.003623`

Conclusion:
- in this text-byte proxy, LeJEPA is near-neutral overall.

Raw outputs:
- `~/RL/engram/artifacts/paper_tasks_compare_det_lejepa/seed*.json`
