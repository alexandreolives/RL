# Engram Variant Benchmark (WSL2 Docker)

Date: 2026-04-21

Protocol:
- device: `cuda` (WSL2 docker, flash-attn enabled)
- model: `tiny`
- input: `symbolic`
- sequence length: `128`
- batch: `16`
- grad accumulation: `1`
- train steps: `300`
- eval steps: `8`
- train cache size: `64`
- seeds: `0, 1`

Compared variants:
- `engram`
- `engram_layerhash`
- `engram_compress`
- `engram_official_gate`
- `engram_noconv`
- `engram_fullconv`

Mean eval accuracy across seeds:

| Task | engram | layerhash | compress | official_gate | noconv | fullconv |
|---|---:|---:|---:|---:|---:|---:|
| passkey | 0.992188 | 0.992188 | 0.765625 | 0.984375 | 0.992188 | 0.992188 |
| multi_query | 0.562500 | 0.554688 | 0.167968 | 0.156250 | 0.929688 | 0.562500 |
| variable_tracking | 0.972657 | 0.992188 | 0.894532 | 0.949218 | 0.992188 | 0.972657 |

Takeaways:
- `engram_noconv` is the most robust overall in this setup, especially on `multi_query`.
- `engram_compress` is consistently weaker than other variants on all three tasks.
- `engram_official_gate` did not beat the default variant in this benchmark.
- `engram_layerhash` helps on `variable_tracking` and is neutral on `passkey`.

Re-run command:

```bash
bash scripts/run_engram_variant_grid.sh
```

## DDP Re-run (3x GPU, corrected distributed path)

Date: 2026-04-21

Protocol:
- launcher: `scripts/run_long_context_compare_ddp.sh`
- world size: `3`
- train task: `passkey`
- device: `cuda` (`torchrun`, DDP)
- model: `tiny`
- input: `symbolic`
- sequence length: `128`
- batch: `16`
- grad accumulation: `1`
- train steps: `300`
- eval steps: `8`
- train cache size: `64`
- seeds: `0, 1`

Mean eval accuracy across seeds:

| Variant | passkey | multi_query | variable_tracking |
|---|---:|---:|---:|
| baseline | 0.906250 | 0.003906 | 0.003906 |
| engram | 1.000000 | 0.234375 | 0.042968 |
| engram_noconv | 1.000000 | 0.132812 | 0.023438 |
| engram_fullconv | 1.000000 | 0.234375 | 0.042968 |
| engram_layerhash | 1.000000 | 0.105469 | 0.085938 |
| engram_official_gate | 1.000000 | 0.289062 | 0.222656 |
| engram_compress | 1.000000 | 0.289062 | 0.027344 |

Notes:
- this run supersedes the earlier "pseudo-multi-gpu" run where each process was
  independent before DDP wiring was fixed.
- with true DDP, Engram variants still beat baseline clearly on cross-task
  transfer while matching or exceeding baseline on `passkey`.
