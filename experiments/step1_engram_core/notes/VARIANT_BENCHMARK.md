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

## DDP Re-run with `full` / `full_noconv`

Date: 2026-05-10

Protocol:
- launcher: `scripts/run_long_context_compare_ddp.sh`
- world size: `3`
- train task: `passkey`
- device: `cuda` (`torchrun`, DDP)
- model: `tiny`
- input: `symbolic`
- attention backend: `sdpa`
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
| engram_noconv | 1.000000 | 0.132812 | 0.023438 |
| full | 0.007812 | 0.000000 | 0.007812 |
| full_noconv | 0.003906 | 0.003906 | 0.003906 |

Notes:
- `full` and `full_noconv` did not train meaningfully with this proxy setup and default hyperparameters.
- both variants stayed near chance on all three tasks, unlike `engram_noconv`.
- this suggests the `full` stack needs a separate tuning pass before it can be compared fairly against the lighter Engram variants.

## DDP Re-run with `v1` / `v2` / `v3` / `v4`

Date: 2026-05-10

Protocol:
- launcher: `scripts/run_long_context_compare_wsl2_ddp.sh`
- world size: `3`
- train task: `passkey`
- device: `cuda` (`torchrun`, DDP)
- model: `tiny`
- input: `symbolic`
- attention backend: `sdpa`
- sequence length: `128`
- batch: `8`
- grad accumulation: `1`
- train steps: `20`
- eval steps: `4`
- train cache size: `8`
- seeds: `0, 1`

Mean eval accuracy across seeds:

| Variant | passkey | multi_query | variable_tracking |
|---|---:|---:|---:|
| v1 | 0.031250 | 0.000000 | 0.031250 |
| v2 | 0.031250 | 0.000000 | 0.031250 |
| v3 | 0.000000 | 0.046875 | 0.000000 |
| v4 | 0.031250 | 0.000000 | 0.031250 |

Notes:
- `v1`, `v2`, and `v4` behave similarly on this short proxy run, with `passkey` and `variable_tracking` slightly above chance and `multi_query` at chance.
- `v3` is the only one that nudges `multi_query` upward in this setup, but it loses `passkey` and `variable_tracking`.
- `v4` is now the closest approximation to the public DeepSeek-V4 spec in this repo, but the proxy tasks still suggest it needs separate tuning before it can outperform the simpler Engram-only variants.

### Interpretation

These results are not yet a strong signal about model quality.

- The benchmark is still a small proxy: short sequences, tiny model, and a very small number of training steps compared with the earlier Engram-focused runs.
- `v4` is closer to the public DeepSeek-V4 architecture, but architecture similarity alone did not translate into better proxy-task scores here.
- `v1`/`v2`/`v4` all sit near chance on `multi_query`, which suggests the current default hyperparameters are not yet the right regime for the new stack.
- `v3` does recover a bit on `multi_query`, but not enough to make it a clear winner overall.
- The practical conclusion for now is that `v4` is a better architectural baseline, not a better-performing model yet.

Next step if we want a fairer comparison:
- run a dedicated tuning sweep for `v4`
- compare against `engram_noconv` with the same tuning budget
- test longer contexts and more training steps before drawing any conclusion

## WSL2 Fair Re-run with Historical Proxy Budget

Date: 2026-05-14

Protocol:
- launcher: direct `docker run` on WSL2 with `torchrun`
- world size: `3`
- train task: `passkey`
- device: `cuda` (`torchrun`, DDP)
- model: `tiny`
- input: `symbolic`
- attention backend: `sdpa`
- sequence length: `256`
- batch: `8`
- grad accumulation: `2`
- train steps: `300`
- eval steps: `64`
- train cache size: `0`
- seeds: `0, 1`

This run matches the historical Engram proxy budget and is the fairest apples-to-apples comparison for the current `v1` / `v2` / `v3` / `v4` family.
Unlike the earlier short or cached runs, this one removes the `train_cache_size` bias and gives each variant enough optimization steps to either fit the proxy or fail cleanly.

Mean metrics across seeds:

| Variant | train_loss_final | train_loss_mean_last10 | train_acc_mean_last10 | passkey | multi_query | variable_tracking |
|---|---:|---:|---:|---:|---:|---:|
| engram_noconv | 0.065003 | 0.065385 | 1.000000 | 1.000000 | 0.121093 | 0.031250 |
| v1 | 5.315755 | 5.325163 | 0.003125 | 0.005859 | 0.007812 | 0.003906 |
| v2 | 0.046018 | 0.046501 | 1.000000 | 1.000000 | 0.197266 | 0.008789 |
| v3 | 2.683677 | 2.680800 | 0.500000 | 0.501953 | 0.097656 | 0.006836 |
| v4 | 5.327799 | 5.328776 | 0.002083 | 0.004882 | 0.001953 | 0.006836 |

Seed-level notes:
- `engram_noconv` stays perfectly fit on `passkey` and remains the strongest cross-task baseline here.
- `v2` also fits `passkey` cleanly and improves `multi_query` more than `v1` or `v4`.
- `v3` splits the difference: it partially learns the task, but it does not outperform `v2` on this fair benchmark.
- `v1` and `v4` remain near chance on both optimization and validation in this exact-budget rerun.

Takeaways:
- This is the most scientifically fair comparison so far because the run budget matches the earlier Engram benchmark family.
- The older `train_cache_size=8` runs were biased toward memorization; removing that cache changes the picture materially.
- `engram_noconv` remains the clearest winner on this proxy.
- `v2` is the best of the newer V4-like variants on this exact budget, but it still does not clearly beat the Engram-only baseline.
- `v4` remains the weakest of the new variants on this exact proxy budget, so the architectural alignment alone is not sufficient.

## V5 Implementation Note

Date: 2026-05-14

`v5` is the next DeepSeek-V4-public-like candidate. It keeps `Engram` out of the architecture and moves closer to the public V4 mechanics instead of using `Engram` as a proxy.

Compared with `v4`, `v5` adds:
- K=V multi-query attention via `tie_kv=True`
- low-rank Q projection via `q_lora_rank`
- RMSNorm inside the low-rank Q path and direct KV path
- partial RoPE on the tail slice of the head dimension
- inverse RoPE on the attention output rope slice
- per-head learnable attention sink in the manual attention path
- CSA with separate `kv_proj` / `gate_proj`, learned `position_bias`, two-series `Ca/Cb` compression, and Lightning Indexer-style `ReLU(q*k) * weights` scoring
- HCA with separate `kv_proj` / `gate_proj`, learned `position_bias`, non-overlapping compression, retained local sliding branch, and compressed long-range prefix
- grouped low-rank attention output projection
- Sinkhorn-constrained mHC residual mixing
- multi-stream mHC with `hc_mult` parallel copies and a final HyperHead contraction
- clamped SwiGLU for routed and shared MoE experts
- a smaller scaled `index_topk` for `tiny`/`small` runs so the Lightning Indexer actually selects blocks at benchmark scale
- the public V4 layer schedule: HCA, HCA, then alternating CSA/HCA
- the same `hash_moe` bootstrap schedule for the first three MLP layers, followed by `moe`
- `noaux_tc`-style top-k selection: correction bias affects expert selection, not routed weights
- YaRN-style RoPE scaling with the public DeepSeek config values
- stateful CSA/HCA cache layers that retain K/V plus compressor/indexer pools and counts
- an optional final sliding-only MTP block through `num_nextn_predict_layers`

This variant should not be judged only by the `seq_len=256` proxy. Its purpose is to test the CSA/HCA/indexer path, so the useful benchmarks are longer-context runs with memory, speed, and accuracy reported together.

Remaining differences from the production DeepSeek/HF runtime:
- `tid2eid` can be injected, but the real checkpoint table is not present in this repo.
- FP8 KV / FP4 indexer paths are represented by storage hooks where PyTorch supports them; this is not a custom low-level kernel path.
- On-disk KV cache and serving-time cache paging are not implemented; this repo remains a research/training harness.

## WSL2 Short Validation Run

Date: 2026-05-14

Protocol:
- runtime: WSL2 Docker, single-process CUDA
- model: `tiny`
- input: `symbolic`
- attention backend: `eager`
- train task: `passkey`
- `seq_len=256`
- `batch=8`
- `grad_accum=2`
- `train_steps=60`
- `eval_steps=16`
- `train_cache_size=0`
- seed: `0`
- variants: `engram_noconv`, `v4`, `v5`

This was a short validation pass, not the full scientific two-seed benchmark. It is useful as a smoke-level sanity check after the DeepSeek-V4/HF port.

Results:

| Variant | train_loss_final | train_acc_mean_last10 | passkey | multi_query | variable_tracking |
|---|---:|---:|---:|---:|---:|
| engram_noconv | 5.552734 | 0.012500 | 0.000000 | 0.000000 | 0.007812 |
| v4 | 5.642578 | 0.000000 | 0.000000 | 0.000000 | 0.007812 |
| v5 | 5.468750 | 0.006250 | 0.000000 | 0.007812 | 0.000000 |

Takeaways:
- `v5` is now the most complete DeepSeek-V4-like candidate in the repo, but on this short proxy it still does not dominate `engram_noconv`.
- `v5` is marginally better on `multi_query` in this short run, while `engram_noconv` still edges it on `variable_tracking`.
- The short run is consistent with the earlier conclusion that architecture alignment alone does not guarantee proxy gains.

## WSL2 Single-Seed Proxy Run (Longer Context)

Date: 2026-05-10

Protocol:
- launcher: direct `docker run` on WSL2 with `torchrun`
- world size: `3`
- train task: `passkey`
- device: `cuda` (`torchrun`, DDP)
- model: `tiny`
- input: `symbolic`
- attention backend: `sdpa`
- sequence length: `256`
- batch: `2`
- grad accumulation: `2`
- train steps: `40`
- eval steps: `8`
- train cache size: `8`
- seeds: `0`

This run is more informative than the earlier short proxy pass, but it is still a single-seed exploratory benchmark.
The backbone sizing is now consistent across variants; `v2` and `v3` still carry a small Engram overhead, but they no longer jump back to the full-width configuration by mistake.

Results:

| Variant | train_loss_final | train_acc_mean_last10 | passkey | multi_query | variable_tracking |
|---|---:|---:|---:|---:|---:|
| v1 | 1.965495 | 0.750000 | 0.000000 | 0.000000 | 0.000000 |
| v2 | 1.037109 | 0.850000 | 0.000000 | 0.000000 | 0.000000 |
| v3 | 0.821777 | 0.991667 | 0.000000 | 0.000000 | 0.062500 |
| v4 | 2.881510 | 0.291667 | 0.000000 | 0.000000 | 0.000000 |

Takeaways:
- `v2` and `v3` now train properly on the proxy task once they are kept at the same `tiny` backbone size.
- `v3` is the only variant to move `variable_tracking` off exact chance in this run, but the gain is still small and not consistent enough to call it a win.
- `v4` still does not outperform the simpler variants on this proxy, so architecture similarity to the public DeepSeek-V4 spec is not enough by itself.
- The main lesson is that the V4-like stack is now being tested fairly, but it still needs either more tuning or a better proxy than `passkey`-style recall.

## V6 Reference Backend

Date: 2026-07-15

`v6` supersedes `v5` for new DeepSeek-V4 architecture experiments. The older
variants remain available so their recorded results stay reproducible, but
their aliases no longer claim exact public compatibility.

Unlike the native approximations, `v6` adapts
`transformers.DeepseekV4ForCausalLM` behind the repository's existing model
interface. It therefore uses the maintained reference implementations of:

- per-query sliding causal attention plus CSA/HCA long-range entries
- shared K=V multi-query attention and interleaved trailing partial RoPE
- inverse output RoPE and the non-renormalized attention sink
- stateful CSA/HCA buffers, overlap state, compressed pools, and counters
- collapse/sublayer/expand mHC dataflow with Sinkhorn projection
- hash-routed bootstrap layers and sparse top-k expert execution

The adapter adds deterministic balanced hash routes for fresh random models;
loading a checkpoint can replace that persistent `tid2eid` table. It also adds
a cache subclass whose `clone()` and `reset()` cover compressor and indexer
state, not only sliding K/V tensors.

Regression checks cover:

- full-context versus token-by-token logits (`atol=2e-5`, `rtol=2e-5`)
- live sliding attention during compressor warm-up
- K=V cache storage identity
- deep cache cloning and complete reset
- finite gradients through attention and routed experts

Run them with:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_deepseek_v4_v6 -v
```

Limits remain explicit: the default configs are small and randomly initialized,
official checkpoint weights are not bundled, production FP8/FP4 kernels are
outside this repository, and the current Hugging Face causal-LM runtime does
not execute the training-only MTP checkpoint module.
