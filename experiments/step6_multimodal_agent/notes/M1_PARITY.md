# M1 parity memory comparison

The parity task compares the stateful GRU and shared Loop Transformer on
sequence lengths 8, 16 and 32. The final action is rewarded only when it
matches the parity of the streamed binary cues.

Seed-0 smoke results:

| Length | GRU | Loop stateful | Episodes |
|---:|---:|---:|---:|
| 8 | 1.00 | 0.95 | 100 |
| 16 | 1.00 | 0.80 | 100 |
| 32 | 0.65 | 0.45 | 20 |

The longer task exposes a capacity/training issue rather than proving a model
ranking. It is substantially more informative than the single-cue task and is
the target for the next multi-seed run.

Three-seed follow-up at length 16 (100 episodes each):

| Seed | GRU | Loop stateful |
|---:|---:|---:|
| 0 | 1.00 | 0.80 |
| 1 | 1.00 | 0.45 |
| 2 | 1.00 | 0.30 |
| **mean** | **1.00** | **0.517** |

The GRU control is more reliable on this under-trained parity setup. This is
useful negative evidence for the current Loop implementation: the shared
attention core needs better optimization or a more suitable state update
before it can replace a recurrent control.

Revised state update (learned write gate around the carried latent) was then
smoke-validated without changing the external interface. At length 16 and
100 episodes, seed 0 reached GRU **1.00** and Loop **1.00**; two additional
100-episode runs also returned **1.00 / 1.00**. These are training-smoke
results, not a publication claim: they must be repeated with fixed budgets,
longer sequences and at least five seeds.

Additional low-thread smoke (the local BLAS runtime otherwise terminates long
processes) gave:

| Length | Episodes | Seed | GRU | Loop gated |
|---:|---:|---:|---:|---:|
| 16 | 20 | 0 | 0.75 | 0.90 |
| 32 | 20 | 0 | 0.65 | 0.80 |

These short runs suggest the gated update remains viable at longer context,
but are diagnostic only because the training budget is small.

Controlled five-seed run (one process, 50 episodes/seed, length 16) produced:

| Seed | GRU | Loop gated |
|---:|---:|---:|
| 0 | 0.95 | 0.95 |
| 1 | 0.90 | 0.65 |
| 2 | 1.00 | 0.15 |
| 3 | 0.35 | 0.55 |
| 4 | 0.95 | 1.00 |
| **mean** | **0.83** | **0.66** |

The variance is large and the GRU mean remains higher. The gated Loop is
therefore operational but not yet a reliable improvement; longer training and
matched compute/parameter controls are still required.

End-to-end CPU forward benchmark (multimodal encoder + memory + policy/value,
batch 1, 500 no-grad calls, one PyTorch thread) measured:

| Agent | Parameters | Latency |
|---|---:|---:|
| GRU baseline | 62,790 | 0.0815 ms |
| Loop gated | 79,686 | 0.3581 ms |

The current Loop is therefore about 4.4× slower end-to-end and has more
parameters. It should not be presented as a compute optimization yet; this is
the baseline to beat with a lighter shared block or GPU kernel.

The same single-process protocol with 100 episodes/seed converged for both
models:

| Seed | GRU | Loop gated |
|---:|---:|---:|
| 0 | 1.00 | 1.00 |
| 1 | 1.00 | 1.00 |
| 2 | 1.00 | 0.90 |
| 3 | 0.95 | 1.00 |
| 4 | 0.95 | 1.00 |
| **mean** | **0.98** | **0.98** |

This removes the earlier process-level artifact, but does not show a quality
advantage: both controls reach the task ceiling at this length and budget.

Long-context validation (length 32, 100 episodes/seed, five seeds) produced:

| Seed | GRU | Loop gated |
|---:|---:|---:|
| 0 | 0.95 | 1.00 |
| 1 | 0.85 | 0.30 |
| 2 | 0.95 | 1.00 |
| 3 | 0.95 | 1.00 |
| 4 | 1.00 | 0.95 |
| **mean** | **0.94** | **0.85** |

At length 32 the Loop remains competitive but has higher seed variance and a
lower mean than the GRU. This is the current M1 limitation, not evidence of a
general architectural advantage.

Depth-equivalence smoke (length 32, 50 episodes, seed 0) now exposes the
internal Loop stack explicitly:

| Loop depth × iterations | Effective depth | GRU | Loop |
|---:|---:|---:|---:|
| 1 × 2 | 2 | 0.90 | 1.00 |
| 2 × 1 | 2 | 0.90 | 0.95 |
| 2 × 2 | 4 | 0.90 | 0.95 |

This is only one seed, but it confirms that the previous one-block comparison
was incomplete. The implementation now supports a Nanbeige-style shared stack;
the next scientific run should repeat these rows over five seeds and report
parameter/FLOP-matched controls.

Completed five-seed depth ablation (length 32, 50 episodes/seed):

| Model/configuration | Parameters | Mean | Std |
|---|---:|---:|---:|
| GRU baseline | 62,790 | **0.92** | 0.027 |
| Loop 1×2 (effective depth 2) | 79,686 | 0.67 | 0.413 |
| Loop 2×1 (effective depth 2) | 113,158 | 0.82 | 0.347 |
| Loop 2×2 (effective depth 4) | 113,158 | 0.86 | 0.171 |

The GRU remains the strongest and most stable at this short budget. Increasing
the shared stack depth substantially reduces Loop variance and closes the gap,
but does not yet beat the recurrent control. These controls are now directly
comparable by explicit effective depth; parameter matching remains a separate
experiment because the deeper Loop has more weights.

CPU forward diagnostic (batch 1, 200 calls, one PyTorch thread) measured 62,790
parameters / 0.114 ms for the recurrent baseline and 41,856 parameters /
0.297 ms for the Loop state core. The Loop timing excludes the shared
multimodal encoder and is therefore not an end-to-end comparison; a matched
policy/encoder benchmark is still required for a cost claim.

Width-matched control: Loop 1×2 with `ff_dim=64` has 71,430 parameters,
closer to the GRU's 62,790, and scores `[1.00, 0.90, 1.00, 0.85, 0.25]`
(mean **0.80**, std **0.314**) on the same 5×50 length-32 protocol. Reducing
width does not remove the Loop's seed sensitivity; parameter count alone is
not the explanation for the observed variance.

The Loop now uses identity-biased LayerScale residuals (initial scale 0.1) on
each shared block, providing a gradient highway for deeper unrolling. A quick
3-seed smoke at depth 2 reached mean **0.917** (std **0.104**) for both one
and two iterations. This is a stability mechanism, not yet a statistically
validated gain; the earlier five-seed tables remain the baseline reference.
