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
