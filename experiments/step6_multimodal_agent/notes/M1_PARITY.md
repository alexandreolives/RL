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
