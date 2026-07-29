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
