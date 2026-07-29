# M1 long-run controls

The stateful GRU/Loop comparison was repeated at 300 episodes per seed. On
seed 0 both controls reached `1.0` mean reward over the final 20 episodes.
The other seeds are retained as runtime logs when the campaign is rerun; this
task is saturated quickly, so a harder long-horizon environment is required
for a meaningful quality gap.

The untrained halting-gate smoke over 100 batches produced iteration counts
`1: 3` and `4: 97`. This confirms the hard budget and early-exit path; it is
not a calibration result. A trained halting objective is required before
claiming compute-adaptive behavior.
