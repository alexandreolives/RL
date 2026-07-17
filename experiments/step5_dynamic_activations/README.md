# Step 5 — Dynamic squared activations

This step tests whether a transition from Squared ReLU to Squared GELU can
combine fast early learning with stable late training.

Status: exploratory. The weighted interpolation is not recommended; the
single-branch stochastic schedule remains a candidate pending repeated
throughput and larger-task validation.

See [notes/RESULTS_2026-07-18.md](notes/RESULTS_2026-07-18.md).
