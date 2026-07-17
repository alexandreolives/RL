# Hybrid recurrent MoE research program

This document captures the proposed architecture as a set of testable
hypotheses. It is a research direction, not a claim that the complete system
already exists or matches a published model.

## Proposed system

### 1. Modular MoE-LoRA experts

Each domain expert is represented by an independent LoRA adapter. A learned
router selects one or more adapters per input or token. The intended benefit
is specialization without forcing unrelated domains through the same
parameter updates.

Tests: routing entropy, expert load balance, cross-domain interference,
parameter count, active FLOPs, and performance against a dense adapter and a
standard shared-MoE control.

### 2. Recurrent depth / implicit reasoning

A compact block is applied repeatedly rather than allocating a unique block
for every reasoning step. A halting or depth gate may predict the required
number of iterations. The claim to test is compute-adaptive refinement, not
that recurrence automatically produces chain-of-thought reasoning.

Tests: fixed versus adaptive iterations, accuracy versus iteration count,
stability of hidden-state norms, early-halt calibration, and a no-recurrence
control at matched active FLOPs.

### 3. Ternary / 1.58-bit deployment path

The intended deployment representation is ternary weights in `{-1, 0, +1}`,
with learned scales where needed. Training should first establish a useful
high-precision baseline, then introduce quantization-aware training (QAT).
The current repository does not yet provide a validated ternary kernel; this
is a future workstream.

Tests: full precision, post-training ternarization, straight-through QAT,
gradual QAT, weight histograms, activation ranges, perplexity, latency,
energy, and numerical drift.

### 4. Double-switch training schedule

The proposed schedule has three phases:

1. high-precision exploration with `SquaredReLU`;
2. a gradual interpolation to `SquaredGELU` while increasing the QAT weight;
3. strict `SquaredGELU` and ternary/QAT constraints.

The repository now contains the auditable activation primitives
`SquaredReLU`, `SquaredGELU`, and `ScheduledSquaredActivation`. The schedule
itself still needs to be integrated into a training loop and evaluated.

Tests: fixed ReLU², fixed GELU², linear and cosine schedules, reversed schedule
control, and schedule ablations with identical seeds and budgets.

### 5. Speculative verification

A high-precision depth gate may predict the number of recurrent iterations,
while additional iterations verify the answer before it is emitted. This is
an inference policy hypothesis, not a guarantee of correctness.

Tests: predicted versus required depth, rejection/correction rate, latency at
different confidence thresholds, and comparison with fixed-depth inference.

### 6. Expert lifecycle for continual RL

For an RL agent, experts can follow an explicit lifecycle:

1. **Birth:** expand the MoE only when novelty, saturation, or sustained
   routing imbalance is detected; initialize conservatively with a brief
   learning-rate warm-up.
2. **Exploration:** give new experts a temporary novelty bonus in the router
   while they collect evidence, using `SquaredReLU` as the fast exploratory
   activation.
3. **Maturation:** once quality and load stability thresholds are met, morph
   the activation toward a gated family such as GeGLU or SwiGLU. Log the
   morphing coefficient and keep a reversible control.
4. **Memory:** maintain an EMA shadow of mature experts and optionally reduce
   their learning rate. EMA is a stability mechanism, not a complete solution
   to catastrophic forgetting; replay and retention controls remain necessary.

Expansion must have a parameter budget, retirement policy, and rollback path;
“grow forever” is not a scientific or operational objective.

Tests: expansion versus fixed capacity, novelty-triggered versus periodic
expansion, router novelty bonus on/off, smooth morphing versus hard
replacement, EMA on/off, expert retirement, and continual-RL replay controls.
Report reward, regret after distribution shifts, expert load/Gini, active
parameters, expansion count, and recovery time after a task switch.

## Evaluation matrix

Every claim should be evaluated across short and long contexts, multiple
seeds, byte and symbolic inputs, matched parameter/FLOP budgets, and at least
one retrieval/dependency task in addition to language-model loss. Report mean,
standard deviation, wall time, peak memory, active parameters, and energy.

## Current status

- Engram, mHC, Attention Residuals, LeJEPA, and DeepSeek-V4-inspired controls
  exist as separate research components.
- The squared activation primitives are implemented and unit-tested.
- Dynamic-activation experiments and current findings are tracked in
  `experiments/step5_dynamic_activations/`.
- MoE-LoRA routing, dynamic expert lifecycle, recurrent depth, ternary QAT,
  and speculative verification remain experimental work items.
- No result in this repository should be presented as evidence of a superior
  general-purpose LLM without the controls above.
