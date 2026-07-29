# Technical note — Multimodal adaptive router

## Objective

In the target agent, the router is the control plane between observations,
latent streams, memory, experts, and recurrent computation. It must select
useful capacity without becoming a hidden source of compute, instability, or
expert collapse. A larger router is not automatically a better router: the
goal is calibrated decisions under a fixed active-FLOP budget.

## Responsibilities

The router may make five separate decisions:

1. **Modality:** text/bytes, image, document layout, or symbolic state.
2. **Function/domain:** perception, retrieval/memory, prediction, planning, or
   control.
3. **Experts:** top-k LoRA/MoE experts and their mixture weights.
4. **Depth:** number of Loop Transformer iterations or an early halt.
5. **Novelty:** whether to explore an under-used expert or request capacity
   expansion.

These decisions should be exposed separately even if a shared representation
is used internally. That makes each routing mechanism independently ablatable.

## Proposed architecture

```text
observation summary
        |
  modality gate
        |
 task/domain gate ---- novelty + uncertainty heads
        |
 expert top-k gate ---- depth/halting gate
        |
 latent streams -> memory/JEPA/Loop Transformer -> policy/value
```

The initial implementation should use small MLP gates over pooled latent
statistics and the recurrent state. A later version can add a recurrent gate
with temporal smoothing. Top-k routing is preferred for bounded active cost;
soft routing remains a control for measuring the quality/latency trade-off.

## Interface contract

Each gate receives a documented summary tensor and returns a typed decision:

- `modality_probs`: normalized modality probabilities;
- `domain_probs`: domain/function probabilities;
- `expert_indices`, `expert_weights`: top-k dispatch;
- `depth_budget` and `halt_probability`;
- `novelty_score` and `uncertainty`.

The forward pass must support `router=off`, which selects a deterministic
baseline, and must log decisions without changing numerical behavior. The
router configuration belongs in the experiment file, not in model code.

## Stabilization losses and constraints

The router should be trained with the task or policy loss plus separately
reported auxiliary terms:

- load-balancing loss and capacity overflow penalty;
- entropy or temperature regularization, scheduled rather than fixed;
- temporal-consistency penalty to prevent expert thrashing;
- novelty bonus for controlled exploration of new experts;
- halting calibration loss for adaptive depth;
- optional uncertainty calibration against rollout error.

Every auxiliary coefficient needs an ablation. A router that improves reward
only by activating more experts is not a valid efficiency gain.

## Required controls

At minimum compare:

1. fixed single expert;
2. random router;
3. linear/MLP top-k router;
4. hierarchical modality/domain/expert router;
5. recurrent or uncertainty-aware router;
6. dynamic-growth router with expansion disabled and enabled.

All comparisons use the same data, seeds, parameter budget, active FLOPs and
training steps. Report both routing-aware and end-task metrics.

## Metrics

- reward, regret and task accuracy;
- expert load, Gini coefficient and overflow rate;
- routing entropy, switch rate and temporal stability;
- active parameters, FLOPs, latency, peak memory and energy;
- calibration of uncertainty and halting;
- performance after modality/domain shifts;
- recovery time after a new task or expert expansion.

## Implementation order

1. Define the typed router interface and deterministic `off` mode.
2. Implement a top-k MLP router for a fixed expert pool.
3. Add load balancing and temporal-consistency logging.
4. Add modality/domain hierarchy and adaptive depth as independent gates.
5. Connect novelty to the dynamic expert lifecycle, with expansion behind a
   feature flag and a hard parameter budget.
6. Evaluate the complete router only after isolated gate ablations pass.

The router is therefore a first-class research axis, not glue code. Its claim
must be demonstrated by better quality at equal active compute, or equal
quality at lower active compute.
