"""Small, dependency-free contracts shared by modular experiments.

These contracts deliberately do not implement a model. They make optional
blocks observable and give experiment runners a stable place to validate
shapes before connecting real components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict

import torch


@dataclass(frozen=True)
class RouterDecision:
    """Typed output of a router for one batch of latent states."""

    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    depth_budget: torch.Tensor | None = None
    halt_probability: torch.Tensor | None = None
    novelty_score: torch.Tensor | None = None

    def validate(self, *, batch_size: int, sequence_length: int, top_k: int) -> None:
        if self.expert_indices.shape[:3] != (batch_size, sequence_length, top_k):
            raise ValueError(
                "expert_indices must have shape "
                f"({batch_size}, {sequence_length}, {top_k}, ...), got {tuple(self.expert_indices.shape)}"
            )
        if self.expert_weights.shape != self.expert_indices.shape:
            raise ValueError("expert_weights and expert_indices must have identical shapes")
        if not torch.is_floating_point(self.expert_weights):
            raise TypeError("expert_weights must be floating point")
        if self.depth_budget is not None and self.depth_budget.shape[:2] != (batch_size, sequence_length):
            raise ValueError("depth_budget must start with (batch_size, sequence_length)")


@dataclass(frozen=True)
class ModularAgentConfig:
    """M0 configuration for independently switchable architecture blocks."""

    use_engram: bool = False
    use_jepa: bool = False
    use_loop: bool = False
    use_router: bool = False
    use_moe_lora: bool = False
    use_ternary_qat: bool = False
    loop_iterations: int = 1
    router_top_k: int = 1
    extra: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.loop_iterations < 1:
            raise ValueError("loop_iterations must be >= 1")
        if self.router_top_k < 1:
            raise ValueError("router_top_k must be >= 1")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable experiment snapshot."""

        return asdict(self)

    @classmethod
    def baseline(cls) -> "ModularAgentConfig":
        """Return the deterministic all-blocks-off control."""

        return cls()
