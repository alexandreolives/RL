"""Minimal top-k router used as the M0 control implementation."""

from __future__ import annotations

import torch
from torch import nn

from .contracts import RouterDecision


class TopKRouter(nn.Module):
    """Token-wise MLP router with a deterministic single-expert mode."""

    def __init__(self, d_model: int, num_experts: int, *, top_k: int = 1, hidden_dim: int | None = None) -> None:
        super().__init__()
        if d_model < 1 or num_experts < 1:
            raise ValueError("d_model and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        hidden_dim = hidden_dim or d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_experts))

    def forward(self, latent: torch.Tensor) -> RouterDecision:
        if latent.ndim != 3:
            raise ValueError(f"latent must have shape (batch, sequence, d_model), got {tuple(latent.shape)}")
        logits = self.gate(latent)
        values, indices = torch.topk(logits, k=self.top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        decision = RouterDecision(expert_indices=indices, expert_weights=weights)
        decision.validate(batch_size=latent.size(0), sequence_length=latent.size(1), top_k=self.top_k)
        return decision


def routing_stats(decision: RouterDecision, *, num_experts: int) -> dict[str, float]:
    """Compute comparable routing diagnostics without changing dispatch."""

    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    flat_indices = decision.expert_indices.reshape(-1)
    counts = torch.bincount(flat_indices, minlength=num_experts).float()
    load = counts / counts.sum().clamp_min(1.0)
    entropy = -(load * load.clamp_min(1e-12).log()).sum()
    gini = (2 * torch.arange(1, num_experts + 1, device=load.device, dtype=load.dtype) - num_experts - 1)
    gini = (gini * torch.sort(load).values).sum() / (num_experts * load.sum().clamp_min(1.0))
    return {
        "routing_entropy": float(entropy.item()),
        "load_gini": float(gini.item()),
        "overflow_rate": float((counts == 0).float().mean().item()),
        "switch_rate": 0.0,
    }


class RandomRouter(nn.Module):
    """Uniform random dispatch control with the same typed output contract."""

    def __init__(self, num_experts: int, *, top_k: int = 1) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts, self.top_k = num_experts, top_k

    def forward(self, latent: torch.Tensor) -> RouterDecision:
        if latent.ndim != 3:
            raise ValueError("latent must have shape (batch, sequence, d_model)")
        shape = (*latent.shape[:2], self.num_experts)
        logits = torch.rand(shape, device=latent.device)
        _, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.full_like(indices, 1.0 / self.top_k, dtype=latent.dtype)
        return RouterDecision(indices, weights)


def build_single_expert_router(d_model: int) -> TopKRouter:
    """Return the deterministic M0 baseline router."""

    return TopKRouter(d_model, num_experts=1, top_k=1)
