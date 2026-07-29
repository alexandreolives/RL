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


def build_single_expert_router(d_model: int) -> TopKRouter:
    """Return the deterministic M0 baseline router."""

    return TopKRouter(d_model, num_experts=1, top_k=1)
