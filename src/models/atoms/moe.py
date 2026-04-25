from __future__ import annotations

import torch
from torch import nn

from .mlp import FeedForward


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        activation: str,
        num_experts: int,
        top_k: int,
        shared_expert: bool,
        dropout: float = 0.0,
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [FeedForward(d_model, hidden_dim, activation=activation, dropout=dropout) for _ in range(num_experts)]
        )
        self.shared = FeedForward(d_model, hidden_dim, activation=activation, dropout=dropout) if shared_expert else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(x)
        if self.training and self.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter

        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        stacked = torch.stack(expert_outputs, dim=-2)

        gather_idx = topk_idx.unsqueeze(-1).expand(*topk_idx.shape, x.size(-1))
        selected = torch.gather(stacked, dim=-2, index=gather_idx)
        mixed = (selected * topk_weights.unsqueeze(-1)).sum(dim=-2)

        if self.shared is not None:
            mixed = mixed + self.shared(x)
        return mixed

