from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LatentMoE(nn.Module):
    """MoE dispatch in a compressed latent space (K3/LatentMoE-inspired)."""
    def __init__(self, d_model: int, latent_dim: int, *, num_experts: int = 4, top_k: int = 1, hidden_dim: int | None = None, shared_experts: int = 0):
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.top_k, self.num_experts = top_k, num_experts
        self.shared_experts = shared_experts
        hidden_dim = hidden_dim or latent_dim * 2
        self.compress = nn.Linear(d_model, latent_dim)
        self.router = nn.Linear(latent_dim, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim)) for _ in range(num_experts)])
        self.shared = nn.ModuleList([nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim)) for _ in range(shared_experts)])
        self.expand = nn.Linear(latent_dim, d_model)
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.compress(x)
        logits = self.router(z)
        probs = logits.softmax(-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        outputs = torch.stack([expert(z) for expert in self.experts], dim=-2)
        gather = indices.unsqueeze(-1).expand(*indices.shape, z.size(-1))
        mixed = (torch.gather(outputs, -2, gather) * weights.unsqueeze(-1)).sum(-2)
        if self.shared:
            mixed = mixed + sum(expert(z) for expert in self.shared) / len(self.shared)
        load = probs.mean(dim=tuple(range(probs.dim() - 1)))
        self.last_aux_loss = self.num_experts * (load * load).sum()
        return x + self.expand(mixed)
