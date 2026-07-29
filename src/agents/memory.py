from __future__ import annotations

import torch
from torch import nn

from models.atoms.engram import EngramMemory


class EngramLatentAdapter(nn.Module):
    """Expose the existing Engram brick as an optional latent-memory block."""

    def __init__(self, d_model: int = 64, *, slots: int = 257, heads: int = 4) -> None:
        super().__init__()
        self.memory = EngramMemory(
            d_model,
            slots=slots,
            heads=heads,
            top_k=2,
            ngram_orders=(2, 3),
            conv_enabled=False,
            long_conv_enabled=False,
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, latent: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or token_ids.shape != latent.shape[:2]:
            raise ValueError("latent must be (batch, sequence, dim) and token_ids must match its first two dims")
        recalled = self.memory(latent, token_ids)
        return latent + torch.sigmoid(self.gate) * recalled
