from __future__ import annotations

import torch
from torch import nn


class LoopTransformerCore(nn.Module):
    """Shared Transformer block applied repeatedly to a latent sequence."""

    def __init__(self, d_model: int = 64, *, heads: int = 4, ff_dim: int = 128, max_iterations: int = 4) -> None:
        super().__init__()
        if d_model % heads or max_iterations < 1:
            raise ValueError("d_model must be divisible by heads and max_iterations >= 1")
        self.max_iterations = max_iterations
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, latent: torch.Tensor, *, iterations: int | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if latent.ndim != 3:
            raise ValueError("latent must have shape (batch, sequence, d_model)")
        steps = self.max_iterations if iterations is None else int(iterations)
        if not 1 <= steps <= self.max_iterations:
            raise ValueError("iterations must be between 1 and max_iterations")
        states = []
        state = latent
        for _ in range(steps):
            state = self.norm(state + self.block(state))
            states.append(state)
        return state, states
