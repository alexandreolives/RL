from __future__ import annotations

import torch
from torch import nn


class LoopTransformerCore(nn.Module):
    """Shared Transformer block applied repeatedly to a latent sequence."""

    def __init__(self, d_model: int = 64, *, heads: int = 4, ff_dim: int = 128, max_iterations: int = 4, depth: int = 1) -> None:
        super().__init__()
        if d_model % heads or max_iterations < 1 or depth < 1:
            raise ValueError("d_model must be divisible by heads; depth and max_iterations >= 1")
        self.max_iterations = max_iterations
        self.depth = depth
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True, dropout=0.0, activation="gelu"
        ) for _ in range(depth)])
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
            for block in self.blocks:
                state = self.norm(state + block(state))
            states.append(state)
        return state, states


class StatefulLoopCore(nn.Module):
    """Streaming wrapper that carries the latest latent state across calls."""

    def __init__(self, d_model: int = 64, *, heads: int = 4, ff_dim: int = 128, max_iterations: int = 2, depth: int = 1) -> None:
        super().__init__()
        self.core = LoopTransformerCore(d_model, heads=heads, ff_dim=ff_dim, max_iterations=max_iterations, depth=depth)
        # A learned write gate makes the carried state genuinely recurrent:
        # the attention block proposes an update, while the gate controls how
        # much of it is committed for the next streamed observation.
        self.update_gate = nn.Linear(2 * d_model, d_model)

    def forward(self, latent: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2:
            raise ValueError("latent must have shape (batch, d_model)")
        if state is not None and state.shape != latent.shape:
            raise ValueError("state must have the same shape as latent")
        sequence = latent.unsqueeze(1) if state is None else torch.stack((state, latent), dim=1)
        output, _ = self.core(sequence)
        candidate = output[:, -1]
        if state is None:
            next_state = candidate
        else:
            gate = torch.sigmoid(self.update_gate(torch.cat((state, latent), dim=-1)))
            next_state = gate * candidate + (1.0 - gate) * state
        return next_state, next_state


class AdaptiveLoopCore(nn.Module):
    """Shared loop with a learned halt gate and a hard iteration budget."""

    def __init__(self, d_model: int = 64, *, heads: int = 4, max_iterations: int = 4, depth: int = 1, halt_threshold: float = 0.5) -> None:
        super().__init__()
        self.core = LoopTransformerCore(d_model, heads=heads, max_iterations=max_iterations, depth=depth)
        self.halt = nn.Linear(d_model, 1)
        self.max_iterations = max_iterations
        self.halt_threshold = halt_threshold

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 3:
            raise ValueError("latent must have shape (batch, sequence, d_model)")
        state = latent
        halt_probability = latent.new_zeros(latent.size(0))
        used = latent.new_zeros((), dtype=torch.long)
        for index in range(self.max_iterations):
            for block in self.core.blocks:
                state = self.core.norm(state + block(state))
            halt_probability = torch.sigmoid(self.halt(state.mean(dim=1)).squeeze(-1))
            used = used + 1
            if index + 1 < self.max_iterations and bool(torch.all(halt_probability >= self.halt_threshold)):
                break
        return state, used
