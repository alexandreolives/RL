from __future__ import annotations

import torch
from torch import nn


class MultimodalActorCritic(nn.Module):
    """Small baseline consuming all environment modalities into one latent."""

    def __init__(self, *, latent_dim: int = 64, num_actions: int = 2, image_size: int = 8) -> None:
        super().__init__()
        image_dim = image_size * image_size
        self.image_encoder = nn.Sequential(nn.Flatten(), nn.Linear(image_dim, latent_dim), nn.Tanh())
        self.byte_encoder = nn.Embedding(256, latent_dim)
        self.symbolic_encoder = nn.Linear(4, latent_dim)
        self.phase_encoder = nn.Linear(1, latent_dim)
        self.fusion = nn.Sequential(nn.Linear(4 * latent_dim, latent_dim), nn.Tanh())
        self.policy = nn.Linear(latent_dim, num_actions)
        self.value = nn.Linear(latent_dim, 1)

    def encode(
        self,
        *,
        image: torch.Tensor,
        bytes_view: torch.Tensor,
        symbolic: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        image_z = self.image_encoder(image)
        byte_z = self.byte_encoder(bytes_view.long()).mean(dim=1)
        symbolic_z = self.symbolic_encoder(symbolic.float())
        phase_z = self.phase_encoder(phase.float())
        return self.fusion(torch.cat((image_z, byte_z, symbolic_z, phase_z), dim=-1))

    def forward(
        self,
        *,
        image: torch.Tensor,
        bytes_view: torch.Tensor,
        symbolic: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(image=image, bytes_view=bytes_view, symbolic=symbolic, phase=phase)
        return self.policy(latent), self.value(latent).squeeze(-1), latent


class RecurrentMultimodalActorCritic(nn.Module):
    """Recurrent control with the same multimodal encoder and heads."""

    def __init__(self, *, latent_dim: int = 64, num_actions: int = 2, image_size: int = 8) -> None:
        super().__init__()
        self.encoder = MultimodalActorCritic(latent_dim=latent_dim, num_actions=num_actions, image_size=image_size)
        self.recurrent = nn.GRUCell(latent_dim, latent_dim)
        self.policy = nn.Linear(latent_dim, num_actions)
        self.value = nn.Linear(latent_dim, 1)
        self.latent_dim = latent_dim

    def forward(
        self,
        *,
        image: torch.Tensor,
        bytes_view: torch.Tensor,
        symbolic: torch.Tensor,
        phase: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder.encode(image=image, bytes_view=bytes_view, symbolic=symbolic, phase=phase)
        if state is None:
            state = latent.new_zeros(latent.size(0), self.latent_dim)
        state = self.recurrent(latent, state)
        return self.policy(state), self.value(state).squeeze(-1), state
