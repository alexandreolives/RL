from __future__ import annotations

import torch
from torch import nn

from .latent_world_model import ActionConditionedLatentPredictor
from .loop_transformer import LoopTransformerCore
from .memory import EngramLatentAdapter
from .multimodal_baseline import MultimodalActorCritic
from .spectral_loop import SpectralAttentionLoop


class ModularMultimodalAgent(nn.Module):
    """M1 composition point with independently switchable architecture blocks."""

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        image_size: int = 8,
        use_engram: bool = False,
        use_jepa: bool = False,
        use_loop: bool = False,
        use_spectral: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MultimodalActorCritic(latent_dim=latent_dim, image_size=image_size)
        self.engram = EngramLatentAdapter(latent_dim) if use_engram else None
        self.jepa = ActionConditionedLatentPredictor(latent_dim) if use_jepa else None
        self.loop = LoopTransformerCore(latent_dim, max_iterations=2) if use_loop else None
        self.spectral = SpectralAttentionLoop(latent_dim) if use_spectral else None
        self.policy = nn.Linear(latent_dim, 2)
        self.value = nn.Linear(latent_dim, 1)

    def forward(
        self,
        *,
        image: torch.Tensor,
        bytes_view: torch.Tensor,
        symbolic: torch.Tensor,
        phase: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        latent = self.encoder.encode(image=image, bytes_view=bytes_view, symbolic=symbolic, phase=phase)
        sequence = latent.unsqueeze(1)
        if self.engram is not None:
            sequence = self.engram(sequence, bytes_view[:, :1])
        if self.spectral is not None:
            sequence, _, _ = self.spectral(sequence, force_attention=False)
        elif self.loop is not None:
            sequence, _ = self.loop(sequence)
        latent = sequence[:, -1]
        predicted = self.jepa(latent, action) if self.jepa is not None and action is not None else None
        return self.policy(latent), self.value(latent).squeeze(-1), latent, predicted
