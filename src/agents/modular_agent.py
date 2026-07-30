from __future__ import annotations

import torch
from torch import nn

from .latent_world_model import ActionConditionedLatentPredictor
from .loop_transformer import LoopTransformerCore
from .memory import EngramLatentAdapter
from .multimodal_baseline import MultimodalActorCritic
from .spectral_loop import SpectralAttentionLoop
from models.atoms.kda import HybridKDA
from models.atoms.latent_moe import LatentMoE


class ModularMultimodalAgent(nn.Module):
    """M1 composition point with independently switchable architecture blocks."""

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        image_size: int = 8,
        use_engram: bool = False,
        use_jepa: bool = False,
        use_loop: bool = True,
        use_spectral: bool = False,
        loop_depth: int = 2,
        loop_iterations: int = 2,
        use_kda: bool = False,
        use_latent_moe: bool = False,
        latent_moe_experts: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = MultimodalActorCritic(latent_dim=latent_dim, image_size=image_size)
        self.engram = EngramLatentAdapter(latent_dim) if use_engram else None
        self.jepa = ActionConditionedLatentPredictor(latent_dim) if use_jepa else None
        self.loop = (LoopTransformerCore(latent_dim, max_iterations=loop_iterations, depth=loop_depth)
                     if use_loop else None)
        self.spectral = SpectralAttentionLoop(latent_dim) if use_spectral else None
        self.kda = HybridKDA(latent_dim) if use_kda else None
        self.latent_moe = LatentMoE(latent_dim, max(8, latent_dim // 2), num_experts=latent_moe_experts) if use_latent_moe else None
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
        if self.latent_moe is not None:
            sequence = self.latent_moe(sequence)
        if self.kda is not None:
            sequence = self.kda(sequence)
        if self.loop is not None:
            sequence, _ = self.loop(sequence)
        latent = sequence[:, -1]
        predicted = self.jepa(latent, action) if self.jepa is not None and action is not None else None
        return self.policy(latent), self.value(latent).squeeze(-1), latent, predicted
