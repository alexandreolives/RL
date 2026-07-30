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
from models.atoms.residual import FullAttentionResidual
from models.atoms.hybrid import ConfigurableHybridCore, AdaptiveHybridCore, AnchoredKDALoopMacroblock, HybridStage


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
        latent_moe_shared_experts: int = 0,
        use_attn_res: bool = False,
        hybrid_stages: list[str | HybridStage] | None = None,
        hybrid_iterations: int = 1,
        hybrid_carry_kda_state: bool = False,
        use_anchored_macroblock: bool = False,
        macroblock_loop_repeats: int = 2,
        macroblock_post_kda: bool = False,
        hybrid_fast_stages: list[str | HybridStage] | None = None,
        hybrid_full_stages: list[str | HybridStage] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = MultimodalActorCritic(latent_dim=latent_dim, image_size=image_size)
        self.engram = EngramLatentAdapter(latent_dim) if use_engram else None
        self.jepa = ActionConditionedLatentPredictor(latent_dim) if use_jepa else None
        self.loop = (LoopTransformerCore(latent_dim, max_iterations=loop_iterations, depth=loop_depth)
                     if use_loop else None)
        self.spectral = SpectralAttentionLoop(latent_dim) if use_spectral else None
        self.kda = HybridKDA(latent_dim) if use_kda else None
        self.latent_moe = LatentMoE(latent_dim, max(8, latent_dim // 2), num_experts=latent_moe_experts, shared_experts=latent_moe_shared_experts) if use_latent_moe else None
        self.attn_res = FullAttentionResidual(latent_dim) if use_attn_res else None
        if hybrid_fast_stages is not None or hybrid_full_stages is not None:
            if hybrid_fast_stages is None or hybrid_full_stages is None:
                raise ValueError("hybrid_fast_stages and hybrid_full_stages must be provided together")
            self.hybrid = AdaptiveHybridCore(latent_dim, fast_stages=hybrid_fast_stages, full_stages=hybrid_full_stages)
        else:
            self.hybrid = (ConfigurableHybridCore(latent_dim, stages=hybrid_stages, carry_kda_state=hybrid_carry_kda_state)
                           if hybrid_stages is not None else None)
        self.hybrid_iterations = hybrid_iterations
        self.macroblock = (AnchoredKDALoopMacroblock(latent_dim, loop_repeats=macroblock_loop_repeats, post_kda=macroblock_post_kda)
                           if use_anchored_macroblock else None)
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
        residual_sources = [sequence]
        if self.engram is not None:
            sequence = self.engram(sequence, bytes_view[:, :1])
            residual_sources.append(sequence)
        if self.spectral is not None:
            sequence, _, _ = self.spectral(sequence, force_attention=False)
            residual_sources.append(sequence)
        if self.latent_moe is not None:
            sequence = self.latent_moe(sequence)
            residual_sources.append(sequence)
        if self.kda is not None:
            sequence = self.kda(sequence)
            residual_sources.append(sequence)
        if self.hybrid is not None:
            sequence = self.hybrid(sequence, iterations=self.hybrid_iterations)
            residual_sources.append(sequence)
        if self.macroblock is not None:
            sequence, _ = self.macroblock(sequence)
            residual_sources.append(sequence)
        if self.loop is not None:
            sequence, _ = self.loop(sequence)
            residual_sources.append(sequence)
        if self.attn_res is not None:
            sequence = self.attn_res(residual_sources)
        latent = sequence[:, -1]
        predicted = self.jepa(latent, action) if self.jepa is not None and action is not None else None
        return self.policy(latent), self.value(latent).squeeze(-1), latent, predicted
