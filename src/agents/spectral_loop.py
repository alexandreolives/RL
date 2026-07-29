from __future__ import annotations

import torch
from torch import nn

from .loop_transformer import LoopTransformerCore


class FourierMixer(nn.Module):
    """Real-valued global token mixer using an orthonormal FFT."""

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError("latent must have shape (batch, sequence, d_model)")
        mixed = torch.fft.fft(latent, dim=1, norm="ortho")
        mixed = torch.fft.fft(mixed, dim=2, norm="ortho")
        return mixed.real


class SpectralAttentionLoop(nn.Module):
    """Adaptive compute control: cheap spectral pass, then shared attention."""

    def __init__(self, d_model: int = 64, *, heads: int = 4, max_attention_iterations: int = 2) -> None:
        super().__init__()
        self.spectral = FourierMixer()
        self.attention = LoopTransformerCore(
            d_model=d_model, heads=heads, max_iterations=max_attention_iterations
        )
        self.gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(
        self, latent: torch.Tensor, *, force_attention: bool = False
    ) -> tuple[torch.Tensor, bool, int]:
        spectral = self.spectral(latent)
        confidence = torch.sigmoid(self.gate(spectral.mean(dim=1))).mean()
        use_attention = force_attention or bool(confidence < 0.5)
        if not use_attention:
            return spectral, False, 0
        refined, states = self.attention(spectral)
        return refined, True, len(states)
