from __future__ import annotations

import torch
from torch import nn


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.silu(gate)


class SquaredReLU(nn.Module):
    """ReLU followed by a square, used by the recurrent/QAT prototype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


class SquaredGELU(nn.Module):
    """GELU followed by a square, with the same interface as other activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate="tanh").square()


class ScheduledSquaredActivation(nn.Module):
    """Differentiable interpolation from squared ReLU to squared GELU.

    ``alpha=0`` is squared ReLU and ``alpha=1`` is squared GELU.  Keeping the
    schedule explicit makes the transition auditable and easy to ablate.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(float(max(0.0, min(1.0, alpha))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = torch.relu(x).square()
        gelu = torch.nn.functional.gelu(x, approximate="tanh").square()
        # Multiplicative interpolation supports bf16 autocast on CUDA, whereas
        # torch.lerp requires a float32 tensor weight on some PyTorch builds.
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        return relu + (gelu - relu) * alpha


def build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key in {"gelu", "gelu_pytorch_tanh"}:
        return nn.GELU(approximate="tanh")
    if key == "relu":
        return nn.ReLU()
    if key in {"squared_relu", "relu2"}:
        return SquaredReLU()
    if key in {"squared_gelu", "gelu2"}:
        return SquaredGELU()
    if key in {"squared_schedule", "scheduled_squared"}:
        return ScheduledSquaredActivation()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "geglu":
        return GEGLU()
    if key == "swiglu":
        return SwiGLU()
    raise ValueError(f"Unsupported activation: {name}")
