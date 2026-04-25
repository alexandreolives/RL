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


def build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key in {"gelu", "gelu_pytorch_tanh"}:
        return nn.GELU(approximate="tanh")
    if key == "relu":
        return nn.ReLU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "geglu":
        return GEGLU()
    if key == "swiglu":
        return SwiGLU()
    raise ValueError(f"Unsupported activation: {name}")

