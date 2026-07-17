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


class StochasticScheduledSquaredActivation(ScheduledSquaredActivation):
    """Choose one squared activation per forward according to ``alpha``.

    During training, GELU² is selected with probability ``alpha`` and ReLU²
    otherwise. Evaluation is deterministic. Unlike the interpolated schedule,
    this computes only one activation per forward.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__(alpha)
        self._alpha_value = float(alpha)
        self._use_gelu = self._alpha_value >= 0.5

    def set_alpha(self, alpha: float) -> None:
        super().set_alpha(alpha)
        self._alpha_value = float(max(0.0, min(1.0, alpha)))
        if self._alpha_value in {0.0, 1.0}:
            self._use_gelu = self._alpha_value == 1.0

    def sample_branch(self) -> None:
        """Sample the branch once, outside the latency-critical forward."""
        self._use_gelu = torch.rand(()).item() < self._alpha_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_gelu = self._use_gelu if self.training else self._alpha_value >= 0.5
        if use_gelu:
            return torch.nn.functional.gelu(x, approximate="tanh").square()
        return torch.relu(x).square()


def set_stochastic_squared_branch(root: nn.Module, *, use_gelu: bool) -> int:
    """Hot-swap stochastic scheduled FFNs to a native single activation.

    The feed-forward module keeps its ``activation_name`` tag, so this can be
    called once per optimizer step. No schedule wrapper remains on the forward
    path after the swap. Returns the number of replaced activations.
    """

    modules = getattr(root, "_stochastic_squared_ffns", None)
    if modules is None:
        modules = tuple(
            module
            for module in root.modules()
            if getattr(module, "activation_name", "") == "squared_stochastic_schedule"
        )
        # Bypass nn.Module.__setattr__: this is an execution cache, not a new
        # registered module hierarchy.
        object.__setattr__(root, "_stochastic_squared_ffns", modules)

    replaced = 0
    for module in modules:
        module.act = SquaredGELU() if use_gelu else SquaredReLU()
        replaced += 1
    return replaced


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
    if key in {"squared_stochastic_schedule", "stochastic_squared"}:
        return StochasticScheduledSquaredActivation()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "geglu":
        return GEGLU()
    if key == "swiglu":
        return SwiGLU()
    raise ValueError(f"Unsupported activation: {name}")
