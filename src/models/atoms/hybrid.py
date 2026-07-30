"""Configurable hybrid schedules for Fourier/KDA/attention/Loop ablations."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .kda import KDA, GatedMLA
from .residual import MHCResidual


class FourierBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("FourierBlock expects [batch, sequence, d_model]")
        y = torch.fft.fft(x, dim=1, norm="ortho")
        y = torch.fft.fft(y, dim=2, norm="ortho").real
        return y


class LoopBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_dim: int):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(d_model, heads, ff_dim, 0.0, batch_first=True, norm_first=True)
        self.scale = nn.Parameter(torch.full((d_model,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


@dataclass(frozen=True)
class HybridStage:
    kind: str
    repeats: int = 1


class ConfigurableHybridCore(nn.Module):
    """Apply a configurable sequence of heterogeneous blocks.

    ``stages`` accepts strings (``"fourier"``, ``"kda"``, ``"attention"``,
    ``"loop"``, ``"mhc"``) or :class:`HybridStage` instances. Repeats are
    explicit and the returned trace records every executed stage.
    """
    VALID = {"fourier", "kda", "attention", "loop", "mhc"}

    def __init__(self, d_model: int, *, stages: list[str | HybridStage] | None = None,
                 heads: int = 4, ff_dim: int | None = None, kda_qat: bool = False):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        ff_dim = ff_dim or d_model * 2
        parsed = [HybridStage(s) if isinstance(s, str) else s for s in (stages or ["fourier", "kda", "attention"])]
        if any(s.kind not in self.VALID or s.repeats < 1 for s in parsed):
            raise ValueError(f"stages must use {sorted(self.VALID)} and positive repeats")
        self.stages = tuple(parsed)
        blocks: list[nn.Module] = []
        for stage in self.stages:
            for _ in range(stage.repeats):
                if stage.kind == "fourier": block = FourierBlock()
                elif stage.kind == "kda": block = KDA(d_model, heads=heads, qat=kda_qat)
                elif stage.kind == "attention": block = GatedMLA(d_model, heads)
                elif stage.kind == "loop": block = LoopBlock(d_model, heads, ff_dim)
                else: block = MHCResidual(2, d_model)
                blocks.append(nn.ModuleDict({"block": block, "norm": nn.LayerNorm(d_model)}))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, *, iterations: int = 1, return_trace: bool = False):
        if x.ndim != 3 or iterations < 1:
            raise ValueError("expected [batch, sequence, d_model] and iterations >= 1")
        y, trace = x, []
        for loop_index in range(iterations):
            for index, (stage, wrapped) in enumerate(zip(self._expanded_stages(), self.blocks)):
                block = wrapped["block"]
                inp = wrapped["norm"](y)
                if stage.kind == "kda": out, _ = block(inp)
                elif stage.kind == "mhc": out = block(y, inp)
                else: out = block(inp)
                y = y + out
                trace.append({"iteration": loop_index, "index": index, "kind": stage.kind})
        return (y, trace) if return_trace else y

    def _expanded_stages(self):
        return [HybridStage(stage.kind) for stage in self.stages for _ in range(stage.repeats)]
