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


class CausalGatedFourierBlock(nn.Module):
    """Causal spectral convolution with content gate and small LayerScale.

    The kernel is applied through an FFT after left-padding, then cropped to
    the causal prefix. No future token can affect an output position.
    """
    def __init__(self, d_model: int, kernel_size: int = 15):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(torch.zeros(d_model, kernel_size))
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.scale = nn.Parameter(torch.full((d_model,), 0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        n = s + self.kernel_size - 1
        signal = torch.nn.functional.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        kernel = torch.nn.functional.pad(self.kernel, (0, n - self.kernel_size))
        full = torch.fft.irfft(torch.fft.rfft(signal, n=n) * torch.fft.rfft(kernel, n=n).unsqueeze(0), n=n)
        conv = full[..., self.kernel_size - 1:self.kernel_size - 1 + s]
        conv = conv.transpose(1, 2)
        return x + self.scale * torch.sigmoid(self.gate(x)) * self.out(conv)


class AnchoredKDALoopMacroblock(nn.Module):
    """KDA→KDA→causal Fourier→shared KDA-loop×R→global MLA anchor."""
    def __init__(self, d_model: int, *, heads: int = 4, loop_repeats: int = 2, post_kda: bool = False, kda_qat: bool = False):
        super().__init__()
        if loop_repeats < 1:
            raise ValueError("loop_repeats must be positive")
        self.loop_repeats = loop_repeats
        self.kda1, self.kda2 = KDA(d_model, heads=heads, qat=kda_qat), KDA(d_model, heads=heads, qat=kda_qat)
        self.fourier = CausalGatedFourierBlock(d_model)
        self.loop_kda = KDA(d_model, heads=heads, qat=kda_qat)
        self.anchor = GatedMLA(d_model, heads)
        self.post_kda = KDA(d_model, heads=heads, qat=kda_qat) if post_kda else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        y, trace = x, []
        for name, block in (("kda", self.kda1), ("kda", self.kda2)):
            y, _ = block(y); trace.append(name)
        y = self.fourier(y); trace.append("causal_fourier")
        anchor = self.anchor(y); trace.append("mla_anchor")
        for _ in range(self.loop_repeats):
            update, _ = self.loop_kda(y)
            y = y + update + 0.1 * anchor
            trace.append("loop_kda")
        y = y + anchor
        if self.post_kda is not None:
            update, _ = self.post_kda(y); y = y + update; trace.append("post_kda")
        return y, trace


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
                 heads: int = 4, ff_dim: int | None = None, kda_qat: bool = False, carry_kda_state: bool = False):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        ff_dim = ff_dim or d_model * 2
        parsed = [HybridStage(s) if isinstance(s, str) else s for s in (stages or ["fourier", "kda", "attention"])]
        if any(s.kind not in self.VALID or s.repeats < 1 for s in parsed):
            raise ValueError(f"stages must use {sorted(self.VALID)} and positive repeats")
        self.stages = tuple(parsed)
        self.carry_kda_state = carry_kda_state
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
        kda_states = {}
        for loop_index in range(iterations):
            for index, (stage, wrapped) in enumerate(zip(self._expanded_stages(), self.blocks)):
                block = wrapped["block"]
                inp = wrapped["norm"](y)
                if stage.kind == "kda":
                    out, state = block(inp, kda_states.get(index) if self.carry_kda_state else None)
                    if self.carry_kda_state:
                        kda_states[index] = state
                elif stage.kind == "mhc": out = block(y, inp)
                else: out = block(inp)
                y = y + out
                trace.append({"iteration": loop_index, "index": index, "kind": stage.kind})
        return (y, trace) if return_trace else y

    def _expanded_stages(self):
        return [HybridStage(stage.kind) for stage in self.stages for _ in range(stage.repeats)]


class AdaptiveHybridCore(nn.Module):
    """Select a cheap or full schedule from a learned uncertainty estimate."""
    def __init__(self, d_model: int, *, fast_stages: list[str | HybridStage], full_stages: list[str | HybridStage], heads: int = 4):
        super().__init__()
        self.fast = ConfigurableHybridCore(d_model, stages=fast_stages, heads=heads)
        self.full = ConfigurableHybridCore(d_model, stages=full_stages, heads=heads)
        self.uncertainty = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, *, threshold: float = 0.5, iterations: int = 1, return_trace: bool = False):
        score = torch.sigmoid(self.uncertainty(x.mean(dim=1))).mean()
        use_full = bool(score >= threshold)
        core = self.full if use_full else self.fast
        result = core(x, iterations=iterations, return_trace=return_trace)
        if return_trace:
            y, trace = result
            return y, [{**item, "adaptive_full": use_full, "uncertainty": float(score.detach())} for item in trace]
        return result
