"""Reference Kimi Linear / KDA-style recurrent attention blocks.

This is intentionally a clear PyTorch reference implementation, not a claim of
the fused production kernel used by Kimi.  It exposes a streaming state so it
can be benchmarked and replaced by a Triton kernel without changing callers.
"""
from __future__ import annotations

import torch
from torch import nn
from .quantization import MXFP8FakeQuant, QATLinear


class KDAState:
    def __init__(self, value: torch.Tensor):
        self.value = value


class KDA(nn.Module):
    """Gated delta-rule linear attention, with channel-wise decay."""
    def __init__(self, d_model: int, *, heads: int = 4, head_dim: int | None = None, qat: bool = False):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim or (d_model // heads)
        if self.heads * self.head_dim != d_model:
            raise ValueError("d_model must be divisible by heads")
        linear = QATLinear if qat else nn.Linear
        self.qkv = linear(d_model, 3 * d_model, bias=False)
        self.gate = nn.Linear(d_model, heads, bias=True)
        self.decay = nn.Parameter(torch.full((heads, self.head_dim), -2.0))
        self.out = linear(d_model, d_model, bias=False)
        self.activation_quant = MXFP8FakeQuant() if qat else nn.Identity()

    def init_state(self, x: torch.Tensor) -> KDAState:
        return KDAState(x.new_zeros(x.size(0), self.heads, self.head_dim, self.head_dim))

    def forward(self, x: torch.Tensor, state: KDAState | None = None) -> tuple[torch.Tensor, KDAState]:
        if x.dim() != 3:
            raise ValueError("KDA expects [batch, sequence, d_model]")
        b, s, _ = x.shape
        q, k, v = self.activation_quant(self.qkv(x)).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.head_dim)
        k = k.view(b, s, self.heads, self.head_dim)
        v = v.view(b, s, self.heads, self.head_dim)
        memory = self.init_state(x).value if state is None else state.value
        decay = torch.sigmoid(self.decay).view(1, self.heads, self.head_dim, 1)
        outs = []
        for t in range(s):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            memory = memory * decay
            pred = torch.einsum("bhij,bhj->bhi", memory, kt)
            delta = vt - pred
            write = torch.sigmoid(self.gate(x[:, t])).unsqueeze(-1)
            memory = memory + write.unsqueeze(-1) * delta.unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.einsum("bhij,bhj->bhi", memory, qt))
        y = torch.stack(outs, dim=1).reshape(b, s, -1)
        return self.out(y), KDAState(memory)


class HybridKDA(nn.Module):
    """Interleave ``kda_blocks`` recurrent blocks then one dense block.

    ``num_cycles=1`` gives the smallest 3:1 reference; larger values are useful
    for fair depth-matched ablations. Every block has its own parameters.
    """
    def __init__(self, d_model: int, *, kda_blocks: int = 3, num_cycles: int = 1, attention_heads: int = 4, qat: bool = False):
        super().__init__()
        if kda_blocks < 1 or num_cycles < 1:
            raise ValueError("kda_blocks and num_cycles must be positive")
        self.kda_blocks = kda_blocks
        self.kda = nn.ModuleList(KDA(d_model, heads=attention_heads, qat=qat) for _ in range(kda_blocks * num_cycles))
        self.attn = nn.ModuleList(GatedMLA(d_model, attention_heads) for _ in range(num_cycles))
        self.norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range((kda_blocks + 1) * num_cycles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        ki = ni = 0
        for cycle in range(len(self.attn)):
            for _ in range(self.kda_blocks):
                z, _ = self.kda[ki](self.norm[ni](y)); ki += 1; ni += 1
                y = y + z
            q = self.norm[ni](y); ni += 1
            a = self.attn[cycle](q)
            y = y + a
        return y


class GatedMLA(nn.Module):
    """Small gated dense-attention reference for K3-style hybrid ablations."""
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, _ = self.attn(x, x, x, need_weights=False)
        return value * torch.sigmoid(self.gate(x))
