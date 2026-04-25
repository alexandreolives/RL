from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    base: int = 10_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.size(-2)
    dim = q.size(-1)
    half_dim = dim // 2
    device = q.device
    dtype = q.dtype

    theta = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, theta)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None, :, :]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None, :, :]

    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k

