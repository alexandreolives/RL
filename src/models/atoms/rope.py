from __future__ import annotations

import torch


def _yarn_linear_ramp(low: float, high: float, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 1e-3
    ramp = (torch.arange(dim, device=device, dtype=dtype) - low) / (high - low)
    return ramp.clamp(0, 1)


def _yarn_correction_dim(num_rotations: float, dim: int, base: int, max_position_embeddings: int) -> float:
    return dim * torch.log(torch.tensor(max_position_embeddings / (num_rotations * 2 * torch.pi))).item() / (2 * torch.log(torch.tensor(base)).item())


def _apply_rope_scaling(
    theta: torch.Tensor,
    *,
    base: int,
    rope_scaling: dict[str, float | int | str] | None,
    rope_dim: int,
) -> torch.Tensor:
    if not rope_scaling:
        return theta
    scaling_type = str(rope_scaling.get("type", "")).lower()
    factor = float(rope_scaling.get("factor", 1.0))
    if factor <= 1.0:
        return theta
    if scaling_type == "linear":
        return theta / factor
    if scaling_type != "yarn":
        return theta

    original = int(rope_scaling.get("original_max_position_embeddings", 2048))
    beta_fast = float(rope_scaling.get("beta_fast", 32.0))
    beta_slow = float(rope_scaling.get("beta_slow", 1.0))
    half_dim = theta.numel()
    low = _yarn_correction_dim(beta_fast, rope_dim, base, original)
    high = _yarn_correction_dim(beta_slow, rope_dim, base, original)
    ramp = _yarn_linear_ramp(low, high, half_dim, device=theta.device, dtype=theta.dtype)
    inv_freq_interpolation = theta / factor
    return inv_freq_interpolation * (1 - ramp) + theta * ramp


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_single(
    x: torch.Tensor,
    *,
    base: int = 10_000,
    partial_rotary_factor: float = 1.0,
    on_tail: bool = False,
    positions: torch.Tensor | None = None,
    rope_scaling: dict[str, float | int | str] | None = None,
) -> torch.Tensor:
    seq_len = x.size(-2)
    dim = x.size(-1)
    rope_dim = int(dim * partial_rotary_factor)
    if rope_dim <= 0:
        return x
    rope_dim = min(dim, rope_dim)
    rope_dim -= rope_dim % 2
    if rope_dim <= 0:
        return x

    if on_tail:
        head = x[..., : dim - rope_dim]
        rope_x = x[..., dim - rope_dim :]
    else:
        rope_x = x[..., :rope_dim]
        tail = x[..., rope_dim:]

    half_dim = rope_dim // 2
    device = x.device
    dtype = x.dtype
    theta = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))
    theta = _apply_rope_scaling(theta, base=base, rope_scaling=rope_scaling, rope_dim=rope_dim)
    if positions is None:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
    else:
        positions = positions.to(device=device, dtype=dtype)
    freqs = torch.outer(positions, theta)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None, :, :]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None, :, :]

    rope_x = rope_x * cos + _rotate_half(rope_x) * sin
    if on_tail:
        if head.numel() == 0:
            return rope_x
        return torch.cat([head, rope_x], dim=-1)
    if tail.numel() == 0:
        return rope_x
    return torch.cat([rope_x, tail], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    base: int = 10_000,
    partial_rotary_factor: float = 1.0,
    on_tail: bool = False,
    positions: torch.Tensor | None = None,
    rope_scaling: dict[str, float | int | str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = apply_rope_single(q, base=base, partial_rotary_factor=partial_rotary_factor, on_tail=on_tail, positions=positions, rope_scaling=rope_scaling)
    k = apply_rope_single(k, base=base, partial_rotary_factor=partial_rotary_factor, on_tail=on_tail, positions=positions, rope_scaling=rope_scaling)
    return q, k
