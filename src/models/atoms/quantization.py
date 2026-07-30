"""Small, differentiable block-float quantizers for K3 QAT experiments."""
from __future__ import annotations

import torch
from torch import nn


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


class BlockFloatFakeQuant(nn.Module):
    """Block-scaled fake quantization with a straight-through estimator.

    This models the training semantics of MXFP formats; it is deliberately not
    a hardware encoder. ``mantissa_bits=2`` is an MXFP4-like range and
    ``mantissa_bits=7`` an MXFP8-like range.
    """
    def __init__(self, *, block_size: int = 32, mantissa_bits: int = 2, exponent_bits: int = 2):
        super().__init__()
        if block_size < 1 or mantissa_bits < 1:
            raise ValueError("invalid block quantizer configuration")
        self.block_size, self.mantissa_bits, self.exponent_bits = block_size, mantissa_bits, exponent_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("fake quantization requires floating point input")
        flat = x.reshape(-1, self.block_size) if x.numel() % self.block_size == 0 else x.reshape(1, -1)
        scale = flat.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
        # symmetric mantissa codebook; exponent_bits controls the representable
        # dynamic range without pretending to pack hardware bytes.
        qmax = (2 ** (self.mantissa_bits + self.exponent_bits - 1)) - 1
        q = _ste_round(flat / scale * qmax).clamp(-qmax, qmax) * scale / qmax
        return q.reshape_as(x)


class MXFP4FakeQuant(BlockFloatFakeQuant):
    def __init__(self, block_size: int = 32):
        super().__init__(block_size=block_size, mantissa_bits=2, exponent_bits=2)


class MXFP8FakeQuant(BlockFloatFakeQuant):
    def __init__(self, block_size: int = 32):
        super().__init__(block_size=block_size, mantissa_bits=4, exponent_bits=3)
