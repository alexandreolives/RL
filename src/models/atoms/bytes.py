from __future__ import annotations

import torch
from torch import nn


class BytePatcher(nn.Module):
    def __init__(self, patch_size: int, pooling: str = "mean") -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        bsz, seq_len, dim = x.shape
        pad = (-seq_len) % self.patch_size
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        new_len = x.size(1) // self.patch_size
        x = x.view(bsz, new_len, self.patch_size, dim)
        if self.pooling == "mean":
            return x.mean(dim=2)
        if self.pooling == "max":
            return x.max(dim=2).values
        raise ValueError(f"Unsupported byte patch pooling: {self.pooling}")

