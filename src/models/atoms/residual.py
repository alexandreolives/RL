from __future__ import annotations

import torch
from torch import nn


class MultiBranchResidual(nn.Module):
    """Lightweight residual mixer inspired by multi-branch residual/hyper-connection ideas."""

    def __init__(self, branches: int, d_model: int) -> None:
        super().__init__()
        self.branches = branches
        self.mix = nn.Parameter(torch.full((branches,), 1.0 / branches))
        self.proj = nn.Linear(d_model * branches, d_model)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if len(xs) != self.branches:
            raise ValueError(f"Expected {self.branches} branches, got {len(xs)}")
        weights = torch.softmax(self.mix, dim=0)
        stacked = [x * w for x, w in zip(xs, weights)]
        return self.proj(torch.cat(stacked, dim=-1))


class MHCResidual(nn.Module):
    """
    Closer approximation to mHC:
    separate pre/post/residual mappings over branch streams.
    This is still simplified, but materially closer than a single weighted concat.
    """

    def __init__(self, branches: int, d_model: int) -> None:
        super().__init__()
        self.branches = branches
        self.pre = nn.Linear(d_model, branches, bias=True)
        self.post = nn.Linear(d_model, branches, bias=True)
        self.res = nn.Linear(d_model, branches * branches, bias=True)
        self.out = nn.Linear(branches * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        pre = torch.sigmoid(self.pre(x)).unsqueeze(-1)
        post = 2.0 * torch.sigmoid(self.post(update)).unsqueeze(-1)
        mixed = pre * x.unsqueeze(-2) + post * update.unsqueeze(-2)

        res = self.res(x).view(*x.shape[:-1], self.branches, self.branches)
        res = torch.softmax(res, dim=-1)
        mixed = torch.einsum("...ij,...jd->...id", res, mixed)
        return self.out(mixed.reshape(*x.shape[:-1], -1))
