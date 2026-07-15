from __future__ import annotations

import torch
import torch.nn.functional as F
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

    def __init__(self, branches: int, d_model: int, *, sinkhorn_iters: int = 0, eps: float = 1e-6) -> None:
        super().__init__()
        self.branches = branches
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.pre = nn.Linear(d_model, branches, bias=True)
        self.post = nn.Linear(d_model, branches, bias=True)
        self.res = nn.Linear(d_model, branches * branches, bias=True)
        self.out = nn.Linear(branches * d_model, d_model, bias=False)
        nn.init.normal_(self.pre.weight, std=0.01)
        nn.init.normal_(self.post.weight, std=0.01)
        nn.init.normal_(self.res.weight, std=0.01)
        nn.init.zeros_(self.pre.bias)
        nn.init.zeros_(self.post.bias)
        nn.init.zeros_(self.res.bias)

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        matrix = torch.softmax(logits, dim=-1) + self.eps
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(self.eps)
        for _ in range(max(self.sinkhorn_iters - 1, 0)):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(self.eps)
        return matrix

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        pre = torch.sigmoid(self.pre(x)).unsqueeze(-1)
        post = 2.0 * torch.sigmoid(self.post(update)).unsqueeze(-1)
        mixed = pre * x.unsqueeze(-2) + post * update.unsqueeze(-2)

        res = self.res(x).view(*x.shape[:-1], self.branches, self.branches)
        res = self._sinkhorn(res) if self.sinkhorn_iters > 0 else torch.softmax(res, dim=-1)
        mixed = torch.einsum("...ij,...jd->...id", res, mixed)
        return self.out(mixed.reshape(*x.shape[:-1], -1))


class DeepseekV4HyperConnection(nn.Module):
    def __init__(self, hc_mult: int, d_model: int, *, sinkhorn_iters: int = 0, eps: float = 1e-6) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.mix_hc = (2 + hc_mult) * hc_mult
        self.hc_fn = nn.Linear(d_model * hc_mult, self.mix_hc, bias=True)
        self.hc_scale = nn.Parameter(torch.ones(3))
        self.hc_base = nn.Parameter(torch.zeros(self.mix_hc))
        nn.init.normal_(self.hc_fn.weight, std=0.01)
        nn.init.zeros_(self.hc_fn.bias)

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        matrix = torch.exp(logits - logits.amax(dim=(-1, -2), keepdim=True))
        for _ in range(self.sinkhorn_iters):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(self.eps)
        return matrix

    def _hc_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError("DeepseekV4HyperConnection expects [B, S, hc_mult, D] tensors")
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(x_flat, self.hc_fn.weight.float(), self.hc_fn.bias.float()) * rsqrt
        pre = mixes[..., : self.hc_mult] * self.hc_scale[0] + self.hc_base[: self.hc_mult]
        post = mixes[..., self.hc_mult : 2 * self.hc_mult] * self.hc_scale[1] + self.hc_base[
            self.hc_mult : 2 * self.hc_mult
        ]
        comb = mixes[..., 2 * self.hc_mult :] * self.hc_scale[2] + self.hc_base[2 * self.hc_mult :]
        comb = comb.view(*x.shape[:2], self.hc_mult, self.hc_mult)
        comb = self._sinkhorn(comb) if self.sinkhorn_iters > 0 else torch.softmax(comb, dim=-1)
        return pre, post, comb

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or update.dim() != 4:
            raise ValueError("DeepseekV4HyperConnection expects [B, S, hc_mult, D] tensors")
        pre, post, comb = self._hc_pre(x)
        pre = (torch.sigmoid(pre) + self.eps).unsqueeze(-1)
        post = 2.0 * torch.sigmoid(post).unsqueeze(-1)
        mixed = pre * x + post * update
        return torch.einsum("...ij,...jd->...id", comb, mixed)


class DeepseekV4HyperHead(nn.Module):
    def __init__(self, hc_mult: int, d_model: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.eps = eps
        hc_dim = hc_mult * d_model
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.zeros(hc_mult))
        self.hc_head_scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.hc_head_fn, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("DeepseekV4HyperHead expects [B, S, hc_mult, D] tensors")
        shape = x.size()
        dtype = x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(x_flat, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.eps
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(dtype)
