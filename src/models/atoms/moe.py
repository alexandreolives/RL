from __future__ import annotations

import torch.nn.functional as F
import torch
from torch import nn

from .mlp import FeedForward


class HashRouter(nn.Module):
    def __init__(
        self,
        *,
        hash_vocab_size: int,
        top_k: int,
        num_experts: int,
        layer_idx: int = 0,
        tid2eid: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hash_vocab_size = max(1, hash_vocab_size)
        self.top_k = top_k
        self.num_experts = num_experts
        self.layer_idx = layer_idx
        if tid2eid is None:
            base = torch.arange(self.hash_vocab_size, dtype=torch.long).unsqueeze(-1)
            offsets = torch.arange(self.top_k, dtype=torch.long).unsqueeze(0)
            tid2eid = (base * 1315423911 + (offsets + 1) * (104729 + self.layer_idx * 8191)) % self.num_experts
        self.register_buffer("tid2eid", tid2eid.to(torch.long), persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hash_ids = torch.remainder(token_ids.to(torch.long), self.hash_vocab_size)
        return self.tid2eid[hash_ids]


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        activation: str,
        num_experts: int,
        top_k: int,
        shared_expert: bool,
        scoring_func: str = "softmax",
        topk_method: str = "greedy",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        dropout: float = 0.0,
        router_jitter: float = 0.0,
        swiglu_limit: float | None = None,
        routing_mode: str = "moe",
        hash_vocab_size: int | None = None,
        layer_idx: int = 0,
        tid2eid: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.routing_mode = routing_mode
        self.hash_vocab_size = hash_vocab_size or max(num_experts * 32, 1)
        self.layer_idx = layer_idx
        self.router = nn.Linear(d_model, num_experts)
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts), persistent=False)
        self.hash_router = (
            HashRouter(
                hash_vocab_size=self.hash_vocab_size,
                top_k=self.top_k,
                num_experts=self.num_experts,
                layer_idx=layer_idx,
                tid2eid=tid2eid,
            )
            if routing_mode == "hash_moe"
            else None
        )
        self.experts = nn.ModuleList(
            [
                FeedForward(d_model, hidden_dim, activation=activation, dropout=dropout, swiglu_limit=swiglu_limit)
                for _ in range(num_experts)
            ]
        )
        self.shared = (
            FeedForward(d_model, hidden_dim, activation=activation, dropout=dropout, swiglu_limit=swiglu_limit)
            if shared_expert
            else None
        )
    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        router_logits = self.router(x)
        if self.training and self.router_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter

        if self.scoring_func == "sqrtsoftplus":
            router_scores = torch.sqrt(F.softplus(router_logits))
        elif self.scoring_func == "sigmoid":
            router_scores = torch.sigmoid(router_logits)
        else:
            router_scores = torch.softmax(router_logits, dim=-1)

        correction = self.e_score_correction_bias.view(*([1] * (router_logits.dim() - 1)), -1)

        if self.routing_mode == "hash_moe":
            if token_ids is None:
                raise ValueError("hash_moe routing requires token_ids")
            topk_idx = self.hash_router(token_ids)
            topk_weights = torch.gather(router_scores, dim=-1, index=topk_idx)
        else:
            if self.topk_method == "noaux_tc":
                _, topk_idx = torch.topk(router_scores + correction, k=self.top_k, dim=-1)
                topk_weights = torch.gather(router_scores, dim=-1, index=topk_idx)
            else:
                topk_weights, topk_idx = torch.topk(router_scores + correction, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            denom = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topk_weights = topk_weights / denom
        topk_weights = topk_weights * self.routed_scaling_factor

        expert_outputs = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=-2)

        gather_idx = topk_idx.unsqueeze(-1).expand(*topk_idx.shape, x.size(-1))
        selected = torch.gather(stacked, dim=-2, index=gather_idx)
        mixed = (selected * topk_weights.unsqueeze(-1)).sum(dim=-2)

        if self.shared is not None:
            mixed = mixed + self.shared(x)
        return mixed
