from __future__ import annotations

import math

import torch
from torch import nn

from .norms import RMSNorm


def build_engram_lookup_state(
    token_ids: torch.Tensor,
    *,
    ngram_orders: tuple[int, ...],
    heads: int,
    slots: int,
    layer_ids: tuple[int, ...] | None = None,
    use_layerwise_hash: bool = True,
    compressed_vocab_size: int | None = None,
    compression_reserved_ids: int = 0,
) -> dict[int | None, dict[str, torch.Tensor]]:
    lookup_state: dict[int | None, dict[str, torch.Tensor]] = {}
    offsets = (torch.arange(heads, device=token_ids.device, dtype=torch.long) * slots).view(1, 1, heads)
    layer_ids = layer_ids or (None,)

    x = token_ids.to(torch.long)
    if compressed_vocab_size is not None and compressed_vocab_size > 0:
        if compression_reserved_ids > 0:
            reserve_start = max(int(x.max().item()) - compression_reserved_ids + 1, 0)
            compress_mask = x < reserve_start
            bucket_size = max(1, reserve_start // compressed_vocab_size)
            compressed = torch.div(x.clamp_max(max(reserve_start - 1, 0)), bucket_size, rounding_mode="floor")
            compressed = compressed.clamp_max(compressed_vocab_size - 1)
            x = torch.where(compress_mask, compressed, x)
        else:
            bucket_size = max(1, (int(x.max().item()) + 1) // compressed_vocab_size)
            x = torch.div(x, bucket_size, rounding_mode="floor").clamp_max(compressed_vocab_size - 1)

    for layer_id in layer_ids:
        layer_lookup: dict[str, torch.Tensor] = {}
        layer_shift = 0 if (layer_id is None or not use_layerwise_hash) else int(layer_id) * 8191

        for n in tuple(sorted(ngram_orders)):
            bsz, seq_len = x.shape
            if n > 1:
                pad = torch.zeros(bsz, n - 1, device=x.device, dtype=x.dtype)
                padded = torch.cat([pad, x], dim=1)
            else:
                padded = x

            gram = torch.stack([padded[:, offset : offset + seq_len] for offset in range(n)], dim=-1)
            seeds = torch.arange(1, n + 1, device=x.device, dtype=torch.long) * (1315423911 + n * 2654435761 + layer_shift)
            head_offsets = torch.arange(heads, device=x.device, dtype=torch.long).view(1, 1, heads, 1) * (104729 + layer_shift)
            head_seeds = seeds.view(1, 1, 1, n) + head_offsets

            hashes = (gram.unsqueeze(-2) * head_seeds).sum(dim=-1)
            hashes = torch.bitwise_xor(hashes, torch.bitwise_right_shift(hashes, 13))
            layer_lookup[str(n)] = torch.remainder(hashes.abs(), slots) + offsets
        lookup_state[layer_id] = layer_lookup

    return lookup_state


class EngramMemory(nn.Module):
    """
    Hash-based conditional memory with a cheaper lookup path:
    - deterministic per-head n-gram hashing
    - direct indexed embedding fetch without table expansion
    - context-aware order selection
    - lightweight causal depthwise mixing
    """

    def __init__(
        self,
        d_model: int,
        *,
        slots: int,
        heads: int,
        top_k: int,
        memory_dim: int | None = None,
        ngram_orders: tuple[int, ...] = (2, 3),
        layer_idx: int | None = None,
        use_layerwise_hash: bool = True,
        compressed_vocab_size: int | None = None,
        compression_reserved_ids: int = 0,
        official_gating: bool = False,
        conv_enabled: bool = True,
        long_conv_threshold: int | None = None,
        long_conv_enabled: bool = True,
        conv_kernel_size: int = 4,
        conv_dilation: int = 3,
        conv_bottleneck_ratio: float = 1.0,
        conv_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.top_k = top_k
        self.memory_dim = memory_dim or max(d_model // heads, 1)
        self.ngram_orders = tuple(sorted(ngram_orders))
        self.num_orders = len(self.ngram_orders)
        self.slots = slots
        self.layer_idx = layer_idx
        self.use_layerwise_hash = use_layerwise_hash
        self.compressed_vocab_size = compressed_vocab_size
        self.compression_reserved_ids = compression_reserved_ids
        self.official_gating = official_gating
        self.conv_enabled = conv_enabled
        self.long_conv_threshold = long_conv_threshold
        self.long_conv_enabled = long_conv_enabled
        self.conv_zero_init = conv_zero_init

        self.order_embeddings = nn.ModuleDict()
        self.hash_seed_buffers: dict[int, str] = {}
        for n in self.ngram_orders:
            emb = nn.Embedding(heads * slots, self.memory_dim)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self.order_embeddings[str(n)] = emb

            seed_name = f"hash_seeds_{n}"
            self.register_buffer(
                seed_name,
                torch.arange(1, n + 1, dtype=torch.long) * (1315423911 + n * 2654435761),
                persistent=False,
            )
            self.hash_seed_buffers[n] = seed_name

        self.register_buffer(
            "head_offsets",
            torch.arange(heads, dtype=torch.long) * slots,
            persistent=False,
        )

        self.query_norm = RMSNorm(d_model)
        self.query_proj = nn.Linear(d_model, heads * self.memory_dim, bias=False)
        self.control_proj = nn.Linear(d_model, heads * (self.num_orders + 1), bias=True)
        self.gate_key_proj = nn.Linear(heads * self.memory_dim * self.num_orders, d_model, bias=False)
        self.gate_key_norm = RMSNorm(d_model)
        self.gate_query_norm = RMSNorm(d_model)
        self.out_norm = RMSNorm(heads * self.memory_dim)
        self.out_proj = nn.Linear(heads * self.memory_dim, d_model, bias=False)
        self.conv_dim = max(1, int(d_model * conv_bottleneck_ratio))
        if self.conv_dim != d_model:
            self.pre_conv_proj = nn.Linear(d_model, self.conv_dim, bias=False)
            self.post_conv_proj = nn.Linear(self.conv_dim, d_model, bias=False)
        else:
            self.pre_conv_proj = nn.Identity()
            self.post_conv_proj = nn.Identity()

        if self.conv_enabled:
            self.depthwise = nn.Conv1d(
                self.conv_dim,
                self.conv_dim,
                kernel_size=conv_kernel_size,
                dilation=conv_dilation,
                groups=self.conv_dim,
                bias=True,
                padding=0,
            )
            if self.conv_zero_init:
                nn.init.zeros_(self.depthwise.weight)
                nn.init.zeros_(self.depthwise.bias)
                if isinstance(self.post_conv_proj, nn.Linear):
                    nn.init.zeros_(self.post_conv_proj.weight)
        else:
            self.depthwise = None

    def _lookup_memory(
        self,
        token_ids: torch.Tensor,
        lookup_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        pieces = []
        for n in self.ngram_orders:
            layer_lookup = lookup_state.get(self.layer_idx) if lookup_state is not None else None
            indices = layer_lookup[str(n)] if layer_lookup is not None else build_engram_lookup_state(
                token_ids,
                ngram_orders=(n,),
                heads=self.heads,
                slots=self.slots,
                layer_ids=(self.layer_idx,),
                use_layerwise_hash=self.use_layerwise_hash,
                compressed_vocab_size=self.compressed_vocab_size,
                compression_reserved_ids=self.compression_reserved_ids,
            )[self.layer_idx][str(n)]
            piece = self.order_embeddings[str(n)](indices)
            pieces.append(piece)
        return torch.stack(pieces, dim=-2)

    def _causal_depthwise(self, x: torch.Tensor) -> torch.Tensor:
        use_conv = self.conv_enabled
        if self.long_conv_threshold is not None and x.size(1) >= self.long_conv_threshold:
            use_conv = self.long_conv_enabled
        if not use_conv:
            return x
        kernel = self.depthwise.kernel_size[0]
        dilation = self.depthwise.dilation[0]
        left_pad = dilation * (kernel - 1)
        y = self.pre_conv_proj(x)
        y = torch.nn.functional.pad(y.transpose(1, 2), (left_pad, 0))
        y = self.depthwise(y).transpose(1, 2)
        return self.post_conv_proj(y)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        lookup_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mem = self._lookup_memory(token_ids, lookup_state=lookup_state)
        normed_x = self.query_norm(x)
        q = self.query_proj(normed_x).view(x.size(0), x.size(1), self.heads, self.memory_dim)
        controls = self.control_proj(normed_x).view(x.size(0), x.size(1), self.heads, self.num_orders + 1)
        order_bias = controls[..., : self.num_orders]
        head_gate = torch.sigmoid(controls[..., self.num_orders :])

        if self.num_orders == 2:
            base = mem[..., 0, :]
            delta = mem[..., 1, :] - base
            bias_delta = order_bias[..., 1] - order_bias[..., 0]
            alpha = torch.sigmoid(((q * delta).sum(dim=-1) / math.sqrt(self.memory_dim) + bias_delta).unsqueeze(-1))
            mixed = base + alpha * delta
        else:
            order_scores = (q.unsqueeze(-2) * mem).sum(dim=-1) / math.sqrt(self.memory_dim)
            order_scores = order_scores + order_bias
            order_weights = torch.softmax(order_scores, dim=-1)
            mixed = (order_weights.unsqueeze(-1) * mem).sum(dim=-2)
        mixed = mixed * head_gate

        mixed = mixed.reshape(x.size(0), x.size(1), self.heads * self.memory_dim)
        gated = self.out_proj(self.out_norm(mixed))
        if self.official_gating:
            flat_mem = mem.reshape(x.size(0), x.size(1), -1)
            gate_key = self.gate_key_norm(self.gate_key_proj(flat_mem))
            gate_query = self.gate_query_norm(x)
            gate_score = (gate_key * gate_query).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
            gate_score = gate_score.abs().clamp_min(1e-6).sqrt() * gate_score.sign()
            gated = torch.sigmoid(gate_score) * gated
        return torch.nn.functional.silu(self._causal_depthwise(gated) + gated)
