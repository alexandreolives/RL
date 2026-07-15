from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .norms import RMSNorm
from .rope import apply_rope, apply_rope_single
from .cache import DeepseekV4LayerCache

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        num_kv_heads: int | None = None,
        q_lora_rank: int | None = None,
        q_lora_norm: bool = False,
        kv_norm: bool = False,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_base: int = 10_000,
        local_window: int | None = None,
        qk_norm: bool = False,
        tie_kv: bool = False,
        use_dsa: bool = False,
        dsa_top_k: int = 256,
        dsa_indexer_hidden: int = 128,
        compress_rate_csa: int | None = None,
        compress_rate_hca: int | None = None,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        partial_rotary_factor: float = 1.0,
        partial_rope_on_tail: bool = False,
        rotate_output_rope: bool = False,
        compress_rope_base: int = 160_000,
        rope_scaling: dict[str, float | int | str] | None = None,
        use_attention_sink: bool = False,
        csa_overlap: bool = False,
        csa_window_factor: int = 1,
        learned_compression: bool = False,
        grouped_o_proj: bool = False,
        o_groups: int = 1,
        o_lora_rank: int = 64,
        kv_cache_storage_dtype: str | None = None,
        index_cache_storage_dtype: str | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        self.local_window = local_window
        self.rope_base = rope_base
        self.qk_norm = qk_norm
        self.tie_kv = tie_kv
        self.use_dsa = use_dsa
        self.global_stride = None
        self.dsa_top_k = dsa_top_k
        self.backend = backend
        self.compress_rates = None if compress_rate_csa is None and compress_rate_hca is None else {
            "compressed_sparse_attention": compress_rate_csa,
            "heavily_compressed_attention": compress_rate_hca,
        }
        self.compress_rate_csa = compress_rate_csa
        self.compress_rate_hca = compress_rate_hca
        self.index_topk = index_topk
        self.partial_rotary_factor = partial_rotary_factor
        self.partial_rope_on_tail = partial_rope_on_tail
        self.rotate_output_rope = rotate_output_rope
        self.compress_rope_base = compress_rope_base
        self.rope_scaling = rope_scaling
        self.use_attention_sink = use_attention_sink
        self.csa_overlap = csa_overlap
        self.csa_window_factor = max(1, csa_window_factor)
        self.learned_compression = learned_compression
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.kv_cache_storage_dtype = kv_cache_storage_dtype
        self.index_cache_storage_dtype = index_cache_storage_dtype

        self.q_proj = (
            LowRankLinear(
                d_model,
                num_heads * self.head_dim,
                rank=q_lora_rank,
                norm=RMSNorm(q_lora_rank, eps=rms_norm_eps) if q_lora_norm else None,
            )
            if q_lora_rank is not None and q_lora_rank > 0
            else nn.Linear(d_model, num_heads * self.head_dim)
        )
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = None if tie_kv else nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if kv_norm else nn.Identity()
        self.out_proj = (
            GroupedLowRankOutputProjection(d_model, num_heads=num_heads, groups=o_groups, rank=o_lora_rank)
            if grouped_o_proj
            else nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.indexer = nn.Sequential(
            nn.Linear(d_model, dsa_indexer_hidden),
            nn.SiLU(),
            nn.Linear(dsa_indexer_hidden, dsa_indexer_hidden),
            nn.SiLU(),
        )
        self.indexer_q = nn.Linear(dsa_indexer_hidden, dsa_indexer_hidden, bias=False)
        self.indexer_k = nn.Linear(dsa_indexer_hidden, dsa_indexer_hidden, bias=False)
        self.lightning_indexer = nn.Sequential(
            nn.Linear(self.head_dim, index_head_dim),
            nn.SiLU(),
            nn.Linear(index_head_dim, index_head_dim),
            nn.SiLU(),
        )
        self.lightning_q = nn.Linear(index_head_dim, index_head_dim, bias=False)
        self.lightning_k = nn.Linear(index_head_dim, index_head_dim, bias=False)
        self.attention_sink = nn.Parameter(torch.zeros(num_heads)) if use_attention_sink else None
        if learned_compression and compress_rate_csa is not None:
            self.csa_kv_proj = nn.Linear(d_model, 2 * self.head_dim, bias=False)
            self.csa_gate_proj = nn.Linear(d_model, 2 * self.head_dim, bias=False)
            self.csa_position_bias = nn.Parameter(torch.zeros(compress_rate_csa, 2 * self.head_dim))
            self.csa_kv_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.index_kv_proj = nn.Linear(d_model, 2 * index_head_dim, bias=False)
            self.index_gate_proj = nn.Linear(d_model, 2 * index_head_dim, bias=False)
            self.index_position_bias = nn.Parameter(torch.zeros(compress_rate_csa, 2 * index_head_dim))
            self.index_kv_norm = RMSNorm(index_head_dim, eps=rms_norm_eps)
            self.index_q_proj = nn.Linear(num_heads * self.head_dim, index_n_heads * index_head_dim, bias=False)
            self.index_weights_proj = nn.Linear(d_model, index_n_heads, bias=False)
        else:
            self.csa_kv_proj = None
            self.csa_gate_proj = None
            self.csa_position_bias = None
            self.csa_kv_norm = None
            self.index_kv_proj = None
            self.index_gate_proj = None
            self.index_position_bias = None
            self.index_kv_norm = None
            self.index_q_proj = None
            self.index_weights_proj = None
        if learned_compression and compress_rate_hca is not None:
            self.hca_kv_proj = nn.Linear(d_model, self.head_dim, bias=False)
            self.hca_gate_proj = nn.Linear(d_model, self.head_dim, bias=False)
            self.hca_position_bias = nn.Parameter(torch.zeros(compress_rate_hca, self.head_dim))
            self.hca_kv_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.hca_kv_proj = None
            self.hca_gate_proj = None
            self.hca_position_bias = None
            self.hca_kv_norm = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _cache_dtype(self, name: str | None) -> torch.dtype | None:
        if name is None:
            return None
        if name in {"fp8", "float8", "float8_e4m3fn"}:
            return getattr(torch, "float8_e4m3fn", None)
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16"}:
            return torch.float16
        return None

    def _store_for_cache(self, x: torch.Tensor | None, storage_name: str | None) -> torch.Tensor | None:
        if x is None:
            return None
        storage_dtype = self._cache_dtype(storage_name)
        if storage_dtype is None or x.device.type == "cpu" and storage_dtype in {getattr(torch, "float8_e4m3fn", None)}:
            return x
        return x.to(storage_dtype)

    def _restore_from_cache(self, x: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return x.to(device=device, dtype=dtype)

    def _align_cached_hidden(self, hidden: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 4 and x.dim() == 3 and hidden.size(0) * hidden.size(2) == x.size(0):
            bsz, past_len, hc_mult, d_model = hidden.shape
            return hidden.transpose(1, 2).reshape(bsz * hc_mult, past_len, d_model)
        return hidden

    def _full_attention_inputs(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cache: DeepseekV4LayerCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cache is None or cache.hidden_states is None:
            return x, attn_mask
        past_hidden = self._align_cached_hidden(cache.hidden_states, x).to(device=x.device, dtype=x.dtype)
        full_x = torch.cat([past_hidden, x], dim=1)
        if cache.attn_mask is None and attn_mask is None:
            return full_x, None
        past_mask = cache.attn_mask
        if past_mask is not None:
            past_mask = past_mask.to(device=x.device)
            if past_mask.size(0) != x.size(0) and x.size(0) % past_mask.size(0) == 0:
                past_mask = past_mask.repeat_interleave(x.size(0) // past_mask.size(0), dim=0)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device)
        if past_mask is None:
            past_mask = torch.zeros(x.size(0), full_x.size(1) - x.size(1), device=x.device, dtype=torch.bool)
        if attn_mask is None:
            attn_mask = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=torch.bool)
        return full_x, torch.cat([past_mask.to(torch.bool), attn_mask.to(torch.bool)], dim=1)

    def _local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = idx[None, :] - idx[:, None]
        return (dist > 0) | (dist < -self.local_window)

    def _dsa_mask(self, seq_len: int, device: torch.device, global_stride: int) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        causal = idx[None, :] > idx[:, None]
        local_ok = (idx[None, :] - idx[:, None]).abs() <= (self.local_window or seq_len)
        global_tokens = (idx % global_stride) == 0
        global_ok = global_tokens[None, :].expand(seq_len, -1)
        allowed = (~causal) & (local_ok | global_ok)
        return ~allowed

    def _dsa_index_mask(self, x: torch.Tensor, global_stride: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        idx_hidden = self.indexer(x)
        iq = self.indexer_q(idx_hidden)
        ik = self.indexer_k(idx_hidden)
        scores = torch.matmul(iq, ik.transpose(-1, -2)) / math.sqrt(iq.size(-1))

        base_mask = self._dsa_mask(seq_len, x.device, global_stride)
        scores = scores.masked_fill(base_mask[None, :, :], float("-inf"))

        topk = min(self.dsa_top_k, seq_len)
        top_idx = torch.topk(scores, k=topk, dim=-1).indices
        keep = torch.zeros(bsz, seq_len, seq_len, device=x.device, dtype=torch.bool)
        keep.scatter_(-1, top_idx, True)
        return ~keep

    def _pool_sequence(self, x: torch.Tensor, rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        if rate <= 1 or x.size(-2) <= 1:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device=x.device)
            return x, positions

        bsz, heads, seq_len, dim = x.shape
        pooled = F.avg_pool1d(
            x.reshape(bsz * heads, seq_len, dim).transpose(1, 2),
            kernel_size=rate,
            stride=rate,
            ceil_mode=True,
            count_include_pad=False,
        ).transpose(1, 2)
        pooled_len = pooled.size(1)
        pooled = pooled.reshape(bsz, heads, pooled_len, dim)
        positions = torch.arange(pooled_len, device=x.device, dtype=torch.long)
        positions = torch.minimum((positions + 1) * rate - 1, torch.full_like(positions, seq_len - 1))
        return pooled, positions

    def _pool_sequence_overlap(self, x: torch.Tensor, rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = max(1, rate * self.csa_window_factor)
        if rate <= 1 or x.size(-2) <= 1:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device=x.device)
            return x, positions
        bsz, heads, seq_len, dim = x.shape
        if seq_len < window:
            return x.new_empty(bsz, heads, 0, dim), torch.empty(0, device=x.device, dtype=torch.long)
        windows = x.unfold(dimension=-2, size=window, step=rate)
        if self.csa_compress_weight is not None and self.csa_compress_weight.size(-1) == window:
            weights = torch.softmax(self.csa_compress_weight[:heads], dim=-1).view(1, heads, 1, 1, window)
            pooled = (windows * weights).sum(dim=-1)
        else:
            pooled = windows.mean(dim=-1)
        positions = torch.arange(window - 1, seq_len, step=rate, device=x.device, dtype=torch.long)
        return pooled, positions

    def _pool_sequence_weighted(self, x: torch.Tensor, rate: int, weight: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if rate <= 1 or x.size(-2) <= 1:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device=x.device)
            return x, positions
        bsz, heads, seq_len, dim = x.shape
        if seq_len < rate:
            return x.new_empty(bsz, heads, 0, dim), torch.empty(0, device=x.device, dtype=torch.long)
        windows = x.unfold(dimension=-2, size=rate, step=rate)
        if weight is not None and weight.size(-1) == rate:
            weights = torch.softmax(weight[:heads], dim=-1).view(1, heads, 1, 1, rate)
            pooled = (windows * weights).sum(dim=-1)
        else:
            pooled = windows.mean(dim=-1)
        positions = torch.arange(rate - 1, seq_len, step=rate, device=x.device, dtype=torch.long)
        return pooled, positions

    def _compress_hca_hidden(self, x: torch.Tensor, rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hca_kv_proj is None or x.size(1) < rate:
            empty = x.new_zeros(x.size(0), 1, 0, self.head_dim)
            return empty, torch.empty(0, device=x.device, dtype=torch.long)
        usable = (x.size(1) // rate) * rate
        kv = self.hca_kv_proj(x[:, :usable])
        gate = self.hca_gate_proj(x[:, :usable])
        batch = x.size(0)
        n_windows = usable // rate
        kv = kv.view(batch, n_windows, rate, self.head_dim)
        gate = gate.view(batch, n_windows, rate, self.head_dim) + self.hca_position_bias.to(gate.dtype)
        compressed = (kv * gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)).sum(dim=2)
        compressed = self.hca_kv_norm(compressed)
        positions = torch.arange(n_windows, device=x.device, dtype=torch.long) * rate
        compressed = apply_rope_single(
            compressed.unsqueeze(1),
            base=self.compress_rope_base,
            partial_rotary_factor=self.partial_rotary_factor,
            on_tail=self.partial_rope_on_tail,
            positions=positions,
            rope_scaling=self.rope_scaling,
        )
        return compressed, positions

    def _compress_csa_hidden(
        self,
        x: torch.Tensor,
        rate: int,
        *,
        for_indexer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_proj = self.index_kv_proj if for_indexer else self.csa_kv_proj
        gate_proj = self.index_gate_proj if for_indexer else self.csa_gate_proj
        position_bias = self.index_position_bias if for_indexer else self.csa_position_bias
        norm = self.index_kv_norm if for_indexer else self.csa_kv_norm
        dim = self.index_head_dim if for_indexer else self.head_dim
        if kv_proj is None or x.size(1) < rate:
            return x.new_zeros(x.size(0), 0, dim), torch.empty(0, device=x.device, dtype=torch.long)

        usable = (x.size(1) // rate) * rate
        kv = kv_proj(x[:, :usable])
        gate = gate_proj(x[:, :usable])
        batch = x.size(0)
        n_windows = usable // rate
        kv = kv.view(batch, n_windows, rate, 2 * dim)
        gate = gate.view(batch, n_windows, rate, 2 * dim) + position_bias.to(gate.dtype)

        new_kv = kv.new_zeros((batch, n_windows, 2 * rate, dim))
        new_gate = gate.new_full((batch, n_windows, 2 * rate, dim), float("-inf"))
        new_kv[:, :, rate:] = kv[..., dim:]
        new_gate[:, :, rate:] = gate[..., dim:]
        if n_windows > 1:
            new_kv[:, 1:, :rate] = kv[:, :-1, :, :dim]
            new_gate[:, 1:, :rate] = gate[:, :-1, :, :dim]

        compressed = (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
        compressed = norm(compressed)
        positions = torch.arange(n_windows, device=x.device, dtype=torch.long) * rate
        compressed = apply_rope_single(
            compressed.unsqueeze(1),
            base=self.compress_rope_base,
            partial_rotary_factor=self.partial_rotary_factor,
            on_tail=self.partial_rope_on_tail,
            positions=positions,
            rope_scaling=self.rope_scaling,
        ).squeeze(1)
        return compressed, positions

    def _csa_index_mask_from_hidden(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        compressed_len: int,
        rate: int,
        query_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            compressed_len == 0
            or self.index_q_proj is None
            or self.index_weights_proj is None
            or self.index_kv_proj is None
        ):
            return torch.zeros(x.size(0), q.size(-2), compressed_len, device=x.device, dtype=torch.bool)

        index_kv, _ = self._compress_csa_hidden(x, rate, for_indexer=True)
        index_kv = index_kv[:, :compressed_len]
        bsz = x.size(0)
        query_len = q.size(-2)
        q_flat = q.transpose(1, 2).reshape(bsz, query_len, self.num_heads * self.head_dim)
        index_q = self.index_q_proj(q_flat).view(bsz, query_len, self.index_n_heads, self.index_head_dim)
        index_q = apply_rope_single(
            index_q.transpose(1, 2),
            base=self.compress_rope_base,
            partial_rotary_factor=self.partial_rotary_factor,
            on_tail=self.partial_rope_on_tail,
            rope_scaling=self.rope_scaling,
        ).transpose(1, 2)
        scores = torch.matmul(index_q.float(), index_kv.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * (self.index_head_dim**-0.5)
        weights = self.index_weights_proj(x).float() * (self.index_n_heads**-0.5)
        query_offset = 0
        if query_positions is not None and query_positions.numel() > 0:
            query_offset = int(query_positions[0].item())
        weights = weights[:, query_offset : query_offset + q.size(-2)]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)

        entry_indices = torch.arange(compressed_len, device=x.device)
        query_pos = query_positions if query_positions is not None else torch.arange(query_len, device=x.device)
        causal_threshold = (query_pos + 1) // rate
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.view(1, -1, 1)
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        topk = min(self.index_topk, compressed_len)
        if topk <= 0:
            return torch.ones_like(future_mask.expand(bsz, -1, -1))
        top_idx = torch.topk(index_scores, k=topk, dim=-1).indices
        keep = torch.zeros_like(index_scores, dtype=torch.bool)
        keep.scatter_(-1, top_idx, True)
        return (~keep) | future_mask

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        base: int,
        q_positions: torch.Tensor | None = None,
        k_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = apply_rope_single(
            q,
            base=base,
            partial_rotary_factor=self.partial_rotary_factor,
            on_tail=self.partial_rope_on_tail,
            positions=q_positions,
            rope_scaling=self.rope_scaling,
        )
        k = apply_rope_single(
            k,
            base=base,
            partial_rotary_factor=self.partial_rotary_factor,
            on_tail=self.partial_rope_on_tail,
            positions=k_positions,
            rope_scaling=self.rope_scaling,
        )
        return q, k

    def _apply_output_rope(self, out: torch.Tensor, *, base: int, positions: torch.Tensor | None = None) -> torch.Tensor:
        if not self.rotate_output_rope:
            return out
        if positions is None:
            positions = torch.arange(out.size(-2), device=out.device)
        positions = -positions
        return apply_rope_single(
            out,
            base=base,
            partial_rotary_factor=self.partial_rotary_factor,
            on_tail=self.partial_rope_on_tail,
            positions=positions,
            rope_scaling=self.rope_scaling,
        )

    def _pool_mask(self, mask: torch.Tensor, rate: int) -> torch.Tensor:
        if rate <= 1 or mask.size(-1) <= 1:
            return mask
        pooled = F.max_pool1d(
            mask.unsqueeze(1).to(torch.float32),
            kernel_size=rate,
            stride=rate,
            ceil_mode=True,
        )
        return pooled.squeeze(1) > 0

    def _pool_mask_overlap(self, mask: torch.Tensor, rate: int) -> torch.Tensor:
        window = max(1, rate * self.csa_window_factor)
        if rate <= 1 or mask.size(-1) <= 1:
            return mask
        if mask.size(-1) < window:
            return mask.new_empty(mask.size(0), 0)
        windows = mask.to(torch.bool).unfold(dimension=-1, size=window, step=rate)
        return windows.any(dim=-1)

    def _apply_attention_sink(self, scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        valid_rows = torch.isfinite(scores).any(dim=-1, keepdim=True)
        safe_scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
        if self.attention_sink is not None:
            sink = self.attention_sink.view(1, -1, 1, 1).expand(scores.size(0), -1, scores.size(-2), -1)
            safe_scores = torch.cat([safe_scores, sink.to(safe_scores.dtype)], dim=-1)
        probs = torch.softmax(safe_scores, dim=-1)
        if self.attention_sink is not None:
            value_probs = probs[..., :-1]
        else:
            value_probs = probs
        value_probs = value_probs * valid_rows.to(value_probs.dtype)
        value_probs = value_probs / value_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_probs = self.dropout(value_probs)
        return torch.matmul(value_probs, v)

    def _attention_v4(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        attn_kind: str,
        query_positions: torch.Tensor | None = None,
        cache: DeepseekV4LayerCache | None = None,
    ) -> torch.Tensor:
        query_len = q.size(-2)
        total_len = k.size(-2)
        query_pos = query_positions if query_positions is not None else torch.arange(query_len, device=x.device)

        if attn_kind == "heavily_compressed_attention":
            rate = max(1, self.compress_rate_hca or 128)
            sliding = self.local_window or total_len
            prefix_len = max(total_len - sliding, 0)
            recent_k = k[..., prefix_len:, :]
            recent_v = v[..., prefix_len:, :]

            if self.learned_compression and self.hca_kv_proj is not None:
                compressed_kv, prefix_pos = self._compress_hca_hidden(x, rate)
                if cache is not None and hasattr(cache, "compressed_pool"):
                    cache.compressed_pool = self._store_for_cache(compressed_kv, self.kv_cache_storage_dtype)
                    cache.compressed_positions = prefix_pos.detach()
                    cache.compressed_count = int(compressed_kv.size(-2))
                compressed_len = compressed_kv.size(-2)
                entry_indices = torch.arange(compressed_len, device=x.device)
                causal_threshold = (query_pos + 1) // rate
                prefix_mask_exp = entry_indices.view(1, 1, -1) >= causal_threshold.view(1, -1, 1)
                prefix_mask_exp = prefix_mask_exp.expand(q.size(0), -1, -1)
                pooled_prefix_k = compressed_kv
                pooled_prefix_v = compressed_kv
            else:
                prefix_k = k[..., :prefix_len, :]
                prefix_v = v[..., :prefix_len, :]
                pooled_prefix_k, prefix_pos = self._pool_sequence_weighted(prefix_k, rate, None) if prefix_len > 0 else (prefix_k, torch.empty(0, device=x.device, dtype=torch.long))
                pooled_prefix_v, _ = self._pool_sequence_weighted(prefix_v, rate, None) if prefix_len > 0 else (prefix_v, torch.empty(0, device=x.device, dtype=torch.long))
                prefix_mask_exp = None

            candidate_pos = torch.cat(
                [prefix_pos.to(x.device), torch.arange(prefix_len, total_len, device=x.device, dtype=torch.long)]
            ) if prefix_pos.numel() > 0 else torch.arange(prefix_len, total_len, device=x.device, dtype=torch.long)
            recent_mask_exp = None
            if attn_mask is not None:
                prefix_mask = self._pool_mask(attn_mask[:, :prefix_len].to(torch.bool), rate) if prefix_len > 0 and not self.learned_compression else None
                recent_mask = attn_mask[:, prefix_len:].to(torch.bool)
                if prefix_mask is not None:
                    prefix_mask_exp = prefix_mask[:, None, :].expand(q.size(0), query_len, -1)
                recent_mask_exp = recent_mask[:, None, :].expand(q.size(0), query_len, -1)
            if pooled_prefix_k.size(-2) > 0:
                if recent_mask_exp is None:
                    recent_mask_exp = torch.zeros((q.size(0), query_len, recent_k.size(-2)), device=x.device, dtype=torch.bool)
                if prefix_mask_exp is None:
                    prefix_mask_exp = torch.zeros((q.size(0), query_len, pooled_prefix_k.size(-2)), device=x.device, dtype=torch.bool)
                candidate_mask = torch.cat([prefix_mask_exp, recent_mask_exp], dim=-1)
                k_for_attn = torch.cat([self._repeat_kv(pooled_prefix_k), self._repeat_kv(recent_k)], dim=-2)
                v_for_attn = torch.cat([self._repeat_kv(pooled_prefix_v), self._repeat_kv(recent_v)], dim=-2)
            else:
                candidate_mask = recent_mask_exp
                k_for_attn = self._repeat_kv(recent_k)
                v_for_attn = self._repeat_kv(recent_v)
            q_for_attn = q
        else:
            rate = max(1, self.compress_rate_csa or 4)
            sliding = self.local_window or total_len
            prefix_len = max(total_len - sliding, 0)
            recent_k = k[..., prefix_len:, :]
            recent_v = v[..., prefix_len:, :]
            if self.learned_compression and self.csa_kv_proj is not None:
                compressed_kv, prefix_pos = self._compress_csa_hidden(x, rate, for_indexer=False)
                if cache is not None and hasattr(cache, "compressed_pool"):
                    cache.compressed_pool = self._store_for_cache(compressed_kv, self.kv_cache_storage_dtype)
                    cache.compressed_positions = prefix_pos.detach()
                    cache.compressed_count = int(compressed_kv.size(-2))
                    index_kv, index_pos = self._compress_csa_hidden(x, rate, for_indexer=True)
                    cache.index_pool = self._store_for_cache(index_kv, self.index_cache_storage_dtype)
                    cache.index_positions = index_pos.detach()
                    cache.index_count = int(index_kv.size(-2))
                pooled_prefix_k = compressed_kv.unsqueeze(1)
                pooled_prefix_v = pooled_prefix_k
                prefix_mask_exp = self._csa_index_mask_from_hidden(
                    x,
                    q,
                    pooled_prefix_k.size(-2),
                    rate,
                    query_positions=query_pos,
                )
            else:
                prefix_k = k[..., :prefix_len, :]
                prefix_v = v[..., :prefix_len, :]
                pool_fn = self._pool_sequence_overlap if self.csa_overlap else self._pool_sequence
                pooled_prefix_k, prefix_pos = pool_fn(prefix_k, rate) if prefix_len > 0 else (prefix_k, torch.empty(0, device=x.device, dtype=torch.long))
                pooled_prefix_v, _ = pool_fn(prefix_v, rate) if prefix_len > 0 else (prefix_v, torch.empty(0, device=x.device, dtype=torch.long))
                prefix_mask_exp = None
            candidate_pos = torch.cat(
                [prefix_pos.to(x.device), torch.arange(prefix_len, total_len, device=x.device, dtype=torch.long)]
            ) if prefix_pos.numel() > 0 else torch.arange(prefix_len, total_len, device=x.device, dtype=torch.long)
            recent_mask_exp = None
            if attn_mask is not None:
                mask_pool_fn = self._pool_mask_overlap if self.csa_overlap else self._pool_mask
                prefix_mask = mask_pool_fn(attn_mask[:, :prefix_len].to(torch.bool), rate) if prefix_len > 0 and not self.learned_compression else None
                recent_mask = attn_mask[:, prefix_len:].to(torch.bool)
                if prefix_mask is not None:
                    prefix_mask_exp = prefix_mask[:, None, :].expand(q.size(0), query_len, -1)
                recent_mask_exp = recent_mask[:, None, :].expand(q.size(0), query_len, -1)

            if pooled_prefix_k.size(-2) > 0:
                prefix_hidden = self.lightning_indexer(pooled_prefix_k.mean(dim=1))
                prefix_keys = self.lightning_k(prefix_hidden)
                query_hidden = self.lightning_indexer(q.mean(dim=1))
                query_keys = self.lightning_q(query_hidden)
                scores = torch.matmul(query_keys, prefix_keys.transpose(-1, -2)) / math.sqrt(self.index_head_dim)
                causal_prefix = prefix_pos[None, None, :].to(x.device) > query_pos[None, :, None]
                scores = scores.masked_fill(causal_prefix, float("-inf"))
                topk = min(self.index_topk, prefix_keys.size(-2))
                if self.learned_compression and self.csa_kv_proj is not None:
                    prefix_drop = prefix_mask_exp
                elif topk < prefix_keys.size(-2):
                    top_idx = torch.topk(scores, k=topk, dim=-1).indices
                    keep_prefix = torch.zeros_like(scores, dtype=torch.bool)
                    keep_prefix.scatter_(-1, top_idx, True)
                    prefix_drop = ~keep_prefix
                else:
                    keep_prefix = torch.ones((q.size(0), query_len, prefix_keys.size(-2)), device=x.device, dtype=torch.bool)
                    prefix_drop = ~keep_prefix
                if prefix_mask_exp is not None:
                    prefix_drop = prefix_drop | prefix_mask_exp
                if recent_mask_exp is None:
                    recent_mask_exp = torch.zeros((q.size(0), query_len, recent_k.size(-2)), device=x.device, dtype=torch.bool)
                candidate_mask = torch.cat([prefix_drop, recent_mask_exp], dim=-1)
                k_for_attn = torch.cat([self._repeat_kv(pooled_prefix_k), self._repeat_kv(recent_k)], dim=-2)
                v_for_attn = torch.cat([self._repeat_kv(pooled_prefix_v), self._repeat_kv(recent_v)], dim=-2)
            else:
                k_for_attn = self._repeat_kv(recent_k)
                v_for_attn = self._repeat_kv(recent_v)
                candidate_mask = recent_mask_exp if recent_mask_exp is not None else None
            q_for_attn = q

        scores = torch.matmul(q_for_attn, k_for_attn.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = candidate_pos[None, None, None, :].to(x.device) > query_pos[None, None, :, None]
        scores = scores.masked_fill(causal, float("-inf"))
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask[:, None, :, :], float("-inf"))
        return self._apply_attention_sink(scores, v_for_attn)

    def _build_block_mask(
        self,
        *,
        x: torch.Tensor,
        seq_len: int,
        attn_mask: torch.Tensor | None,
        use_dsa: bool,
        global_stride: int,
    ) -> torch.Tensor | None:
        block_mask = None
        if use_dsa and self.local_window is not None:
            block_mask = self._dsa_index_mask(x, global_stride)[:, None, :, :]
        elif self.local_window is not None:
            block_mask = self._local_mask(seq_len, x.device)[None, None, :, :]

        if attn_mask is not None:
            padding_mask = attn_mask[:, None, None, :].to(torch.bool)
            block_mask = padding_mask if block_mask is None else (block_mask | padding_mask)
        return block_mask

    def _attention_eager(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        use_dsa: bool,
        global_stride: int,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = q.size(-2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if query_positions is not None and key_positions is not None:
            causal = key_positions[None, :] > query_positions[:, None]
            if self.local_window is not None:
                causal = causal | (key_positions[None, :] < (query_positions[:, None] - self.local_window + 1))
        else:
            causal = torch.triu(torch.ones(seq_len, k.size(-2), device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if query_positions is None or key_positions is None:
            block_mask = self._build_block_mask(
                x=x,
                seq_len=seq_len,
                attn_mask=attn_mask,
                use_dsa=use_dsa,
                global_stride=global_stride,
            )
            if block_mask is not None:
                scores = scores.masked_fill(block_mask, float("-inf"))
        elif attn_mask is not None:
            scores = scores.masked_fill(attn_mask[:, None, None, :].to(torch.bool), float("-inf"))
        return self._apply_attention_sink(scores, v)

    def _attention_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        use_dsa: bool,
        global_stride: int,
    ) -> torch.Tensor:
        seq_len = q.size(-2)
        block_mask = self._build_block_mask(
            x=x,
            seq_len=seq_len,
            attn_mask=attn_mask,
            use_dsa=use_dsa,
            global_stride=global_stride,
        )
        if block_mask is not None:
            additive_mask = torch.zeros_like(block_mask, dtype=q.dtype)
            additive_mask = additive_mask.masked_fill(block_mask, float("-inf"))
        else:
            additive_mask = None
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

    def _can_use_flash(
        self,
        q: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None,
        use_dsa: bool,
    ) -> bool:
        if self.backend == "flash" and flash_attn_func is None:
            raise RuntimeError("attention.backend='flash' requested but flash-attn is not installed")
        return (
            flash_attn_func is not None
            and q.is_cuda
            and q.dtype in {torch.float16, torch.bfloat16}
            and attn_mask is None
            and not use_dsa
            and self.local_window is None
        )

    def _attention_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            dropout_p=self.dropout.p if self.training else 0.0,
            causal=True,
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        *,
        attn_kind: str | None = None,
        use_dsa: bool = False,
        global_stride: int = 4,
        cache: DeepseekV4LayerCache | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = k if self.tie_kv else self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self.kv_norm(k)
        k_cache = k
        v_cache = v
        past_len = 0
        if cache is not None and cache.key_states is not None and cache.value_states is not None:
            past_k = self._restore_from_cache(cache.key_states, device=k.device, dtype=k.dtype)
            past_v = self._restore_from_cache(cache.value_states, device=v.device, dtype=v.dtype)
            past_len = past_k.size(-2)
            k_cache = torch.cat([past_k, k_cache], dim=-2)
            v_cache = torch.cat([past_v, v_cache], dim=-2)
        total_len = k_cache.size(-2)
        query_positions = torch.arange(past_len, past_len + seq_len, device=x.device, dtype=torch.long)
        key_positions = torch.arange(total_len, device=x.device, dtype=torch.long)
        full_x, full_attn_mask = self._full_attention_inputs(x, attn_mask, cache)

        q = self.q_norm(q)
        k = self.k_norm(k_cache)
        if attn_kind in {"compressed_sparse_attention", "heavily_compressed_attention"} and self.compress_rates is not None:
            base = self.compress_rope_base
            q, k = self._apply_rope(q, k, base=base, q_positions=query_positions, k_positions=key_positions)
            attn_out = self._attention_v4(
                q,
                k,
                v_cache,
                x=full_x,
                attn_mask=full_attn_mask,
                attn_kind=attn_kind,
                query_positions=query_positions,
                cache=cache,
            )
        else:
            q, k = self._apply_rope(q, k, base=self.rope_base, q_positions=query_positions, k_positions=key_positions)
            flash_ready = past_len == 0 and self._can_use_flash(q, attn_mask=attn_mask, use_dsa=use_dsa)
            if flash_ready:
                attn_out = self._attention_flash(q, self._repeat_kv(k), self._repeat_kv(v_cache))
            else:
                k = self._repeat_kv(k)
                v = self._repeat_kv(v_cache)
                use_sdpa = (
                    past_len == 0
                    and self.backend in {"auto", "sdpa", "flash"}
                    and not use_dsa
                    and self.attention_sink is None
                )
                if use_sdpa:
                    attn_out = self._attention_sdpa(
                        q,
                        k,
                        v,
                        x=x,
                        attn_mask=attn_mask,
                        use_dsa=use_dsa,
                        global_stride=global_stride,
                    )
                else:
                    attn_out = self._attention_eager(
                        q,
                        k,
                        v,
                        x=x,
                        attn_mask=full_attn_mask if past_len > 0 else attn_mask,
                        use_dsa=use_dsa,
                        global_stride=global_stride,
                        query_positions=query_positions if past_len > 0 else None,
                        key_positions=key_positions if past_len > 0 else None,
                    )

        rope_base = self.compress_rope_base if attn_kind in {"compressed_sparse_attention", "heavily_compressed_attention"} else self.rope_base
        attn_out = self._apply_output_rope(attn_out, base=rope_base, positions=query_positions)
        if cache is not None:
            cache.set_kv(
                self._store_for_cache(k_cache, self.kv_cache_storage_dtype),
                self._store_for_cache(v_cache, self.kv_cache_storage_dtype),
            )
        out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)


class GroupedLowRankOutputProjection(nn.Module):
    def __init__(self, d_model: int, *, num_heads: int, groups: int, rank: int) -> None:
        super().__init__()
        if groups <= 0 or num_heads % groups != 0:
            raise ValueError(f"o_groups must divide num_heads, got groups={groups}, num_heads={num_heads}")
        self.groups = groups
        self.group_dim = d_model // groups
        self.rank = rank
        self.o_a_weight = nn.Parameter(torch.empty(groups, self.group_dim, rank))
        self.o_a_bias = nn.Parameter(torch.zeros(groups, rank))
        self.o_b_proj = nn.Linear(groups * rank, d_model, bias=False)
        nn.init.xavier_uniform_(self.o_a_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        grouped = x.view(bsz, seq_len, self.groups, self.group_dim)
        low_rank = torch.einsum("bsgd,gdr->bsgr", grouped, self.o_a_weight)
        low_rank = low_rank + self.o_a_bias.view(1, 1, self.groups, self.rank)
        return self.o_b_proj(low_rank.reshape(bsz, seq_len, self.groups * self.rank))


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, rank: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.norm = norm if norm is not None else nn.Identity()
        self.up = nn.Linear(rank, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.norm(self.down(x)))
