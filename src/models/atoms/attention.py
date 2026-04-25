from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .rope import apply_rope

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
        dropout: float = 0.0,
        rope_base: int = 10_000,
        local_window: int | None = None,
        qk_norm: bool = False,
        tie_kv: bool = False,
        use_dsa: bool = False,
        dsa_top_k: int = 256,
        dsa_indexer_hidden: int = 128,
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

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = None if tie_kv else nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
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

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

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
    ) -> torch.Tensor:
        seq_len = q.size(-2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        block_mask = self._build_block_mask(
            x=x,
            seq_len=seq_len,
            attn_mask=attn_mask,
            use_dsa=use_dsa,
            global_stride=global_stride,
        )
        if block_mask is not None:
            scores = scores.masked_fill(block_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, v)

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
        use_dsa: bool = False,
        global_stride: int = 4,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = k if self.tie_kv else self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, base=self.rope_base)
        flash_ready = self._can_use_flash(q, attn_mask=attn_mask, use_dsa=use_dsa)
        if flash_ready:
            attn_out = self._attention_flash(q, k, v)
        else:
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
            use_sdpa = self.backend in {"auto", "sdpa", "flash"} and not use_dsa
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
                    attn_mask=attn_mask,
                    use_dsa=use_dsa,
                    global_stride=global_stride,
                )

        out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)
