from __future__ import annotations

from torch import nn

from .attention import MultiHeadAttention
from .config import TransformerConfig
from .mlp import FeedForward
from .moe import SparseMoE
from .norms import RMSNorm
from .residual import MultiBranchResidual, MHCResidual
from .engram import EngramMemory


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, *, layer_idx: int, is_global_layer: bool = True) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        attn_cfg = config.attention
        local_window = None if is_global_layer else attn_cfg.local_window
        norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm1 = norm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = norm(config.d_model, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(
            config.d_model,
            num_heads=attn_cfg.num_heads,
            num_kv_heads=attn_cfg.num_kv_heads,
            dropout=attn_cfg.dropout,
            rope_base=attn_cfg.rope_base,
            local_window=local_window,
            qk_norm=attn_cfg.qk_norm,
            tie_kv=attn_cfg.tie_kv,
            use_dsa=attn_cfg.use_dsa,
            dsa_top_k=attn_cfg.dsa_top_k,
            dsa_indexer_hidden=attn_cfg.dsa_indexer_hidden,
            backend=attn_cfg.backend,
        )
        self.engram = (
            EngramMemory(
                config.d_model,
                slots=config.engram.slots,
                heads=config.engram.heads,
                top_k=config.engram.top_k,
                memory_dim=config.engram.memory_dim,
                ngram_orders=config.engram.ngram_orders,
                layer_idx=layer_idx,
                use_layerwise_hash=config.engram.use_layerwise_hash,
                compressed_vocab_size=config.engram.compressed_vocab_size,
                compression_reserved_ids=config.engram.compression_reserved_ids,
                official_gating=config.engram.official_gating,
                conv_enabled=config.engram.conv_enabled,
                long_conv_threshold=config.engram.long_conv_threshold,
                long_conv_enabled=config.engram.long_conv_enabled,
                conv_kernel_size=config.engram.conv_kernel_size,
                conv_dilation=config.engram.conv_dilation,
                conv_bottleneck_ratio=config.engram.conv_bottleneck_ratio,
                conv_zero_init=config.engram.conv_zero_init,
            )
            if config.engram.enabled and layer_idx in config.engram.insert_layers
            else None
        )
        hidden_dim = int(config.d_model * config.mlp_ratio)
        if config.moe.enabled:
            self.ff = SparseMoE(
                config.d_model,
                hidden_dim,
                activation=config.activation,
                num_experts=config.moe.num_experts,
                top_k=config.moe.top_k,
                shared_expert=config.moe.shared_expert,
                dropout=config.dropout,
                router_jitter=config.moe.router_jitter,
            )
        else:
            self.ff = FeedForward(config.d_model, hidden_dim, activation=config.activation, dropout=config.dropout)
        if config.use_mhc:
            self.branch = MHCResidual(config.residual_branches, config.d_model)
        elif config.use_multibranch_residual:
            self.branch = MultiBranchResidual(config.residual_branches, config.d_model)
        else:
            self.branch = None

    def _apply_residual_branch(self, x, update):
        if self.branch is None:
            return x + update
        if isinstance(self.branch, MHCResidual):
            return x + self.branch(x, update)

        streams = [x, update]
        if self.branch.branches >= 3:
            streams.append(x + update)
        if self.branch.branches >= 4:
            streams.append(x - update)
        while len(streams) < self.branch.branches:
            streams.append(update)
        return x + self.branch(*streams[: self.branch.branches])

    def forward(self, x, *, token_ids=None, attn_mask=None, engram_lookup_state=None):
        attn_out = self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            use_dsa=self.attn.use_dsa and self.attn.local_window is not None,
            global_stride=4,
        )
        x = self._apply_residual_branch(x, attn_out)

        if self.engram is not None:
            if token_ids is None:
                raise ValueError("Engram-enabled block requires token_ids/byte_ids for n-gram lookup")
            engram_out = self.engram(self.norm2(x), token_ids=token_ids, lookup_state=engram_lookup_state)
            x = self._apply_residual_branch(x, engram_out)

        ff_out = self.ff(self.norm2(x))
        x = self._apply_residual_branch(x, ff_out)
        return x
