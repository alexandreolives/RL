from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention
from .config import TransformerConfig
from .mlp import FeedForward
from .moe import SparseMoE
from .norms import RMSNorm
from .residual import DeepseekV4HyperConnection, FullAttentionResidual, MultiBranchResidual, MHCResidual
from .engram import EngramMemory
from .cache import DeepseekV4LayerCache


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        layer_idx: int,
        is_global_layer: bool = True,
        attn_kind: str | None = None,
        mlp_kind: str | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_kind = attn_kind
        self.mlp_kind = mlp_kind
        self.use_mhc_streams = config.use_mhc_streams
        self.use_attnres = config.use_attnres
        self.attnres_engram_mode = getattr(config, "attnres_engram_mode", "source")
        if self.attnres_engram_mode not in {"source", "fused", "bypass"}:
            raise ValueError(f"Unknown AttnRes Engram mode: {self.attnres_engram_mode}")
        self.hc_mult = config.hc_mult if self.use_mhc_streams else 1
        attn_cfg = config.attention
        if attn_kind in {"sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"}:
            local_window = attn_cfg.sliding_window
        else:
            local_window = None if is_global_layer else attn_cfg.local_window
        norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm1 = norm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = norm(config.d_model, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(
            config.d_model,
            num_heads=attn_cfg.num_heads,
            num_kv_heads=attn_cfg.num_kv_heads,
            q_lora_rank=attn_cfg.q_lora_rank,
            q_lora_norm=attn_cfg.q_lora_norm,
            kv_norm=attn_cfg.kv_norm,
            dropout=attn_cfg.dropout,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=attn_cfg.rope_base,
            local_window=local_window,
            qk_norm=attn_cfg.qk_norm,
            tie_kv=attn_cfg.tie_kv,
            use_dsa=attn_cfg.use_dsa,
            dsa_top_k=attn_cfg.dsa_top_k,
            dsa_indexer_hidden=attn_cfg.dsa_indexer_hidden,
            compress_rate_csa=(attn_cfg.compress_rates or {}).get("compressed_sparse_attention") if attn_cfg.compress_rates else None,
            compress_rate_hca=(attn_cfg.compress_rates or {}).get("heavily_compressed_attention") if attn_cfg.compress_rates else None,
            index_n_heads=attn_cfg.index_n_heads,
            index_head_dim=attn_cfg.index_head_dim,
            index_topk=attn_cfg.index_topk,
            partial_rotary_factor=attn_cfg.partial_rotary_factor,
            partial_rope_on_tail=attn_cfg.partial_rope_on_tail,
            rotate_output_rope=attn_cfg.rotate_output_rope,
            compress_rope_base=attn_cfg.compress_rope_base,
            rope_scaling=attn_cfg.rope_scaling,
            use_attention_sink=attn_cfg.use_attention_sink,
            csa_overlap=attn_cfg.csa_overlap,
            csa_window_factor=attn_cfg.csa_window_factor,
            learned_compression=attn_cfg.learned_compression,
            grouped_o_proj=attn_cfg.grouped_o_proj,
            o_groups=attn_cfg.o_groups,
            o_lora_rank=attn_cfg.o_lora_rank,
            kv_cache_storage_dtype=attn_cfg.kv_cache_storage_dtype,
            index_cache_storage_dtype=attn_cfg.index_cache_storage_dtype,
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
            routing_mode = "hash_moe" if (
                mlp_kind == "hash_moe"
                or (mlp_kind is None and config.num_hash_layers is not None and layer_idx < config.num_hash_layers)
            ) else "moe"
            self.ff = SparseMoE(
                config.d_model,
                hidden_dim,
                activation=config.activation,
                num_experts=config.moe.num_experts,
                top_k=config.moe.top_k,
                shared_expert=config.moe.shared_expert,
                scoring_func=config.moe.scoring_func,
                topk_method=config.moe.topk_method,
                norm_topk_prob=config.moe.norm_topk_prob,
                routed_scaling_factor=config.moe.routed_scaling_factor,
                dropout=config.dropout,
                router_jitter=config.moe.router_jitter,
                swiglu_limit=config.moe.swiglu_limit,
                routing_mode=routing_mode,
                hash_vocab_size=config.vocab_size if not config.use_byte_first else config.bytes.vocab_size,
                layer_idx=layer_idx,
            )
        else:
            self.ff = FeedForward(config.d_model, hidden_dim, activation=config.activation, dropout=config.dropout)
        if config.use_mhc:
            if self.use_mhc_streams:
                self.attn_hc = DeepseekV4HyperConnection(
                    self.hc_mult,
                    config.d_model,
                    sinkhorn_iters=config.mhc_sinkhorn_iters,
                    eps=config.mhc_eps,
                )
                self.ffn_hc = DeepseekV4HyperConnection(
                    self.hc_mult,
                    config.d_model,
                    sinkhorn_iters=config.mhc_sinkhorn_iters,
                    eps=config.mhc_eps,
                )
                self.branch = None
            else:
                self.attn_hc = None
                self.ffn_hc = None
                self.branch = MHCResidual(
                    config.residual_branches,
                    config.d_model,
                    sinkhorn_iters=config.mhc_sinkhorn_iters,
                    eps=config.mhc_eps,
                )
        elif config.use_multibranch_residual:
            self.branch = MultiBranchResidual(config.residual_branches, config.d_model)
        else:
            self.attn_hc = None
            self.ffn_hc = None
            self.branch = None
        self.attn_res_attn = (
            FullAttentionResidual(config.d_model, eps=config.rms_norm_eps)
            if self.use_attnres
            else None
        )
        self.attn_res_engram = (
            FullAttentionResidual(config.d_model, eps=config.rms_norm_eps)
            if self.use_attnres and self.engram is not None
            else None
        )
        self.engram_bypass_gate = (
            nn.Parameter(torch.ones(config.d_model))
            if self.attnres_engram_mode == "bypass" and self.engram is not None
            else None
        )
        self.attn_res_ffn = (
            FullAttentionResidual(config.d_model, eps=config.rms_norm_eps)
            if self.use_attnres
            else None
        )

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

    def _flatten_streams(self, x):
        if not self.use_mhc_streams:
            return x, None
        bsz, seq_len, hc_mult, d_model = x.shape
        return x.reshape(bsz * hc_mult, seq_len, d_model), (bsz, seq_len, hc_mult, d_model)

    def _unflatten_streams(self, x, shape):
        if shape is None:
            return x
        bsz, seq_len, hc_mult, d_model = shape
        return x.reshape(bsz, hc_mult, seq_len, d_model).transpose(1, 2).contiguous()

    def forward(self, x, *, token_ids=None, attn_mask=None, engram_lookup_state=None, cache: DeepseekV4LayerCache | None = None):
        if self.use_attnres:
            raise RuntimeError("AttnRes blocks must be called through forward_attnres")
        use_dsa = self.attn.use_dsa and self.attn.local_window is not None and self.attn_kind == "compressed_sparse_attention"
        if self.use_mhc_streams:
            stream_token_ids = token_ids.repeat_interleave(self.hc_mult, dim=0) if token_ids is not None else None
            stream_attn_mask = attn_mask.repeat_interleave(self.hc_mult, dim=0) if attn_mask is not None else None
            attn_x, shape = self._flatten_streams(x)
            attn_out = self.attn(
                self.norm1(attn_x),
                attn_mask=stream_attn_mask,
                attn_kind=self.attn_kind,
                use_dsa=use_dsa,
                global_stride=4,
                cache=cache,
            )
            attn_out = self._unflatten_streams(attn_out, shape)
            x = self.attn_hc(x, attn_out)

            if self.engram is not None:
                if stream_token_ids is None:
                    raise ValueError("Engram-enabled block requires token_ids/byte_ids for n-gram lookup")
                engram_in, shape = self._flatten_streams(x)
                engram_out = self.engram(self.norm2(engram_in), token_ids=stream_token_ids, lookup_state=engram_lookup_state)
                engram_out = self._unflatten_streams(engram_out, shape)
                x = self.ffn_hc(x, engram_out)

            ff_x, shape = self._flatten_streams(x)
            if isinstance(self.ff, SparseMoE):
                ff_out = self.ff(self.norm2(ff_x), token_ids=stream_token_ids)
            else:
                ff_out = self.ff(self.norm2(ff_x))
            ff_out = self._unflatten_streams(ff_out, shape)
            x = self.ffn_hc(x, ff_out)
            return x

        attn_out = self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            attn_kind=self.attn_kind,
            use_dsa=use_dsa,
            global_stride=4,
            cache=cache,
        )
        x = self._apply_residual_branch(x, attn_out)

        if self.engram is not None:
            if token_ids is None:
                raise ValueError("Engram-enabled block requires token_ids/byte_ids for n-gram lookup")
            engram_out = self.engram(self.norm2(x), token_ids=token_ids, lookup_state=engram_lookup_state)
            x = self._apply_residual_branch(x, engram_out)

        if isinstance(self.ff, SparseMoE):
            ff_out = self.ff(self.norm2(x), token_ids=token_ids)
        else:
            ff_out = self.ff(self.norm2(x))
        x = self._apply_residual_branch(x, ff_out)
        return x

    def forward_attnres(
        self,
        sources: list,
        *,
        engram_bypass=None,
        token_ids=None,
        attn_mask=None,
        engram_lookup_state=None,
        cache: DeepseekV4LayerCache | None = None,
    ):
        """Append this block's raw sublayer outputs to a Full AttnRes history."""
        if not self.use_attnres:
            raise RuntimeError("forward_attnres requires config.use_attnres=True")
        if self.use_mhc_streams or self.branch is not None:
            raise ValueError("Full AttnRes cannot be combined with mHC/multibranch residuals")

        use_dsa = self.attn.use_dsa and self.attn.local_window is not None and self.attn_kind == "compressed_sparse_attention"
        attn_x = self.attn_res_attn(sources)
        if engram_bypass is not None:
            attn_x = attn_x + engram_bypass
        attn_out = self.attn(
            self.norm1(attn_x),
            attn_mask=attn_mask,
            attn_kind=self.attn_kind,
            use_dsa=use_dsa,
            global_stride=4,
            cache=cache,
        )
        sources.append(attn_out)

        if self.engram is not None:
            if token_ids is None:
                raise ValueError("Engram-enabled block requires token_ids/byte_ids for n-gram lookup")
            engram_x = self.attn_res_engram(sources)
            if engram_bypass is not None:
                engram_x = engram_x + engram_bypass
            engram_out = self.engram(
                self.norm2(engram_x),
                token_ids=token_ids,
                lookup_state=engram_lookup_state,
            )
            if self.attnres_engram_mode == "bypass":
                if engram_bypass is None:
                    raise RuntimeError("AttnRes Engram bypass state was not initialized")
                engram_bypass = engram_bypass + self.engram_bypass_gate * engram_out
            elif self.attnres_engram_mode == "fused":
                # Keep Engram as a strong additive injection within the current
                # transformed delta instead of a separate softmax competitor.
                sources[-1] = sources[-1] + engram_out
            else:
                sources.append(engram_out)

        ff_x = self.attn_res_ffn(sources)
        if engram_bypass is not None:
            ff_x = ff_x + engram_bypass
        if isinstance(self.ff, SparseMoE):
            ff_out = self.ff(self.norm2(ff_x), token_ids=token_ids)
        else:
            ff_out = self.ff(self.norm2(ff_x))
        sources.append(ff_out)
        return sources, attn_x, engram_bypass
