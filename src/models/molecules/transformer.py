from __future__ import annotations

import torch
from torch import nn

from ..atoms.config import TransformerConfig
from ..atoms.embeddings import TokenEmbedding, ByteEmbedding, ModalityEmbedding
from ..atoms.cache import DeepseekV4LayerCache, DynamicCache
from ..atoms.bytes import BytePatcher
from ..atoms.engram import build_engram_lookup_state
from ..atoms.layers import TransformerBlock
from ..atoms.norms import RMSNorm
from ..atoms.residual import DeepseekV4HyperHead, FullAttentionResidual


class TransformerMolecule(nn.Module):
    """
    Configurable transformer assembly with hooks for:
    - DeepSeek-like choices: RMSNorm, SwiGLU, sparse MoE, multibranch residual
    - Gemma-like choices: grouped KV heads, tie_kv, hybrid local/global attention
    - BLT/Facebook-like bytes: byte embeddings and optional byte pooling
    - Early-fusion multimodal pretraining: modality embeddings on a single sequence
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.text_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.byte_embedding = ByteEmbedding(
            config.bytes.vocab_size,
            config.d_model,
            pool_bytes=False,
        )
        self.byte_patcher = (
            BytePatcher(config.bytes.patch_size, pooling=config.bytes.patch_pooling)
            if config.use_byte_first and config.bytes.use_byte_patching
            else None
        )
        self.modality_embedding = (
            ModalityEmbedding(config.multimodal.num_modalities, config.d_model)
            if config.multimodal.enabled
            else None
        )

        self.dropout = nn.Dropout(config.dropout)
        self.use_attnres = config.use_attnres
        self.attnres_engram_mode = getattr(config, "attnres_engram_mode", "source")
        if self.use_attnres and (config.use_mhc or config.use_mhc_streams or config.use_multibranch_residual):
            raise ValueError("Full AttnRes cannot be combined with mHC/multibranch residuals")
        self.use_mhc_streams = config.use_mhc_streams
        self.hc_mult = config.hc_mult if config.use_mhc_streams else 1
        self.blocks = nn.ModuleList()
        total_depth = config.depth + max(0, config.num_nextn_predict_layers)
        for idx in range(total_depth):
            layer_type = None
            if config.layer_types is not None and idx < len(config.layer_types):
                layer_type = config.layer_types[idx]
            elif idx >= config.depth:
                layer_type = "sliding_attention"
            mlp_type = None
            if config.mlp_layer_types is not None and idx < len(config.mlp_layer_types):
                mlp_type = config.mlp_layer_types[idx]
            elif idx >= config.depth:
                mlp_type = "moe"
            is_global = True
            if config.attention.local_window is not None:
                # Gemma-like hybrid attention: sparse local blocks with periodic global ones
                is_global = (idx % 4 == 0) or (idx == config.depth - 1)
            if layer_type in {"sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"}:
                is_global = False
            self.blocks.append(
                TransformerBlock(
                    config,
                    layer_idx=idx,
                    is_global_layer=is_global,
                    attn_kind=layer_type,
                    mlp_kind=mlp_type,
                )
            )
        self.has_engram = any(block.engram is not None for block in self.blocks)
        self.output_vocab_size = config.bytes.vocab_size if config.use_byte_first else config.vocab_size

        self.hc_head = (
            DeepseekV4HyperHead(config.hc_mult, config.d_model, eps=config.mhc_eps)
            if config.use_mhc_streams
            else None
        )

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
        self.final_attn_res = (
            FullAttentionResidual(config.d_model, eps=config.rms_norm_eps)
            if self.use_attnres
            else None
        )
        self.lm_head = nn.Linear(config.d_model, self.output_vocab_size, bias=False)

    def embed(
        self,
        token_ids: torch.Tensor | None = None,
        byte_ids: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pieces = []
        memory_ids = None
        if token_ids is not None and not self.config.use_byte_first:
            pieces.append(self.text_embedding(token_ids))
            memory_ids = token_ids
        if byte_ids is not None:
            byte_x = self.byte_embedding(byte_ids)
            if self.byte_patcher is not None:
                byte_x = self.byte_patcher(byte_x)
                # Match patch-wise hidden states with patch-wise ids via the first byte in each patch.
                patch = self.config.bytes.patch_size
                pad = (-byte_ids.size(1)) % patch
                if pad:
                    byte_ids = torch.nn.functional.pad(byte_ids, (0, pad))
                memory_ids = byte_ids.view(byte_ids.size(0), -1, patch)[:, :, 0]
            else:
                memory_ids = byte_ids
            pieces.append(byte_x)

        if not pieces:
            raise ValueError("At least one of token_ids or byte_ids must be provided")

        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        if modality_ids is not None and self.modality_embedding is not None:
            x = x + self.modality_embedding(modality_ids)
        if self.use_mhc_streams:
            x = x.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        return self.dropout(x), memory_ids

    def forward(
        self,
        *,
        token_ids: torch.Tensor | None = None,
        byte_ids: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        past_key_values: list[DeepseekV4LayerCache | None] | None = None,
        use_cache: bool | None = None,
        return_hidden: bool = False,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, DynamicCache] | tuple[torch.Tensor, torch.Tensor, DynamicCache]:
        x, memory_ids = self.embed(token_ids=token_ids, byte_ids=byte_ids, modality_ids=modality_ids)
        engram_lookup_state = None
        if self.has_engram and memory_ids is not None:
            engram_lookup_state = build_engram_lookup_state(
                memory_ids,
                ngram_orders=self.config.engram.ngram_orders,
                heads=self.config.engram.heads,
                slots=self.config.engram.slots,
                layer_ids=tuple(idx for idx, block in enumerate(self.blocks) if block.engram is not None),
                use_layerwise_hash=self.config.engram.use_layerwise_hash,
                compressed_vocab_size=self.config.engram.compressed_vocab_size,
                compression_reserved_ids=self.config.engram.compression_reserved_ids,
            )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if return_cache:
            use_cache = True
        cache_obj: DynamicCache | None = None
        cache_layers: list[DeepseekV4LayerCache | None] | None = None
        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                cache_obj = past_key_values
                cache_layers = list(cache_obj.layers)
            else:
                cache_layers = list(past_key_values)
            use_cache = True
        elif use_cache:
            cache_obj = DynamicCache(
                layer_types=self.config.layer_types,
                depth=len(self.blocks),
                max_seq_len=self.config.cache_max_length or self.config.max_seq_len,
                detach=not self.training,
            )
            cache_layers = cache_obj.layers

        current_x = x
        attnres_sources = [x] if self.use_attnres else None
        engram_bypass = torch.zeros_like(x) if self.use_attnres and self.attnres_engram_mode in {"bypass", "bounded_bypass"} else None
        current_ids = memory_ids
        current_mask = attn_mask
        for idx, block in enumerate(self.blocks):
            layer_cache = cache_layers[idx] if cache_layers is not None and idx < len(cache_layers) else None
            if attnres_sources is not None:
                attnres_sources, block_input, engram_bypass = block.forward_attnres(
                    attnres_sources,
                    engram_bypass=engram_bypass,
                    token_ids=current_ids,
                    attn_mask=current_mask,
                    engram_lookup_state=engram_lookup_state,
                    cache=layer_cache,
                )
            else:
                block_input = current_x
                current_x = block(
                    block_input,
                    token_ids=current_ids,
                    attn_mask=current_mask,
                    engram_lookup_state=engram_lookup_state,
                    cache=layer_cache,
                )
            if cache_layers is not None and idx < len(cache_layers):
                cache_layers[idx].update(block_input, token_ids=current_ids, attn_mask=current_mask)
        if attnres_sources is not None:
            current_x = self.final_attn_res(attnres_sources)
            if engram_bypass is not None:
                current_x = current_x + engram_bypass
        if self.hc_head is not None:
            current_x = self.hc_head(current_x)
        hidden = self.final_norm(current_x)
        logits = self.lm_head(hidden)
        def wrap_cache() -> DynamicCache:
            wrapped = cache_obj if cache_obj is not None else DynamicCache(
                layer_types=self.config.layer_types,
                depth=len(self.blocks),
                max_seq_len=self.config.cache_max_length or self.config.max_seq_len,
                detach=not self.training,
            )
            if cache_layers is not None:
                wrapped.layers = cache_layers
            return wrapped
        if return_hidden and return_cache:
            return logits, hidden, wrap_cache()
        if return_hidden:
            return logits, hidden
        if return_cache:
            return logits, wrap_cache()
        return logits
