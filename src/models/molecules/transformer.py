from __future__ import annotations

import torch
from torch import nn

from ..atoms.config import TransformerConfig
from ..atoms.embeddings import TokenEmbedding, ByteEmbedding, ModalityEmbedding
from ..atoms.bytes import BytePatcher
from ..atoms.engram import build_engram_lookup_state
from ..atoms.layers import TransformerBlock
from ..atoms.norms import RMSNorm


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
        self.blocks = nn.ModuleList()
        for idx in range(config.depth):
            is_global = True
            if config.attention.local_window is not None:
                # Gemma-like hybrid attention: sparse local blocks with periodic global ones
                is_global = (idx % 4 == 0) or (idx == config.depth - 1)
            self.blocks.append(TransformerBlock(config, layer_idx=idx, is_global_layer=is_global))
        self.has_engram = any(block.engram is not None for block in self.blocks)
        self.output_vocab_size = config.bytes.vocab_size if config.use_byte_first else config.vocab_size

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
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
        return self.dropout(x), memory_ids

    def forward(
        self,
        *,
        token_ids: torch.Tensor | None = None,
        byte_ids: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        for block in self.blocks:
            x = block(x, token_ids=memory_ids, attn_mask=attn_mask, engram_lookup_state=engram_lookup_state)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits
