from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.atoms.engram import build_engram_lookup_state
from models.example import build_variant
from models.molecules import TransformerMolecule


@dataclass
class OCRLikeConfig:
    d_model: int | None = None
    patch_size: int = 16
    max_visual_tokens: int = 96
    attention_backend: str = "auto"
    decoder_variant: str = "engram_noconv"
    byte_patching: bool = False
    visual_modality_id: int = 1
    text_modality_id: int = 0
    visual_dummy_byte_id: int = 1


class VisualCompressor(nn.Module):
    """
    DeepSeek-OCR-like visual context compression:
    image -> patch embeddings -> compressed visual tokens.
    """

    def __init__(self, d_model: int, patch_size: int, max_visual_tokens: int) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.max_visual_tokens = max_visual_tokens

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, 3, H, W]
        x = self.patch(image)  # [B, D, H', W']
        b, d, h, w = x.shape
        x = x.view(b, d, h * w).transpose(1, 2)  # [B, N, D]
        if x.size(1) > self.max_visual_tokens:
            # Uniform downsample in token space (cheap deterministic compression).
            idx = torch.linspace(0, x.size(1) - 1, self.max_visual_tokens, device=x.device).long()
            x = x.index_select(1, idx)
        return self.norm(x)


class DeepSeekOCRLike(nn.Module):
    """
    Copy-adapt design:
    - visual encoder/compressor path
    - prefix fusion with text-byte stream
    - decoder path based on our Engram stack (`engram_noconv`)
    """

    def __init__(self, cfg: OCRLikeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or OCRLikeConfig()
        self.decoder: TransformerMolecule = build_variant(self.cfg.decoder_variant, attention_backend=self.cfg.attention_backend)
        if self.cfg.d_model is None:
            self.cfg.d_model = self.decoder.config.d_model
        elif self.cfg.d_model != self.decoder.config.d_model:
            raise ValueError(
                f"OCRLikeConfig.d_model={self.cfg.d_model} does not match decoder d_model={self.decoder.config.d_model}"
            )
        # Keep decoder lightweight and byte-centric.
        self.decoder.config.use_byte_first = True
        self.decoder.config.bytes.use_byte_patching = bool(self.cfg.byte_patching)
        self.decoder.byte_patcher = (
            self.decoder.byte_patcher if self.decoder.config.bytes.use_byte_patching else None
        )
        self.decoder.config.multimodal.enabled = True
        self.decoder.config.multimodal.num_modalities = max(self.decoder.config.multimodal.num_modalities, 4)
        self.visual = VisualCompressor(
            d_model=self.cfg.d_model,
            patch_size=self.cfg.patch_size,
            max_visual_tokens=self.cfg.max_visual_tokens,
        )

    def _byte_embed(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        byte_x = self.decoder.byte_embedding(byte_ids)
        memory_ids = byte_ids
        if self.decoder.byte_patcher is not None:
            byte_x = self.decoder.byte_patcher(byte_x)
            patch = self.decoder.config.bytes.patch_size
            pad = (-byte_ids.size(1)) % patch
            if pad:
                byte_ids = torch.nn.functional.pad(byte_ids, (0, pad))
            memory_ids = byte_ids.view(byte_ids.size(0), -1, patch)[:, :, 0]
        return byte_x, memory_ids

    def forward(self, image: torch.Tensor, byte_ids: torch.Tensor, *, return_hidden: bool = False):
        """
        image: [B, 3, H, W]
        byte_ids: [B, T] in byte vocab space
        """
        vis = self.visual(image)  # [B, V, D]
        txt, memory_ids = self._byte_embed(byte_ids)  # [B, T', D], [B, T']

        x = torch.cat([vis, txt], dim=1)

        # Modality tags: visual prefix + text suffix.
        if self.decoder.modality_embedding is not None:
            b = x.size(0)
            v = vis.size(1)
            t = txt.size(1)
            vis_mod = torch.full((b, v), self.cfg.visual_modality_id, device=x.device, dtype=torch.long)
            txt_mod = torch.full((b, t), self.cfg.text_modality_id, device=x.device, dtype=torch.long)
            mod = torch.cat([vis_mod, txt_mod], dim=1)
            x = x + self.decoder.modality_embedding(mod)

        x = self.decoder.dropout(x)

        # Engram lookup over visual+text IDs (visual gets dummy byte id).
        engram_lookup_state = None
        full_ids = None
        if self.decoder.has_engram and memory_ids is not None:
            b = memory_ids.size(0)
            v = vis.size(1)
            vis_ids = torch.full(
                (b, v),
                self.cfg.visual_dummy_byte_id,
                device=memory_ids.device,
                dtype=memory_ids.dtype,
            )
            full_ids = torch.cat([vis_ids, memory_ids], dim=1)
            engram_lookup_state = build_engram_lookup_state(
                full_ids,
                ngram_orders=self.decoder.config.engram.ngram_orders,
                heads=self.decoder.config.engram.heads,
                slots=self.decoder.config.engram.slots,
                layer_ids=tuple(i for i, block in enumerate(self.decoder.blocks) if block.engram is not None),
                use_layerwise_hash=self.decoder.config.engram.use_layerwise_hash,
                compressed_vocab_size=self.decoder.config.engram.compressed_vocab_size,
                compression_reserved_ids=self.decoder.config.engram.compression_reserved_ids,
            )

        for block in self.decoder.blocks:
            x = block(x, token_ids=full_ids, attn_mask=None, engram_lookup_state=engram_lookup_state)
        hidden = self.decoder.final_norm(x)
        logits = self.decoder.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits
