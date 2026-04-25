from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class ByteEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pool_bytes: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pool_bytes = pool_bytes

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(byte_ids)
        if self.pool_bytes and x.dim() == 4:
            x = x.mean(dim=-2)
        return x


class ModalityEmbedding(nn.Module):
    def __init__(self, num_modalities: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, d_model)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(modality_ids)

