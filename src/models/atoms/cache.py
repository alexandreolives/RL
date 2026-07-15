from __future__ import annotations

import torch


LAYER_TYPE_CACHE_MAPPING: dict[str, type["DeepseekV4LayerCache"]] = {}


class DeepseekV4LayerCache:
    layer_type: str | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        layer_type = getattr(cls, "layer_type", None)
        if layer_type is not None:
            LAYER_TYPE_CACHE_MAPPING[layer_type] = cls

    def __init__(self, *, max_seq_len: int | None = None, detach: bool = True) -> None:
        self.max_seq_len = max_seq_len
        self.detach = detach
        self.hidden_states: torch.Tensor | None = None
        self.key_states: torch.Tensor | None = None
        self.value_states: torch.Tensor | None = None
        self.token_ids: torch.Tensor | None = None
        self.attn_mask: torch.Tensor | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seq_len={self.get_seq_length()})"

    def _maybe_detach(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return x.detach() if self.detach else x

    def _trim_left(self) -> None:
        if self.max_seq_len is None or self.hidden_states is None:
            return
        seq_len = self.hidden_states.size(1)
        if seq_len <= self.max_seq_len:
            return
        start = seq_len - self.max_seq_len
        self.hidden_states = self.hidden_states[:, start:].contiguous()
        if self.key_states is not None and self.key_states.size(-2) > self.max_seq_len:
            self.key_states = self.key_states[..., start:, :].contiguous()
        if self.value_states is not None and self.value_states.size(-2) > self.max_seq_len:
            self.value_states = self.value_states[..., start:, :].contiguous()
        if self.token_ids is not None:
            self.token_ids = self.token_ids[:, start:].contiguous()
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask[:, start:].contiguous()

    def set_sequence(
        self,
        hidden_states: torch.Tensor,
        *,
        token_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> None:
        self.hidden_states = self._maybe_detach(hidden_states)
        self.token_ids = self._maybe_detach(token_ids)
        self.attn_mask = self._maybe_detach(attn_mask)
        self._trim_left()

    def update(
        self,
        hidden_states: torch.Tensor,
        *,
        token_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.hidden_states is None:
            self.set_sequence(hidden_states, token_ids=token_ids, attn_mask=attn_mask)
        else:
            self.hidden_states = torch.cat([self.hidden_states, self._maybe_detach(hidden_states)], dim=1)
            if token_ids is not None:
                token_ids = self._maybe_detach(token_ids)
                self.token_ids = token_ids if self.token_ids is None else torch.cat([self.token_ids, token_ids], dim=1)
            if attn_mask is not None:
                attn_mask = self._maybe_detach(attn_mask)
                self.attn_mask = attn_mask if self.attn_mask is None else torch.cat([self.attn_mask, attn_mask], dim=1)
            self._trim_left()
        return self.hidden_states, self.attn_mask

    def set_kv(self, key_states: torch.Tensor | None, value_states: torch.Tensor | None) -> None:
        self.key_states = self._maybe_detach(key_states)
        self.value_states = self._maybe_detach(value_states)

    def append_kv(self, key_states: torch.Tensor | None, value_states: torch.Tensor | None) -> None:
        if key_states is None or value_states is None:
            return
        key_states = self._maybe_detach(key_states)
        value_states = self._maybe_detach(value_states)
        if self.key_states is None or self.value_states is None:
            self.key_states = key_states
            self.value_states = value_states
        else:
            self.key_states = torch.cat([self.key_states, key_states], dim=-2)
            self.value_states = torch.cat([self.value_states, value_states], dim=-2)
        self._trim_left()

    def get_seq_length(self) -> int:
        if self.hidden_states is None:
            return 0
        return int(self.hidden_states.size(1))

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        cache_len = self.get_seq_length()
        return cache_len, cache_len + query_length

    def get_max_cache_shape(self) -> int:
        return 0 if self.max_seq_len is None else self.max_seq_len

    def clone(self) -> "DeepseekV4LayerCache":
        clone = self.__class__(max_seq_len=self.max_seq_len, detach=self.detach)
        clone.hidden_states = None if self.hidden_states is None else self.hidden_states.clone()
        clone.key_states = None if self.key_states is None else self.key_states.clone()
        clone.value_states = None if self.value_states is None else self.value_states.clone()
        clone.token_ids = None if self.token_ids is None else self.token_ids.clone()
        clone.attn_mask = None if self.attn_mask is None else self.attn_mask.clone()
        return clone

    def reset(self) -> None:
        self.hidden_states = None
        self.key_states = None
        self.value_states = None
        self.token_ids = None
        self.attn_mask = None


class DeepseekV4SlidingCache(DeepseekV4LayerCache):
    layer_type = "sliding_attention"


class DeepseekV4HCACache(DeepseekV4LayerCache):
    layer_type = "heavily_compressed_attention"

    def __init__(self, *, max_seq_len: int | None = None, detach: bool = True) -> None:
        super().__init__(max_seq_len=max_seq_len, detach=detach)
        self.compressed_pool: torch.Tensor | None = None
        self.compressed_positions: torch.Tensor | None = None
        self.compressed_count: int = 0


class DeepseekV4CSACache(DeepseekV4LayerCache):
    layer_type = "compressed_sparse_attention"

    def __init__(self, *, max_seq_len: int | None = None, detach: bool = True) -> None:
        super().__init__(max_seq_len=max_seq_len, detach=detach)
        self.compressed_pool: torch.Tensor | None = None
        self.compressed_positions: torch.Tensor | None = None
        self.index_pool: torch.Tensor | None = None
        self.index_positions: torch.Tensor | None = None
        self.overlap_pool: torch.Tensor | None = None
        self.overlap_positions: torch.Tensor | None = None
        self.compressed_count: int = 0
        self.index_count: int = 0


class DynamicCache:
    def __init__(
        self,
        *,
        layer_types: tuple[str, ...] | None = None,
        depth: int,
        max_seq_len: int | None = None,
        detach: bool = True,
    ) -> None:
        self.layers = build_layer_caches(layer_types, depth=depth, max_seq_len=max_seq_len, detach=detach)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> DeepseekV4LayerCache:
        return self.layers[idx]

    def __iter__(self):
        return iter(self.layers)

    def clone(self) -> "DynamicCache":
        clone = object.__new__(DynamicCache)
        clone.layers = [layer.clone() if layer is not None else None for layer in self.layers]
        return clone

    def reset(self) -> None:
        for layer in self.layers:
            if layer is not None:
                layer.reset()

    def get_seq_length(self) -> int:
        for layer in self.layers:
            if layer is not None:
                return layer.get_seq_length()
        return 0


def make_layer_cache(
    layer_type: str | None,
    *,
    max_seq_len: int | None = None,
    detach: bool = True,
) -> DeepseekV4LayerCache:
    if layer_type == "compressed_sparse_attention":
        return DeepseekV4CSACache(max_seq_len=max_seq_len, detach=detach)
    if layer_type == "heavily_compressed_attention":
        return DeepseekV4HCACache(max_seq_len=max_seq_len, detach=detach)
    return DeepseekV4SlidingCache(max_seq_len=max_seq_len, detach=detach)


def build_layer_caches(
    layer_types: tuple[str, ...] | None,
    *,
    depth: int,
    max_seq_len: int | None = None,
    detach: bool = True,
) -> list[DeepseekV4LayerCache]:
    caches: list[DeepseekV4LayerCache] = []
    for idx in range(depth):
        layer_type = None
        if layer_types is not None and idx < len(layer_types):
            layer_type = layer_types[idx]
        caches.append(make_layer_cache(layer_type, max_seq_len=max_seq_len, detach=detach))
    return caches
