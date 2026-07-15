from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.cache_utils import DynamicCache

from ..atoms.config import TransformerConfig


class DeepseekV4ReferenceCache(DynamicCache):
    """DeepSeek-V4 cache with explicit cloning and complete state reset."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__(config=config)
        self.config = config

    def clone(self) -> "DeepseekV4ReferenceCache":
        return copy.deepcopy(self)

    def reset(self) -> None:
        super().reset()
        for layer in self.layers:
            for name in ("buffer_kv", "buffer_gate", "compressed_kv", "overlap_kv", "overlap_gate"):
                state = getattr(layer, name, None)
                if isinstance(state, dict):
                    for key in state:
                        state[key] = None
            entry_count = getattr(layer, "entry_count", None)
            if isinstance(entry_count, dict):
                for key in entry_count:
                    entry_count[key] = 0


def build_hf_deepseek_v4_config(config: TransformerConfig) -> DeepseekV4Config:
    """Translate the local experiment config to the official HF V4 config."""

    attn = config.attention
    moe = config.moe
    num_heads = attn.num_heads
    head_dim = attn.head_dim or config.d_model // num_heads
    o_groups = min(attn.o_groups, num_heads)
    if num_heads % o_groups != 0:
        raise ValueError(f"o_groups={o_groups} must divide num_heads={num_heads}")
    if moe.top_k > moe.num_experts:
        raise ValueError(f"MoE top_k={moe.top_k} exceeds num_experts={moe.num_experts}")

    kwargs = {
        "vocab_size": config.bytes.vocab_size if config.use_byte_first else config.vocab_size,
        "hidden_size": config.d_model,
        "moe_intermediate_size": int(config.d_model * config.mlp_ratio),
        "num_hidden_layers": config.depth,
        "num_attention_heads": num_heads,
        "num_key_value_heads": 1,
        "head_dim": head_dim,
        "q_lora_rank": attn.q_lora_rank or max(head_dim, config.d_model // 4),
        "num_experts_per_tok": moe.top_k,
        "n_routed_experts": moe.num_experts,
        "n_shared_experts": 1 if moe.shared_expert else 0,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": moe.norm_topk_prob,
        "routed_scaling_factor": moe.routed_scaling_factor,
        "max_position_embeddings": config.max_seq_len,
        "rope_theta": attn.rope_base,
        "compress_rates": attn.compress_rates
        or {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        "compress_rope_theta": attn.compress_rope_base,
        "hc_mult": config.hc_mult,
        "hc_sinkhorn_iters": config.mhc_sinkhorn_iters,
        "hc_eps": config.mhc_eps,
        "swiglu_limit": moe.swiglu_limit or 10.0,
        "sliding_window": attn.sliding_window,
        "o_groups": o_groups,
        "o_lora_rank": attn.o_lora_rank,
        "index_n_heads": attn.index_n_heads,
        "index_head_dim": attn.index_head_dim,
        "index_topk": attn.index_topk,
        "num_nextn_predict_layers": config.num_nextn_predict_layers,
        "attention_dropout": attn.dropout,
        "rms_norm_eps": config.rms_norm_eps,
        "use_cache": config.use_cache,
        "partial_rotary_factor": attn.partial_rotary_factor,
    }
    if config.layer_types is not None:
        kwargs["layer_types"] = list(config.layer_types)
    if config.mlp_layer_types is not None:
        kwargs["mlp_layer_types"] = list(config.mlp_layer_types)
    return DeepseekV4Config(**kwargs)


class DeepseekV4ReferenceMolecule(nn.Module):
    """Adapter around the maintained Hugging Face DeepSeek-V4 implementation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.implementation != "hf_deepseek_v4":
            raise ValueError("DeepseekV4ReferenceMolecule requires implementation='hf_deepseek_v4'")
        if config.attention.backend not in {"auto", "eager"}:
            raise ValueError("The official DeepSeek-V4 backend currently supports eager attention only")
        self.config = config
        self.hf_config = build_hf_deepseek_v4_config(config)
        self.output_vocab_size = self.hf_config.vocab_size
        self.hf_model = DeepseekV4ForCausalLM(self.hf_config)
        self._initialize_hash_routes()

    def _initialize_hash_routes(self) -> None:
        """Provide balanced deterministic routes until checkpoint routes are loaded."""

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.hf_model.model.layers):
                table = getattr(layer.mlp.gate, "tid2eid", None)
                if table is None:
                    continue
                token_ids = torch.arange(table.size(0), device=table.device).unsqueeze(1)
                offsets = torch.arange(table.size(1), device=table.device).unsqueeze(0)
                routes = (token_ids + offsets + layer_idx) % self.hf_config.n_routed_experts
                table.copy_(routes)

    def new_cache(self) -> DeepseekV4ReferenceCache:
        return DeepseekV4ReferenceCache(self.hf_config)

    def forward(
        self,
        *,
        token_ids: torch.Tensor | None = None,
        byte_ids: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = None,
        return_hidden: bool = False,
        return_cache: bool = False,
    ):
        if (token_ids is None) == (byte_ids is None):
            raise ValueError("Exactly one of token_ids or byte_ids must be provided")
        input_ids = byte_ids if byte_ids is not None else token_ids
        if modality_ids is not None and torch.any(modality_ids != 0):
            raise ValueError("DeepSeek-V4 v6 is text/byte-token only and does not accept non-zero modality IDs")

        cache_enabled = bool(
            return_cache
            or past_key_values is not None
            or (self.config.use_cache if use_cache is None else use_cache)
        )
        if cache_enabled and past_key_values is None:
            past_key_values = self.new_cache()

        attention_mask = None
        if attn_mask is not None:
            # Native models use True for padding; Hugging Face uses 1 for valid tokens.
            attention_mask = (~attn_mask.to(torch.bool)).to(torch.long)

        outputs = self.hf_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=cache_enabled,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.hf_model.lm_head(hidden)
        cache = outputs.past_key_values

        if return_hidden and return_cache:
            return logits, hidden, cache
        if return_hidden:
            return logits, hidden
        if return_cache:
            return logits, cache
        return logits
