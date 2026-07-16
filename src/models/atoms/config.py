from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    num_heads: int = 8
    num_kv_heads: int | None = None
    head_dim: int | None = None
    q_lora_rank: int | None = None
    q_lora_norm: bool = False
    kv_norm: bool = False
    dropout: float = 0.0
    rope_base: int = 10_000
    partial_rotary_factor: float = 1.0
    partial_rope_on_tail: bool = False
    rotate_output_rope: bool = False
    local_window: int | None = None
    sliding_window: int = 128
    qk_norm: bool = False
    tie_kv: bool = False
    use_dsa: bool = False
    global_stride: int = 4
    dsa_top_k: int = 256
    dsa_indexer_hidden: int = 128
    compress_rates: dict[str, int] | None = None
    compress_rope_base: int = 160_000
    rope_scaling: dict[str, float | int | str] | None = None
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    use_attention_sink: bool = False
    csa_overlap: bool = False
    csa_window_factor: int = 1
    learned_compression: bool = False
    grouped_o_proj: bool = False
    o_groups: int = 1
    o_lora_rank: int = 64
    kv_cache_storage_dtype: str | None = None
    index_cache_storage_dtype: str | None = None
    backend: str = "auto"


@dataclass
class EngramConfig:
    enabled: bool = False
    slots: int = 4099
    heads: int = 8
    top_k: int = 8
    memory_dim: int | None = None
    ngram_orders: tuple[int, ...] = (2, 3)
    use_layerwise_hash: bool = False
    compressed_vocab_size: int | None = None
    compression_reserved_ids: int = 16
    official_gating: bool = False
    conv_enabled: bool = True
    long_conv_threshold: int | None = 256
    long_conv_enabled: bool = False
    conv_kernel_size: int = 4
    conv_dilation: int = 3
    conv_bottleneck_ratio: float = 0.5
    conv_zero_init: bool = True
    insert_layers: tuple[int, ...] = (1, 4)


@dataclass
class ByteConfig:
    enabled: bool = False
    vocab_size: int = 260
    patch_size: int = 4
    use_byte_patching: bool = True
    patch_pooling: str = "mean"


@dataclass
class MoEConfig:
    enabled: bool = False
    num_experts: int = 4
    top_k: int = 2
    shared_expert: bool = True
    scoring_func: str = "softmax"
    topk_method: str = "greedy"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    router_jitter: float = 0.0
    swiglu_limit: float | None = None


@dataclass
class MultimodalConfig:
    enabled: bool = False
    vocab_size_text: int = 32_000
    vocab_size_byte: int = 260
    num_modalities: int = 4
    early_fusion: bool = True


@dataclass
class TransformerConfig:
    d_model: int = 512
    depth: int = 8
    vocab_size: int = 32_000
    max_seq_len: int = 2048
    use_cache: bool = False
    cache_max_length: int | None = None
    num_nextn_predict_layers: int = 0
    mlp_ratio: float = 4.0
    activation: str = "swiglu"
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    use_rmsnorm: bool = True
    use_attnres: bool = False
    attnres_engram_mode: str = "source"
    use_multibranch_residual: bool = False
    residual_branches: int = 2
    use_mhc: bool = False
    use_mhc_streams: bool = False
    hc_mult: int = 4
    mhc_sinkhorn_iters: int = 0
    mhc_eps: float = 1e-6
    byte_latent_pooling: bool = False
    use_byte_first: bool = False
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)
    bytes: ByteConfig = field(default_factory=ByteConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    layer_types: tuple[str, ...] | None = None
    mlp_layer_types: tuple[str, ...] | None = None
    num_hash_layers: int | None = None
    implementation: str = "native"
