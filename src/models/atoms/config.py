from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    num_heads: int = 8
    num_kv_heads: int | None = None
    dropout: float = 0.0
    rope_base: int = 10_000
    local_window: int | None = None
    qk_norm: bool = False
    tie_kv: bool = False
    use_dsa: bool = False
    global_stride: int = 4
    dsa_top_k: int = 256
    dsa_indexer_hidden: int = 128
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
    router_jitter: float = 0.0


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
    mlp_ratio: float = 4.0
    activation: str = "swiglu"
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    use_rmsnorm: bool = True
    use_multibranch_residual: bool = False
    residual_branches: int = 2
    use_mhc: bool = False
    byte_latent_pooling: bool = False
    use_byte_first: bool = False
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)
    bytes: ByteConfig = field(default_factory=ByteConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
