from models.atoms.config import AttentionConfig, MoEConfig, MultimodalConfig, TransformerConfig, EngramConfig, ByteConfig
from models.molecules import DeepseekV4ReferenceMolecule, TransformerMolecule


def build_deepseek_v4_attention_schedule(depth: int) -> tuple[str, ...]:
    """
    Match the public DeepSeek-V4 attention schedule described by Hugging Face:
    layers 0-1 are HCA, then layers 2+ alternate CSA / HCA.
    """
    if depth <= 0:
        return ()
    if depth == 1:
        return ("heavily_compressed_attention",)
    schedule = ["heavily_compressed_attention", "heavily_compressed_attention"]
    for layer_idx in range(2, depth):
        schedule.append("compressed_sparse_attention" if layer_idx % 2 == 0 else "heavily_compressed_attention")
    return tuple(schedule)


def build_deepseek_v4_mlp_schedule(depth: int, *, num_hash_layers: int = 3) -> tuple[str, ...]:
    hash_layers = min(depth, num_hash_layers)
    return tuple("hash_moe" if layer_idx < hash_layers else "moe" for layer_idx in range(depth))


def build_deepseek_v4_v1_config(
    *,
    attention_backend: str = "auto",
    input_mode: str = "symbolic",
) -> TransformerConfig:
    cfg = build_config(
        use_engram=False,
        use_dsa=True,
        use_mhc=True,
        use_moe=True,
        activation="swiglu",
        attention_backend=attention_backend,
    )
    cfg.multimodal.enabled = False
    cfg.use_rmsnorm = True
    cfg.use_byte_first = input_mode == "byte"
    cfg.attention.qk_norm = True
    cfg.attention.num_heads = 8
    cfg.attention.num_kv_heads = 1
    cfg.attention.local_window = 128
    cfg.attention.dsa_top_k = 64
    cfg.attention.dsa_indexer_hidden = 128
    cfg.residual_branches = 4
    cfg.moe.num_experts = 8
    cfg.moe.top_k = 4
    cfg.moe.shared_expert = True
    cfg.moe.router_jitter = 0.0
    if input_mode == "byte":
        cfg.bytes.use_byte_patching = False
        cfg.bytes.patch_size = 1
    elif input_mode == "symbolic":
        cfg.vocab_size = 260
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return cfg


def build_deepseek_v4_v2_config(
    *,
    attention_backend: str = "auto",
    input_mode: str = "symbolic",
) -> TransformerConfig:
    cfg = build_deepseek_v4_v1_config(attention_backend=attention_backend, input_mode=input_mode)
    cfg.engram.enabled = True
    cfg.engram.use_layerwise_hash = True
    cfg.engram.conv_enabled = False
    cfg.engram.long_conv_enabled = False
    cfg.engram.insert_layers = (0, 2, 4)
    cfg.engram.heads = 8
    cfg.engram.top_k = 8
    cfg.engram.ngram_orders = (2, 3)
    return cfg


def build_deepseek_v4_v3_config(
    *,
    attention_backend: str = "auto",
    input_mode: str = "symbolic",
) -> TransformerConfig:
    cfg = build_deepseek_v4_v2_config(attention_backend=attention_backend, input_mode=input_mode)
    cfg.layer_types = (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "compressed_sparse_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "heavily_compressed_attention",
    )
    cfg.engram.insert_layers = (0, 1, 2)
    cfg.engram.top_k = 6
    cfg.moe.top_k = 6
    cfg.moe.scoring_func = "sqrtsoftplus"
    cfg.moe.norm_topk_prob = True
    cfg.moe.routed_scaling_factor = 2.5
    cfg.attention.dsa_top_k = 64
    cfg.attention.num_heads = 8
    cfg.attention.num_kv_heads = 1
    cfg.attention.qk_norm = True
    cfg.residual_branches = 4
    return cfg


def build_deepseek_v4_v4_config(
    *,
    attention_backend: str = "auto",
    input_mode: str = "symbolic",
) -> TransformerConfig:
    cfg = build_config(
        use_engram=False,
        use_dsa=True,
        use_mhc=True,
        use_moe=True,
        activation="swiglu",
        attention_backend=attention_backend,
    )
    cfg.multimodal.enabled = False
    cfg.use_rmsnorm = True
    cfg.use_byte_first = input_mode == "byte"
    cfg.attention.qk_norm = True
    cfg.attention.num_heads = 8
    cfg.attention.num_kv_heads = 1
    cfg.attention.local_window = 128
    cfg.attention.sliding_window = 128
    cfg.attention.partial_rotary_factor = 0.125
    cfg.attention.compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    cfg.attention.compress_rope_base = 160_000
    cfg.attention.rope_scaling = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65_536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    cfg.attention.index_n_heads = cfg.attention.num_heads
    cfg.attention.index_head_dim = 128
    cfg.attention.index_topk = 512
    cfg.attention.dsa_top_k = 64
    cfg.attention.dsa_indexer_hidden = 128
    cfg.residual_branches = 4
    cfg.moe.num_experts = 8
    cfg.moe.top_k = 6
    cfg.moe.shared_expert = True
    cfg.moe.scoring_func = "sqrtsoftplus"
    cfg.moe.topk_method = "noaux_tc"
    cfg.moe.norm_topk_prob = True
    cfg.moe.routed_scaling_factor = 2.5
    cfg.moe.router_jitter = 0.0
    cfg.moe.swiglu_limit = 10.0
    cfg.layer_types = build_deepseek_v4_attention_schedule(cfg.depth)
    cfg.mlp_layer_types = build_deepseek_v4_mlp_schedule(cfg.depth, num_hash_layers=3)
    cfg.num_hash_layers = 3
    if input_mode == "byte":
        cfg.bytes.use_byte_patching = False
        cfg.bytes.patch_size = 1
    elif input_mode == "symbolic":
        cfg.vocab_size = 260
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return cfg


def build_deepseek_v4_v5_config(
    *,
    attention_backend: str = "auto",
    input_mode: str = "symbolic",
) -> TransformerConfig:
    cfg = build_config(
        use_engram=False,
        use_dsa=False,
        use_mhc=True,
        use_moe=True,
        activation="swiglu",
        attention_backend=attention_backend,
    )
    cfg.multimodal.enabled = False
    cfg.use_rmsnorm = True
    cfg.use_byte_first = input_mode == "byte"
    cfg.attention.qk_norm = True
    cfg.attention.num_heads = 8
    cfg.attention.num_kv_heads = 1
    cfg.attention.q_lora_rank = 128
    cfg.attention.q_lora_norm = True
    cfg.attention.kv_norm = True
    cfg.attention.tie_kv = True
    cfg.attention.local_window = 128
    cfg.attention.sliding_window = 128
    cfg.attention.partial_rotary_factor = 0.125
    cfg.attention.partial_rope_on_tail = True
    cfg.attention.rotate_output_rope = True
    cfg.attention.compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    cfg.attention.compress_rope_base = 160_000
    cfg.attention.rope_scaling = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65_536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    cfg.attention.index_n_heads = cfg.attention.num_heads
    cfg.attention.index_head_dim = 128
    cfg.attention.index_topk = 64
    cfg.attention.use_attention_sink = True
    cfg.attention.csa_overlap = True
    cfg.attention.csa_window_factor = 2
    cfg.attention.learned_compression = True
    cfg.attention.grouped_o_proj = True
    cfg.attention.o_groups = 8
    cfg.attention.o_lora_rank = 128
    cfg.attention.kv_cache_storage_dtype = "float8_e4m3fn"
    cfg.attention.index_cache_storage_dtype = "float16"
    cfg.residual_branches = 4
    cfg.hc_mult = 4
    cfg.mhc_sinkhorn_iters = 8
    cfg.mhc_eps = 1e-6
    cfg.moe.num_experts = 8
    cfg.moe.top_k = 6
    cfg.moe.shared_expert = True
    cfg.moe.scoring_func = "sqrtsoftplus"
    cfg.moe.topk_method = "noaux_tc"
    cfg.moe.norm_topk_prob = True
    cfg.moe.routed_scaling_factor = 2.5
    cfg.moe.router_jitter = 0.0
    cfg.use_mhc_streams = True
    cfg.layer_types = build_deepseek_v4_attention_schedule(cfg.depth)
    cfg.mlp_layer_types = build_deepseek_v4_mlp_schedule(cfg.depth, num_hash_layers=3)
    cfg.num_hash_layers = 3
    cfg.num_nextn_predict_layers = 1
    if input_mode == "byte":
        cfg.bytes.use_byte_patching = False
        cfg.bytes.patch_size = 1
    elif input_mode == "symbolic":
        cfg.vocab_size = 260
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return cfg


def build_deepseek_v4_v6_config(
    *,
    attention_backend: str = "auto",
    input_mode: str = "symbolic",
) -> TransformerConfig:
    """Build the corrected V4 variant backed by Hugging Face Transformers."""

    cfg = build_config(
        use_engram=False,
        use_dsa=False,
        use_mhc=True,
        use_moe=True,
        activation="swiglu",
        attention_backend=attention_backend,
    )
    cfg.implementation = "hf_deepseek_v4"
    cfg.multimodal.enabled = False
    cfg.use_byte_first = input_mode == "byte"
    cfg.use_mhc_streams = True
    cfg.hc_mult = 4
    cfg.mhc_sinkhorn_iters = 20
    cfg.mhc_eps = 1e-6
    cfg.use_cache = False
    cfg.num_nextn_predict_layers = 1

    cfg.attention.num_heads = 8
    cfg.attention.num_kv_heads = 1
    cfg.attention.head_dim = cfg.d_model // cfg.attention.num_heads
    cfg.attention.q_lora_rank = 128
    cfg.attention.sliding_window = 128
    cfg.attention.partial_rotary_factor = 0.125
    cfg.attention.compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    cfg.attention.compress_rope_base = 160_000
    cfg.attention.o_groups = 8
    cfg.attention.o_lora_rank = 128
    cfg.attention.index_n_heads = 8
    cfg.attention.index_head_dim = 128
    cfg.attention.index_topk = 64

    cfg.moe.num_experts = 8
    cfg.moe.top_k = 6
    cfg.moe.shared_expert = True
    cfg.moe.scoring_func = "sqrtsoftplus"
    cfg.moe.topk_method = "noaux_tc"
    cfg.moe.norm_topk_prob = True
    cfg.moe.routed_scaling_factor = 2.5
    cfg.moe.swiglu_limit = 10.0

    # Let the maintained HF config generate the checkpoint-compatible layer
    # and hash-MoE schedules for the selected depth.
    cfg.layer_types = None
    cfg.mlp_layer_types = None
    cfg.num_hash_layers = 3
    if input_mode == "byte":
        cfg.bytes.use_byte_patching = False
        cfg.bytes.patch_size = 1
    elif input_mode == "symbolic":
        cfg.vocab_size = 260
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return cfg


def apply_model_size(cfg: TransformerConfig, model_size: str, *, input_mode: str) -> TransformerConfig:
    if model_size == "tiny":
        cfg.d_model = 192
        cfg.depth = 4
        cfg.mlp_ratio = 2.0
        cfg.attention.num_heads = 4
        cfg.attention.num_kv_heads = 1
        cfg.attention.local_window = 128
        cfg.attention.index_n_heads = cfg.attention.num_heads
        cfg.attention.index_head_dim = cfg.d_model // cfg.attention.num_heads
        if cfg.attention.head_dim is not None:
            cfg.attention.head_dim = cfg.d_model // cfg.attention.num_heads
        if cfg.attention.q_lora_rank is not None:
            cfg.attention.q_lora_rank = min(cfg.attention.q_lora_rank, 64)
        if cfg.attention.csa_overlap:
            cfg.attention.index_topk = min(cfg.attention.index_topk, 32)
        if cfg.attention.grouped_o_proj:
            cfg.attention.o_groups = min(cfg.attention.o_groups, cfg.attention.num_heads)
            cfg.attention.o_lora_rank = min(cfg.attention.o_lora_rank, 32)
        cfg.attention.dsa_indexer_hidden = 128
        cfg.residual_branches = 4
        if cfg.layer_types is not None:
            cfg.layer_types = cfg.layer_types[:cfg.depth]
        if cfg.mlp_layer_types is not None:
            cfg.mlp_layer_types = cfg.mlp_layer_types[:cfg.depth]
        if cfg.engram.enabled:
            cfg.engram.slots = 1024
            cfg.engram.heads = 4
            cfg.engram.top_k = 4
            cfg.engram.insert_layers = tuple(idx for idx in cfg.engram.insert_layers if idx < cfg.depth)
    elif model_size == "small":
        cfg.d_model = 256
        cfg.depth = 6
        cfg.mlp_ratio = 2.5
        cfg.attention.num_heads = 4
        cfg.attention.num_kv_heads = 1
        cfg.attention.local_window = 192
        cfg.attention.index_n_heads = cfg.attention.num_heads
        cfg.attention.index_head_dim = cfg.d_model // cfg.attention.num_heads
        if cfg.attention.head_dim is not None:
            cfg.attention.head_dim = cfg.d_model // cfg.attention.num_heads
        if cfg.attention.q_lora_rank is not None:
            cfg.attention.q_lora_rank = min(cfg.attention.q_lora_rank, 96)
        if cfg.attention.csa_overlap:
            cfg.attention.index_topk = min(cfg.attention.index_topk, 64)
        if cfg.attention.grouped_o_proj:
            cfg.attention.o_groups = min(cfg.attention.o_groups, cfg.attention.num_heads)
            cfg.attention.o_lora_rank = min(cfg.attention.o_lora_rank, 64)
        cfg.attention.dsa_indexer_hidden = 128
        cfg.residual_branches = 4
        if cfg.layer_types is not None:
            cfg.layer_types = cfg.layer_types[:cfg.depth]
        if cfg.mlp_layer_types is not None:
            cfg.mlp_layer_types = cfg.mlp_layer_types[:cfg.depth]
        if cfg.engram.enabled:
            cfg.engram.slots = 2048
            cfg.engram.heads = 4
            cfg.engram.top_k = 4
            cfg.engram.insert_layers = tuple(idx for idx in cfg.engram.insert_layers if idx < cfg.depth)
    elif model_size != "base":
        raise ValueError(f"Unknown model_size: {model_size}")

    if input_mode == "byte":
        cfg.use_byte_first = True
        cfg.bytes.use_byte_patching = False
        cfg.bytes.patch_size = 1
    elif input_mode == "symbolic":
        cfg.use_byte_first = False
        cfg.vocab_size = 260
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return cfg


def build_config(
    *,
    use_engram: bool = True,
    use_dsa: bool = True,
    use_mhc: bool = True,
    use_moe: bool = True,
    use_attnres: bool = False,
    use_multibranch_residual: bool = False,
    activation: str = "swiglu",
    attention_backend: str = "auto",
) -> TransformerConfig:
    return TransformerConfig(
        d_model=512,
        depth=8,
        vocab_size=32000,
        activation=activation,
        use_rmsnorm=True,
        use_mhc=use_mhc,
        use_attnres=use_attnres,
        use_multibranch_residual=use_multibranch_residual,
        residual_branches=4,
        use_byte_first=True,
        attention=AttentionConfig(
            num_heads=8,
            num_kv_heads=2,
            local_window=256,
            qk_norm=True,
            tie_kv=False,
            use_dsa=use_dsa,
            global_stride=4,
            dsa_top_k=32,
            dsa_indexer_hidden=128,
            backend=attention_backend,
        ),
        engram=EngramConfig(
            enabled=use_engram,
            slots=4099,
            heads=8,
            top_k=8,
            ngram_orders=(2, 3),
            long_conv_threshold=256,
            long_conv_enabled=False,
            conv_bottleneck_ratio=0.5,
            insert_layers=(1, 4),
        ),
        bytes=ByteConfig(
            enabled=True,
            vocab_size=260,
            patch_size=4,
            use_byte_patching=True,
            patch_pooling="mean",
        ),
        moe=MoEConfig(
            enabled=use_moe,
            num_experts=4,
            top_k=2,
            shared_expert=True,
        ),
        multimodal=MultimodalConfig(
            enabled=True,
            num_modalities=4,
        ),
    )


def build_model(config: TransformerConfig) -> TransformerMolecule | DeepseekV4ReferenceMolecule:
    if config.implementation == "hf_deepseek_v4":
        return DeepseekV4ReferenceMolecule(config)
    return TransformerMolecule(config)


def build_demo_model() -> TransformerMolecule | DeepseekV4ReferenceMolecule:
    cfg = build_config()
    return build_model(cfg)


def build_variant(
    name: str,
    *,
    attention_backend: str = "auto",
) -> TransformerMolecule | DeepseekV4ReferenceMolecule:
    key = name.lower()
    if key == "baseline":
        cfg = build_config(use_engram=False, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key == "attnres":
        cfg = build_config(use_engram=False, use_dsa=False, use_mhc=False, use_moe=False, use_attnres=True, activation="gelu", attention_backend=attention_backend)
    elif key in {"engram", "engram_adaptive"}:
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key == "engram_noconv":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
        cfg.engram.conv_enabled = False
    elif key == "engram_attnres":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, use_attnres=True, activation="gelu", attention_backend=attention_backend)
    elif key == "engram_noconv_attnres":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, use_attnres=True, activation="gelu", attention_backend=attention_backend)
        cfg.engram.conv_enabled = False
    elif key == "engram_layerhash":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
        cfg.engram.use_layerwise_hash = True
    elif key == "engram_compress":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
        cfg.engram.compressed_vocab_size = 128
        cfg.engram.compression_reserved_ids = 20
    elif key == "engram_official_gate":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
        cfg.engram.official_gating = True
    elif key == "engram_lightconv":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
        cfg.engram.conv_bottleneck_ratio = 0.25
    elif key == "engram_fullconv":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
        cfg.engram.long_conv_enabled = True
    elif key in {"deepseek_v4_like", "v4_like", "full_v4_like", "v1", "deepseek_v4_v1"}:
        cfg = build_deepseek_v4_v1_config(attention_backend=attention_backend, input_mode="symbolic")
    elif key in {"v2", "deepseek_v4_v2"}:
        cfg = build_deepseek_v4_v2_config(attention_backend=attention_backend, input_mode="symbolic")
    elif key in {"v3", "deepseek_v4_v3", "deepseek_v4_public_like"}:
        cfg = build_deepseek_v4_v3_config(attention_backend=attention_backend, input_mode="symbolic")
    elif key in {"v4", "deepseek_v4_v4"}:
        cfg = build_deepseek_v4_v4_config(attention_backend=attention_backend, input_mode="symbolic")
    elif key in {"v5", "deepseek_v4_v5"}:
        cfg = build_deepseek_v4_v5_config(attention_backend=attention_backend, input_mode="symbolic")
    elif key in {
        "v6",
        "deepseek_v4_v6",
        "deepseek_v4_reference",
        "deepseek_v4_public_exact",
        "deepseek_v4_public_solid",
    }:
        cfg = build_deepseek_v4_v6_config(attention_backend=attention_backend, input_mode="symbolic")
    elif key == "dsa":
        cfg = build_config(use_engram=False, use_dsa=True, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key == "mhc":
        cfg = build_config(use_engram=False, use_dsa=False, use_mhc=True, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key == "full":
        cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu", attention_backend=attention_backend)
    elif key == "full_noconv":
        cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu", attention_backend=attention_backend)
        cfg.engram.conv_enabled = False
    else:
        raise ValueError(f"Unknown variant: {name}")
    return build_model(cfg)
