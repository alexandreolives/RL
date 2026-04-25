from models.atoms.config import AttentionConfig, MoEConfig, MultimodalConfig, TransformerConfig, EngramConfig, ByteConfig
from models.molecules import TransformerMolecule


def build_config(
    *,
    use_engram: bool = True,
    use_dsa: bool = True,
    use_mhc: bool = True,
    use_moe: bool = True,
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


def build_demo_model() -> TransformerMolecule:
    cfg = build_config()
    return TransformerMolecule(cfg)


def build_variant(name: str, *, attention_backend: str = "auto") -> TransformerMolecule:
    key = name.lower()
    if key == "baseline":
        cfg = build_config(use_engram=False, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key in {"engram", "engram_adaptive"}:
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key == "engram_noconv":
        cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
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
    elif key == "dsa":
        cfg = build_config(use_engram=False, use_dsa=True, use_mhc=False, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key == "mhc":
        cfg = build_config(use_engram=False, use_dsa=False, use_mhc=True, use_moe=False, activation="gelu", attention_backend=attention_backend)
    elif key == "full":
        cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu", attention_backend=attention_backend)
    else:
        raise ValueError(f"Unknown variant: {name}")
    return TransformerMolecule(cfg)
