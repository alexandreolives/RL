from __future__ import annotations

import json

from models.example import build_config, build_deepseek_v4_v4_config, build_deepseek_v4_v5_config, build_deepseek_v4_v6_config, build_model
from models.molecules import TransformerMolecule

from eval.transformer.common import make_batch, run_forward


def build_ablation_models():
    variants = {}

    cfg = build_config(use_engram=False, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu")
    variants["baseline"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu")
    variants["engram_only"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu")
    cfg.engram.conv_enabled = False
    variants["engram_noconv"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=False, use_mhc=False, use_moe=False, activation="gelu")
    cfg.engram.long_conv_enabled = True
    variants["engram_fullconv"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=False, use_mhc=True, use_moe=False, activation="swiglu")
    variants["engram_mhc"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=False, activation="swiglu")
    variants["engram_mhc_dsa"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu")
    variants["full"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu")
    cfg.engram.conv_enabled = False
    variants["full_noconv"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=False, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu")
    cfg.use_multibranch_residual = False
    cfg.residual_branches = 4
    cfg.attention.qk_norm = True
    cfg.attention.num_kv_heads = 1
    cfg.attention.local_window = 128
    cfg.attention.dsa_top_k = 64
    cfg.moe.num_experts = 8
    cfg.moe.top_k = 4
    variants["v1"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu")
    cfg.use_multibranch_residual = False
    cfg.residual_branches = 4
    cfg.attention.qk_norm = True
    cfg.attention.num_kv_heads = 1
    cfg.attention.local_window = 128
    cfg.attention.dsa_top_k = 64
    cfg.moe.num_experts = 8
    cfg.moe.top_k = 4
    cfg.engram.use_layerwise_hash = True
    cfg.engram.conv_enabled = False
    cfg.engram.long_conv_enabled = False
    cfg.engram.insert_layers = (0, 2, 4)
    variants["v2"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=True, use_mhc=True, use_moe=True, activation="swiglu")
    cfg.use_multibranch_residual = False
    cfg.residual_branches = 4
    cfg.attention.qk_norm = True
    cfg.attention.num_kv_heads = 1
    cfg.attention.local_window = 128
    cfg.attention.dsa_top_k = 64
    cfg.moe.num_experts = 8
    cfg.moe.top_k = 6
    cfg.moe.scoring_func = "sqrtsoftplus"
    cfg.moe.norm_topk_prob = True
    cfg.moe.routed_scaling_factor = 2.5
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
    cfg.engram.use_layerwise_hash = True
    cfg.engram.conv_enabled = False
    cfg.engram.long_conv_enabled = False
    cfg.engram.insert_layers = (0, 1, 2)
    variants["v3"] = TransformerMolecule(cfg).eval()

    cfg = build_deepseek_v4_v4_config()
    variants["v4"] = TransformerMolecule(cfg).eval()

    cfg = build_deepseek_v4_v5_config()
    variants["v5"] = TransformerMolecule(cfg).eval()

    cfg = build_deepseek_v4_v6_config()
    variants["v6"] = build_model(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=False, use_mhc=True, use_moe=False, activation="swiglu")
    cfg.engram.conv_kernel_size = 1
    variants["wout_conv"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=False, use_mhc=True, use_moe=False, activation="swiglu")
    cfg.engram.insert_layers = (1,)
    variants["single_insert"] = TransformerMolecule(cfg).eval()

    cfg = build_config(use_engram=True, use_dsa=False, use_mhc=True, use_moe=False, activation="swiglu")
    cfg.engram.ngram_orders = (2, 3, 4)
    variants["ngram_234"] = TransformerMolecule(cfg).eval()

    return variants


def main():
    byte_ids, modality_ids = make_batch(batch=1, seq_len=128, patch_size=4)
    out = []
    for name, model in build_ablation_models().items():
        _, perf = run_forward(model, byte_ids=byte_ids, modality_ids=modality_ids, steps=3)
        out.append(
            {
                "name": name,
                "params": sum(p.numel() for p in model.parameters()),
                "mean_sec": round(perf["mean_sec"], 6),
                "tok_per_s": round(perf["tok_per_s"], 2),
            }
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
