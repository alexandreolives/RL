from __future__ import annotations

import json

from models.example import build_config
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
