from __future__ import annotations

import json

import torch

from eval.transformer.common import make_model, run_forward


def make_needle_batch(seq_len: int = 512, batch: int = 1):
    byte_ids = torch.randint(0, 260, (batch, seq_len))
    needle = torch.tensor([250, 251, 252, 253], dtype=torch.long)
    pos = seq_len // 2
    byte_ids[:, pos : pos + needle.numel()] = needle
    patch_len = (seq_len + 3) // 4
    modality_ids = torch.zeros((batch, patch_len), dtype=torch.long)
    return byte_ids, modality_ids, pos


def score_needle(logits: torch.Tensor, pos: int):
    patch_pos = pos // 4
    patch_pos = min(patch_pos, logits.size(1) - 1)
    probs = torch.softmax(logits[0, patch_pos], dim=-1)
    top = torch.topk(probs, k=5)
    return {
        "patch_pos": patch_pos,
        "top_ids": top.indices.tolist(),
        "top_probs": [round(x, 6) for x in top.values.tolist()],
    }


def main():
    byte_ids, modality_ids, pos = make_needle_batch()
    results = []
    for name in ["baseline", "engram", "dsa", "mhc", "full"]:
        model = make_model(name)
        logits, perf = run_forward(model, byte_ids=byte_ids, modality_ids=modality_ids, steps=2)
        results.append(
            {
                "name": name,
                "perf": {
                    "mean_sec": round(perf["mean_sec"], 6),
                    "tok_per_s": round(perf["tok_per_s"], 2),
                },
                "needle_probe": score_needle(logits, pos),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
