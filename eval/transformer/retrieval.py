from __future__ import annotations

import json

import torch

from eval.transformer.common import make_model, run_forward


def make_retrieval_batch(seq_len: int = 256, batch: int = 1):
    byte_ids = torch.randint(0, 240, (batch, seq_len))
    pattern = torch.tensor([11, 22, 33, 44], dtype=torch.long)
    positions = [32, 96, 160]
    for pos in positions:
        byte_ids[:, pos : pos + pattern.numel()] = pattern
    patch_len = (seq_len + 3) // 4
    modality_ids = torch.zeros((batch, patch_len), dtype=torch.long)
    return byte_ids, modality_ids, positions


def patch_stats(hidden_or_logits: torch.Tensor, positions: list[int]):
    stats = []
    for pos in positions:
        patch_pos = min(pos // 4, hidden_or_logits.size(1) - 1)
        vec = hidden_or_logits[0, patch_pos]
        stats.append(
            {
                "patch_pos": patch_pos,
                "l2": round(torch.linalg.norm(vec).item(), 6),
                "mean": round(vec.mean().item(), 6),
                "std": round(vec.std().item(), 6),
            }
        )
    return stats


def main():
    byte_ids, modality_ids, positions = make_retrieval_batch()
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
                "retrieval_probe": patch_stats(logits, positions),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
