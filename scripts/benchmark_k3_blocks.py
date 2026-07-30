"""Small reproducible benchmark for the local K3-inspired blocks."""
from __future__ import annotations

import argparse
import json
import time
import torch

from models.atoms.kda import KDA, HybridKDA
from models.atoms.latent_moe import LatentMoE
from models.atoms.hybrid import ConfigurableHybridCore


def count(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def bench(model, x, warmup, steps, iterations=1):
    model.eval()
    def run():
        return model(x, iterations=iterations) if isinstance(model, ConfigurableHybridCore) else model(x)
    for _ in range(warmup): run()
    if x.is_cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps): run()
    if x.is_cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--hybrid-iterations", type=int, default=1)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    x = torch.randn(args.batch, args.sequence, args.d_model, device=device)
    variants = {
        "kda": KDA(args.d_model, heads=4),
        "kda_qat": KDA(args.d_model, heads=4, qat=True),
        "hybrid_3_to_1": HybridKDA(args.d_model, kda_blocks=3, attention_heads=4),
        "latent_moe": LatentMoE(args.d_model, args.d_model // 2, num_experts=4, top_k=2, shared_experts=2),
        "fourier_kda_attention": ConfigurableHybridCore(args.d_model, stages=["fourier", "kda", "attention"]),
        "fourier_kda_loop_attention": ConfigurableHybridCore(args.d_model, stages=["fourier", "kda", "loop", "attention"],),
    }
    rows = []
    for name, model in variants.items():
        model.to(device)
        rows.append({"variant": name, "parameters": count(model), "latency_ms": bench(model, x, 3, args.steps, args.hybrid_iterations)})
    print(json.dumps({"device": str(device), "batch": args.batch, "sequence": args.sequence, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
