from __future__ import annotations

import argparse
import json
import time
from statistics import mean

import torch

from models.example import build_variant


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_component(model, byte_ids: torch.Tensor, steps: int) -> dict[str, float]:
    block = next(block for block in model.blocks if block.engram is not None)
    x, memory_ids = model.embed(byte_ids=byte_ids, modality_ids=torch.zeros((byte_ids.size(0), (byte_ids.size(1) + 3) // 4), dtype=torch.long, device=byte_ids.device))
    lookup_state = model.has_engram and memory_ids is not None
    if lookup_state:
        from models.atoms.engram import build_engram_lookup_state

        lookup_state = build_engram_lookup_state(
            memory_ids,
            ngram_orders=model.config.engram.ngram_orders,
            heads=model.config.engram.heads,
            slots=model.config.engram.slots,
        )

    engram = block.engram
    assert engram is not None

    lookup_times = []
    mix_times = []
    conv_times = []

    with torch.no_grad():
        for _ in range(steps):
            sync(byte_ids.device)
            t0 = time.perf_counter()
            mem = engram._lookup_memory(memory_ids, lookup_state=lookup_state)
            sync(byte_ids.device)
            lookup_times.append(time.perf_counter() - t0)

            normed_x = engram.query_norm(x)
            q = engram.query_proj(normed_x).view(x.size(0), x.size(1), engram.heads, engram.memory_dim)
            controls = engram.control_proj(normed_x).view(x.size(0), x.size(1), engram.heads, engram.num_orders + 1)
            order_bias = controls[..., : engram.num_orders]
            head_gate = torch.sigmoid(controls[..., engram.num_orders :])
            sync(byte_ids.device)
            t0 = time.perf_counter()
            order_scores = (q.unsqueeze(-2) * mem).sum(dim=-1) / (engram.memory_dim**0.5)
            order_scores = order_scores + order_bias
            order_weights = torch.softmax(order_scores, dim=-1)
            mixed = (order_weights.unsqueeze(-1) * mem).sum(dim=-2)
            mixed = mixed * head_gate
            mixed = mixed.reshape(x.size(0), x.size(1), engram.heads * engram.memory_dim)
            gated = engram.out_proj(engram.out_norm(mixed))
            sync(byte_ids.device)
            mix_times.append(time.perf_counter() - t0)

            sync(byte_ids.device)
            t0 = time.perf_counter()
            _ = torch.nn.functional.silu(engram._causal_depthwise(gated) + gated)
            sync(byte_ids.device)
            conv_times.append(time.perf_counter() - t0)

    total = mean(lookup_times) + mean(mix_times) + mean(conv_times)
    return {
        "lookup_sec": mean(lookup_times),
        "mix_sec": mean(mix_times),
        "conv_sec": mean(conv_times),
        "total_sec": total,
        "lookup_pct": mean(lookup_times) / total,
        "mix_pct": mean(mix_times) / total,
        "conv_pct": mean(conv_times) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = build_variant("engram").to(device).eval()
    byte_ids = torch.randint(0, 260, (args.batch, args.seq_len), device=device)
    results = bench_component(model, byte_ids, args.steps)
    print(json.dumps({k: round(v, 6) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
