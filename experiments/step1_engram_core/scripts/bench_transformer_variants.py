from __future__ import annotations

import argparse
import json
import time
from statistics import mean

import torch

from models.example import build_variant


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def bench_variant(
    name: str,
    seq_len: int,
    batch: int,
    warmup: int,
    steps: int,
    device: torch.device,
    attention_backend: str,
) -> dict:
    model = build_variant(name, attention_backend=attention_backend).to(device).eval()
    params = count_params(model)
    byte_ids = torch.randint(0, 260, (batch, seq_len), device=device)
    patch_len = (seq_len + model.config.bytes.patch_size - 1) // model.config.bytes.patch_size
    modality_ids = torch.zeros((batch, patch_len), dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(byte_ids=byte_ids, modality_ids=modality_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    with torch.no_grad():
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(byte_ids=byte_ids, modality_ids=modality_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

    toks = batch * seq_len
    return {
        "variant": name,
        "params": params,
        "seq_len": seq_len,
        "batch": batch,
        "device": str(device),
        "attention_backend": attention_backend,
        "mean_sec": round(mean(times), 6),
        "tok_per_s": round(toks / mean(times), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "eager", "sdpa", "flash"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "engram", "engram_noconv", "dsa", "mhc", "full", "full_noconv", "v1", "v2", "v3", "v4", "v5"],
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    results = [
        bench_variant(
            name=v,
            seq_len=args.seq_len,
            batch=args.batch,
            warmup=args.warmup,
            steps=args.steps,
            device=device,
            attention_backend=args.attention_backend,
        )
        for v in args.variants
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
