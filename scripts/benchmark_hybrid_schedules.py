"""Compare configurable hybrid schedules and loop counts."""
from __future__ import annotations

import argparse, json, time
import torch
from models.atoms.hybrid import ConfigurableHybridCore


SCHEDULES = {
    "attention_only": ["attention"],
    "fourier_attention": ["fourier", "attention"],
    "fourier_kda_attention": ["fourier", "kda", "attention"],
    "fourier_kda_loop_attention": ["fourier", "kda", "loop", "attention"],
    "kda_attention_3to1": ["kda", "kda", "kda", "attention"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--iterations", type=int, nargs="+", default=[1, 2])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--carry-kda-state", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    x = torch.randn(args.batch, args.sequence, args.d_model, device=device)
    rows = []
    for name, stages in SCHEDULES.items():
        for loops in args.iterations:
            model = ConfigurableHybridCore(args.d_model, stages=stages, carry_kda_state=args.carry_kda_state).to(device).eval()
            with torch.no_grad():
                for _ in range(2): model(x, iterations=loops)
                if device.type == "cuda": torch.cuda.synchronize(device)
                start = time.perf_counter()
                for _ in range(args.steps): model(x, iterations=loops)
                if device.type == "cuda": torch.cuda.synchronize(device)
            rows.append({"schedule": name, "stages": stages, "iterations": loops,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "carry_kda_state": args.carry_kda_state,
                         "latency_ms": (time.perf_counter() - start) * 1000 / args.steps})
    print(json.dumps({"device": str(device), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
