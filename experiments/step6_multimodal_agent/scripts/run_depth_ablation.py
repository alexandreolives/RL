"""Reproducible depth/loop ablation for the M1 parity task."""
from __future__ import annotations

import argparse
import json
import statistics

import torch

from compare_stateful_memory import LoopMemoryActor, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scores = [train("loop", args.episodes, seed, "parity", args.sequence_length,
                    loop_depth=args.depth, loop_iterations=args.iterations,
                    loop_ff_dim=args.ff_dim, device=args.device)
              for seed in range(args.seeds)]
    model = LoopMemoryActor(depth=args.depth, max_iterations=args.iterations, ff_dim=args.ff_dim)
    result = {
        "depth": args.depth,
        "iterations": args.iterations,
        "effective_depth": args.depth * args.iterations,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "scores": scores,
        "mean": sum(scores) / len(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "ff_dim": args.ff_dim,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
