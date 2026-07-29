from __future__ import annotations

import argparse
import json
from collections import Counter

import torch

from agents.loop_transformer import AdaptiveLoopCore


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--batches", type=int, default=100); args = parser.parse_args()
    torch.manual_seed(0); core = AdaptiveLoopCore(d_model=64, max_iterations=4)
    counts = Counter()
    for _ in range(args.batches):
        _, used = core(torch.randn(8, 4, 64))
        counts[int(used)] += 1
    print(json.dumps({"batches": args.batches, "iteration_histogram": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__": main()
