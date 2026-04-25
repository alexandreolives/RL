#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_KEYS = ("arc_challenge", "hellaswag", "mmlu", "bbh", "drop", "gsm8k", "math")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def find_metric(block: dict) -> float:
    if "acc_norm,none" in block:
        return float(block["acc_norm,none"])
    if "acc,none" in block:
        return float(block["acc,none"])
    if "acc" in block:
        return float(block["acc"])
    raise KeyError("No accuracy metric found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--engram", type=Path, required=True)
    args = parser.parse_args()

    b = load(args.baseline)["results"]
    e = load(args.engram)["results"]

    print("task,baseline,engram,delta")
    for task in TASK_KEYS:
        if task not in b or task not in e:
            continue
        bv = find_metric(b[task])
        ev = find_metric(e[task])
        print(f"{task},{bv:.6f},{ev:.6f},{(ev - bv):.6f}")


if __name__ == "__main__":
    main()
