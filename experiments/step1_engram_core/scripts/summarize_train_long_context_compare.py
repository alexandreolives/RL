#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_json_array(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end < start:
        raise ValueError("No JSON array found in input file")
    return json.loads(text[start : end + 1])


def metric(row: dict, task: str, key: str = "accuracy") -> float:
    return float(row["eval"][task][key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to raw JSON output file")
    args = parser.parse_args()

    rows = extract_json_array(args.input.read_text())
    print("variant,world_size,train_acc,passkey,multi_query,variable_tracking")
    for row in rows:
        train_acc = row.get("train_acc_mean_last10_mean", row.get("train_acc_mean_last10"))
        world_size = row.get("world_size", 1)
        print(
            f"{row['variant']},{world_size},{float(train_acc):.6f},"
            f"{metric(row, 'passkey'):.6f},{metric(row, 'multi_query'):.6f},{metric(row, 'variable_tracking'):.6f}"
        )


if __name__ == "__main__":
    main()
