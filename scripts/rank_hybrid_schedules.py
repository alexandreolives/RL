"""Rank schedules by a reproducible Pareto rule, never by random selection.

Input is JSON emitted by ``benchmark_hybrid_schedules.py``. A schedule is kept
when no other row has both lower latency and fewer parameters. Optional quality
fields (for example ``accuracy``) can be added by an external training run;
the ranker then maximizes quality before applying the cost Pareto filter.
"""
from __future__ import annotations

import argparse, json, sys


def dominates(a, b):
    quality_a, quality_b = a.get("quality", 0.0), b.get("quality", 0.0)
    return (quality_a >= quality_b and a["latency_ms"] <= b["latency_ms"] and a["parameters"] <= b["parameters"]
            and (quality_a > quality_b or a["latency_ms"] < b["latency_ms"] or a["parameters"] < b["parameters"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="-", help="benchmark JSON path, or - for stdin")
    args = parser.parse_args()
    payload = json.load(sys.stdin if args.input == "-" else open(args.input))
    rows = payload["results"]
    frontier = [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]
    print(json.dumps({"method": "quality-first Pareto frontier", "frontier": frontier}, indent=2))


if __name__ == "__main__":
    main()
