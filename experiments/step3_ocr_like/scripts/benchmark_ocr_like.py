from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont
import pytesseract


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def norm_tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def ocr_image(image: Image.Image, lang: str) -> str:
    return pytesseract.image_to_string(image, lang=lang, config="--oem 1 --psm 3")


def render_text_image(text: str, *, width: int = 1024, height: int = 256) -> Image.Image:
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.multiline_text((24, 24), text, fill="black", font=font, spacing=10)
    return img


def synthetic_samples(limit: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    vocab = [
        "invoice", "total", "date", "order", "customer", "item", "price", "quantity",
        "address", "amount", "tax", "reference", "number", "status", "payment", "due",
    ]
    rows = []
    for i in range(limit):
        parts = [
            f"doc {i}",
            f"{rng.choice(vocab)} {rng.randint(10, 9999)}",
            f"{rng.choice(vocab)} {rng.randint(10, 9999)}",
            f"{rng.choice(vocab)} {rng.randint(10, 9999)}",
        ]
        rows.append("\n".join(parts))
    return rows


def eval_funsd(limit: int, lang: str) -> dict:
    from datasets import load_dataset

    ds = load_dataset("nielsr/funsd", split="test")
    n = min(limit, len(ds))
    recalls = []
    latencies = []
    for i in range(n):
        row = ds[i]
        image = row["image"]
        words = row.get("words", [])
        gt = norm_tokens(" ".join(w for w in words if isinstance(w, str)))
        t0 = time.perf_counter()
        pred_text = ocr_image(image, lang=lang)
        dt = time.perf_counter() - t0
        pred = norm_tokens(pred_text)
        latencies.append(dt)
        if not gt:
            recalls.append(0.0)
            continue
        recalls.append(len(gt.intersection(pred)) / max(len(gt), 1))
    return {
        "dataset": "nielsr/funsd:test",
        "kind": "existing_dataset",
        "n": n,
        "word_recall_mean": float(sum(recalls) / max(len(recalls), 1)),
        "word_recall_std": float(statistics.pstdev(recalls) if len(recalls) > 1 else 0.0),
        "latency_sec_mean": float(sum(latencies) / max(len(latencies), 1)),
        "latency_sec_std": float(statistics.pstdev(latencies) if len(latencies) > 1 else 0.0),
    }


def eval_synthetic_text(limit: int, lang: str, seed: int) -> dict:
    samples = synthetic_samples(limit, seed)
    recalls = []
    latencies = []
    for text in samples:
        image = render_text_image(text)
        gt = norm_tokens(text)
        t0 = time.perf_counter()
        pred_text = ocr_image(image, lang=lang)
        dt = time.perf_counter() - t0
        pred = norm_tokens(pred_text)
        latencies.append(dt)
        recalls.append(len(gt.intersection(pred)) / max(len(gt), 1))
    return {
        "dataset": "synthetic_text_image",
        "kind": "synthetic_dataset",
        "n": len(samples),
        "word_recall_mean": float(sum(recalls) / max(len(recalls), 1)),
        "word_recall_std": float(statistics.pstdev(recalls) if len(recalls) > 1 else 0.0),
        "latency_sec_mean": float(sum(latencies) / max(len(latencies), 1)),
        "latency_sec_std": float(statistics.pstdev(latencies) if len(latencies) > 1 else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["funsd", "synthetic_text"], default="synthetic_text")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--lang", type=str, default="eng")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.dataset == "funsd":
        payload = eval_funsd(limit=args.limit, lang=args.lang)
    elif args.dataset == "synthetic_text":
        payload = eval_synthetic_text(limit=args.limit, lang=args.lang, seed=args.seed)
    else:
        raise ValueError(args.dataset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
