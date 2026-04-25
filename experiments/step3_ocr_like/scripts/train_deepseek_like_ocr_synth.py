from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from experiments.step3_ocr_like.scripts.deepseek_like_ocr_model import DeepSeekOCRLike, OCRLikeConfig


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_eval_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def word_recall(reference: str, prediction: str) -> float:
    ref = {t.lower() for t in TOKEN_RE.findall(reference)}
    pred = {t.lower() for t in TOKEN_RE.findall(prediction)}
    if not ref:
        return 0.0
    return len(ref.intersection(pred)) / len(ref)


def render_text_image(text: str, image_size: int) -> torch.Tensor:
    img = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.multiline_text((24, 24), text, fill="black", font=font, spacing=10)
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(image_size, image_size, 3)
    arr = arr.permute(2, 0, 1).unsqueeze(0) / 255.0
    return arr


def synthetic_document_text(idx: int, seed: int) -> str:
    rng = random.Random((seed * 1_000_003) + idx)
    vendors = ["aphelis", "novatek", "orison", "meridian", "altair", "helix"]
    cities = ["paris", "lyon", "lille", "berlin", "madrid", "milan"]
    items = ["sensor", "adapter", "module", "router", "display", "scanner"]
    statuses = ["paid", "pending", "approved", "shipped", "closed"]
    qty = rng.randint(1, 9)
    unit_price = rng.randint(12, 480)
    subtotal = qty * unit_price
    tax = rng.randint(2, 60)
    total = subtotal + tax
    date = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    due = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    return "\n".join(
        [
            f"invoice {1000 + idx}",
            f"vendor {rng.choice(vendors)}",
            f"city {rng.choice(cities)}",
            f"item {rng.choice(items)}",
            f"quantity {qty}",
            f"unit_price {unit_price}",
            f"tax {tax}",
            f"total {total}",
            f"date {date}",
            f"due {due}",
            f"status {rng.choice(statuses)}",
        ]
    )


def build_corpus(count: int, seed: int) -> list[str]:
    return [synthetic_document_text(i, seed) for i in range(count)]


def encode_bytes(text: str, limit: int) -> list[int]:
    raw = list(text.encode("utf-8", errors="ignore"))[:limit]
    return raw if raw else [32]


def build_batch(
    texts: list[str],
    indices: list[int],
    *,
    prompt_bytes: list[int],
    max_target_bytes: int,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
    prompt_len = len(prompt_bytes)
    seq_len = prompt_len + max_target_bytes - 1
    images = []
    inp_rows = []
    tgt_rows = []
    mask_rows = []
    batch_texts: list[str] = []
    target_lengths: list[int] = []

    for idx in indices:
        text = texts[idx]
        target = encode_bytes(text, max_target_bytes)
        seq = prompt_bytes + target
        inp = seq[:-1]
        tgt = seq[1:]
        inp_pad = inp + ([32] * (seq_len - len(inp)))
        tgt_pad = tgt + ([0] * (seq_len - len(tgt)))
        mask = [False] * seq_len
        target_start = max(prompt_len - 1, 0)
        for pos in range(target_start, min(target_start + len(target), seq_len)):
            mask[pos] = True

        images.append(render_text_image(text, image_size))
        inp_rows.append(torch.tensor(inp_pad, dtype=torch.long))
        tgt_rows.append(torch.tensor(tgt_pad, dtype=torch.long))
        mask_rows.append(torch.tensor(mask, dtype=torch.bool))
        batch_texts.append(text)
        target_lengths.append(len(target))

    image = torch.cat(images, dim=0).to(device)
    inp = torch.stack(inp_rows, dim=0).to(device)
    tgt = torch.stack(tgt_rows, dim=0).to(device)
    mask = torch.stack(mask_rows, dim=0).to(device)
    return image, inp, tgt, mask, batch_texts, target_lengths


def masked_ce_loss(logits: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view_as(tgt)
    return (per_tok * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def generate_bytes(
    model: DeepSeekOCRLike,
    image: torch.Tensor,
    prompt_bytes: list[int],
    *,
    max_new_bytes: int,
) -> list[int]:
    generated: list[int] = []
    current = torch.tensor([prompt_bytes], dtype=torch.long, device=image.device)
    with torch.no_grad():
        for _ in range(max_new_bytes):
            logits = model(image, current, return_hidden=False)
            next_id = int(logits[:, -1, :256].argmax(dim=-1).item())
            generated.append(next_id)
            next_tok = torch.tensor([[next_id]], dtype=torch.long, device=image.device)
            current = torch.cat([current, next_tok], dim=1)
    return generated


def evaluate_model(
    model: DeepSeekOCRLike,
    eval_texts: list[str],
    *,
    prompt_bytes: list[int],
    max_target_bytes: int,
    image_size: int,
    device: torch.device,
    sample_count: int,
) -> dict:
    exact_matches = []
    recalls = []
    examples = []
    count = min(sample_count, len(eval_texts))
    for text in eval_texts[:count]:
        image = render_text_image(text, image_size).to(device)
        target_len = len(encode_bytes(text, max_target_bytes))
        pred_bytes = generate_bytes(model, image, prompt_bytes, max_new_bytes=target_len)
        pred_text = bytes(pred_bytes).decode("utf-8", errors="ignore")
        exact = float(normalize_eval_text(pred_text) == normalize_eval_text(text))
        recall = word_recall(text, pred_text)
        exact_matches.append(exact)
        recalls.append(recall)
        if len(examples) < 3:
            examples.append(
                {
                    "target": text,
                    "prediction": pred_text,
                    "exact_match": exact,
                    "word_recall": recall,
                }
            )
    return {
        "eval_samples": count,
        "exact_match_mean": float(sum(exact_matches) / max(len(exact_matches), 1)),
        "word_recall_mean": float(sum(recalls) / max(len(recalls), 1)),
        "word_recall_std": float(statistics.pstdev(recalls) if len(recalls) > 1 else 0.0),
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "engram_noconv"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--train-samples", type=int, default=1024)
    ap.add_argument("--eval-samples", type=int, default=128)
    ap.add_argument("--max-target-bytes", type=int, default=192)
    ap.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Read the document exactly.\n")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    prompt_bytes = encode_bytes(args.prompt, 512)

    train_texts = build_corpus(args.train_samples, seed=args.seed)
    eval_texts = build_corpus(args.eval_samples, seed=args.seed + 10_000)

    model = DeepSeekOCRLike(OCRLikeConfig(decoder_variant=args.variant)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    batch_rng = random.Random(args.seed + 4242)
    train_losses: list[float] = []

    model.train()
    t0 = time.perf_counter()
    for step in range(args.train_steps):
        batch_indices = [batch_rng.randrange(len(train_texts)) for _ in range(args.batch_size)]
        image, inp, tgt, mask, _texts, _lens = build_batch(
            train_texts,
            batch_indices,
            prompt_bytes=prompt_bytes,
            max_target_bytes=args.max_target_bytes,
            image_size=args.image_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(image, inp, return_hidden=False)
        text_logits = logits[:, -inp.size(1) :, :]
        loss = masked_ce_loss(text_logits, tgt, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(float(loss.item()))

    train_time = time.perf_counter() - t0

    model.eval()
    eval_metrics = evaluate_model(
        model,
        eval_texts,
        prompt_bytes=prompt_bytes,
        max_target_bytes=args.max_target_bytes,
        image_size=args.image_size,
        device=device,
        sample_count=min(32, args.eval_samples),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    metrics = {
        "variant": args.variant,
        "seed": args.seed,
        "device": str(device),
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": args.image_size,
        "train_samples": args.train_samples,
        "eval_samples_total": args.eval_samples,
        "max_target_bytes": args.max_target_bytes,
        "prompt_len_bytes": len(prompt_bytes),
        "train_loss_last": float(train_losses[-1]),
        "train_loss_mean_last_20": float(sum(train_losses[-20:]) / min(len(train_losses), 20)),
        "train_loss_min": float(min(train_losses)),
        "train_time_sec": float(train_time),
        "steps_per_sec": float(args.train_steps / max(train_time, 1e-9)),
        **eval_metrics,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
