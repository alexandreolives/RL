from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.step3_ocr_like.scripts.deepseek_like_ocr_model import DeepSeekOCRLike, OCRLikeConfig


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(image_size, image_size, 3)
    arr = arr.permute(2, 0, 1).unsqueeze(0) / 255.0
    return arr


def render_text_image(text: str, image_size: int) -> torch.Tensor:
    img = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.multiline_text((24, 24), text, fill="black", font=font, spacing=10)
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(image_size, image_size, 3)
    arr = arr.permute(2, 0, 1).unsqueeze(0) / 255.0
    return arr


def text_to_bytes(prompt: str, max_len: int) -> torch.Tensor:
    b = list(prompt.encode("utf-8", errors="ignore"))[:max_len]
    if not b:
        b = [32]
    return torch.tensor([b], dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument(
        "--text-image",
        type=str,
        default="invoice 1042\namount 98\ndate 2026-04-22\nstatus paid",
        help="If --image is omitted, render this text into a synthetic document image.",
    )
    ap.add_argument("--prompt", type=str, default="<image>\\n<|grounding|>Convert the document to markdown.")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--max-bytes", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--decoder-variant", choices=["baseline", "engram_noconv"], default="engram_noconv")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = DeepSeekOCRLike(OCRLikeConfig(decoder_variant=args.decoder_variant)).to(device).eval()
    if args.image is not None:
        image = image_to_tensor(args.image, args.image_size).to(device)
        image_desc = str(args.image)
    else:
        image = render_text_image(args.text_image, args.image_size).to(device)
        image_desc = "<synthetic_text_image>"
    byte_ids = text_to_bytes(args.prompt, args.max_bytes).to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(image, byte_ids, return_hidden=False)
        times = []
        logits = None
        hidden = None
        for _ in range(args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, hidden = model(image, byte_ids, return_hidden=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    payload = {
        "image": image_desc,
        "synthetic_text_image": args.image is None,
        "decoder_variant": args.decoder_variant,
        "prompt_len_bytes": int(byte_ids.size(1)),
        "logits_shape": list(logits.shape),
        "hidden_shape": list(hidden.shape),
        "device": str(device),
        "forward_sec_mean": float(sum(times) / max(len(times), 1)),
        "forward_sec_min": float(min(times)),
        "forward_sec_max": float(max(times)),
        "status": "ok",
    }
    print(json.dumps(payload, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
