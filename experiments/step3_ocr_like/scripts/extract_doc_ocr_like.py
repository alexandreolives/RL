from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import fitz
from PIL import Image


def ocr_page_with_tesseract(page: fitz.Page, lang: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "page.png"
        out_base = Path(td) / "ocr"
        pix = page.get_pixmap(dpi=200, alpha=False)
        pix.save(str(img_path))
        cmd = [
            "tesseract",
            str(img_path),
            str(out_base),
            "-l",
            lang,
            "--oem",
            "1",
            "--psm",
            "3",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        txt_path = out_base.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8", errors="ignore")
    return ""


def ocr_image_with_tesseract(path: Path, lang: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        out_base = Path(td) / "ocr"
        cmd = [
            "tesseract",
            str(path),
            str(out_base),
            "-l",
            lang,
            "--oem",
            "1",
            "--psm",
            "3",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        txt_path = out_base.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8", errors="ignore")
    return ""


def extract_pdf(path: Path, max_pages: int, lang: str, ocr_fallback: bool) -> dict:
    doc = fitz.open(path)
    pages = []
    total_pages = min(len(doc), max_pages if max_pages > 0 else len(doc))
    for i in range(total_pages):
        page = doc[i]
        blocks = page.get_text("blocks")
        text_blocks = []
        for b in blocks:
            x0, y0, x1, y1, txt, *_ = b
            txt = (txt or "").strip()
            if not txt:
                continue
            text_blocks.append(
                {
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": txt,
                }
            )
        page_text = "\n".join(b["text"] for b in text_blocks).strip()
        used_ocr = False
        if ocr_fallback and not page_text:
            page_text = ocr_page_with_tesseract(page, lang=lang).strip()
            used_ocr = bool(page_text)
        pages.append(
            {
                "page_index": i,
                "size": {"width": float(page.rect.width), "height": float(page.rect.height)},
                "text": page_text,
                "blocks": text_blocks,
                "ocr_fallback_used": used_ocr,
            }
        )
    return {
        "source": str(path),
        "kind": "pdf",
        "num_pages": len(doc),
        "pages_extracted": len(pages),
        "schema_version": "ocr_like_v1",
        "pages": pages,
    }


def extract_image(path: Path, lang: str) -> dict:
    img = Image.open(path)
    text = ocr_image_with_tesseract(path, lang=lang).strip()
    return {
        "source": str(path),
        "kind": "image",
        "num_pages": 1,
        "pages_extracted": 1,
        "schema_version": "ocr_like_v1",
        "pages": [
            {
                "page_index": 0,
                "size": {"width": float(img.width), "height": float(img.height)},
                "text": text,
                "blocks": [],
                "ocr_fallback_used": True,
            }
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pages", type=int, default=4)
    ap.add_argument("--ocr-fallback", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ocr-lang", type=str, default="eng")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    suffix = args.input.suffix.lower()
    if suffix == ".pdf":
        payload = extract_pdf(args.input, max_pages=args.max_pages, lang=args.ocr_lang, ocr_fallback=args.ocr_fallback)
    else:
        payload = extract_image(args.input, lang=args.ocr_lang)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")
    print(f"Pages extracted: {payload['pages_extracted']} / {payload['num_pages']}")


if __name__ == "__main__":
    main()
