# Step 3 — OCR-like Foundation

Status: `in_progress`

First MVP script:
- `scripts/extract_doc_ocr_like.py`
- `scripts/benchmark_ocr_like.py`
- `scripts/deepseek_like_ocr_model.py`
- `scripts/run_deepseek_like_ocr_smoke.py`
- `scripts/train_deepseek_like_ocr_synth.py`

What it does:
- PDF ingest via PyMuPDF
- extracts page text blocks with bounding boxes
- optional OCR fallback with `tesseract` if native text is empty
- writes normalized JSON (`schema_version=ocr_like_v1`)

Optional Docker launchers:
- `scripts/setup_ocr_like.sh`
- `scripts/run_ocr_like_batch.sh`
- `scripts/run_benchmark_ocr_like.sh`
- `scripts/run_train_deepseek_like_ocr_synth.sh`

Note:
- script names keep `_wsl2` for backward compatibility, but they now support
  local execution by default and optional remote execution via `REMOTE_HOST`.
- public wrapper names are available in `scripts/` (without `_wsl2` suffix) for
  Step 1; Step 3 wrappers can be called directly from `experiments/step3_ocr_like/scripts/`.

DeepSeek-like copy/adapt (current):
- visual compression encoder (patch -> compressed visual tokens)
- prefix fusion with byte-text stream
- decoder replaced by our `engram_noconv` stack

Preferred inputs for now:
- existing datasets (`FUNSD`)
- synthetic text rendered as image

PDF batch extraction is secondary and optional.

Current training target:
- synthetic document image -> exact text bytes
- same decoder variants as Step 1 (`baseline`, `engram_noconv`)
- deterministic corpus from seed, so A/B runs are comparable

Quick run:

```bash
python experiments/step3_ocr_like/scripts/extract_doc_ocr_like.py \
  --input papers/bycloud/<file>.pdf \
  --out artifacts/step3_ocr_like/sample.json \
  --max-pages 3

VARIANT=engram_noconv GPU_ID=0 TRAIN_STEPS=200 \
  bash scripts/run_train_deepseek_like_ocr_synth.sh
```
