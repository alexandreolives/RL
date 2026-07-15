# Multimodal Roadmap (DeepSeek OCR-like)

## Status global

- `Step 1`: done (baseline/engram/engram_noconv text-byte)
- `Step 2`: done; the paper-aligned nine-seed text-byte evaluation found a
  small, statistically inconclusive regression against `engram_noconv`, which
  remains the preferred default
- `Step 3`: in progress (OCR-like foundation)

## Step 3 — OCR-like Foundation (`experiments/step3_ocr_like`)

Goal:
- build a minimal document pipeline inspired by DeepSeek OCR:
  - ingest PDF/image
  - extract text + layout + reading order
  - produce normalized doc records

Outputs:
- scripts for extraction
- canonical JSON schema
- smoke dataset + sanity checks

## Step 4 — Engram Integration (`experiments/step4_engram_integration`)

Goal:
- plug `engram_noconv` into the doc pipeline representation path.

Outputs:
- training/eval scripts for doc representation
- configs for deterministic runs

## Step 5 — Hyper-token Compression (`experiments/step5_hypertoken_compression`)

Goal:
- compress doc semantics into dense hyper-tokens while keeping factual recovery.

Outputs:
- compression module
- reconstruction/factual probes

## Step 6 — Doc Benchmark (`experiments/step6_doc_benchmark`)

Goal:
- compare:
  - baseline doc pipeline
  - `engram_noconv`
  - `engram_noconv + perceptual lejepa`

Metrics:
- factual EM/F1
- reasoning accuracy
- latency + VRAM
