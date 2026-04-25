# Getting Started

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 2) Verify runtime

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3) Navigate by experiment step

- Step 1: `experiments/step1_engram_core/`
- Step 2: `experiments/step2_engram_lejepa_eval/`
- Step 3: `experiments/step3_ocr_like/`

Global status:
- `experiments/ROADMAP.md`

## 4) Optional container runtime

If you prefer Docker launchers (local or remote host):
- `docker/README.md`
