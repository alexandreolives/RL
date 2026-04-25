# Docker Runtime Notes (Optional)

This repository can run without Docker.

If you use Docker, there are two supported modes:
- local host execution,
- remote host execution via `REMOTE_HOST`.

Some launcher names still include `_wsl2` for backward compatibility, but they
are no longer hard-wired to WSL2.

## Images

- base runtime: `docker/Dockerfile.wsl2.gpu` -> `rl-engram:gpu`
- eval runtime: `docker/Dockerfile.wsl2.gpu.eval` -> `rl-engram:gpu-eval`
- ocr runtime: `docker/Dockerfile.wsl2.gpu.ocr` -> `rl-engram:gpu-ocr`

## 1) Build images

Local:

```bash
docker build -f docker/Dockerfile.wsl2.gpu -t rl-engram:gpu .
docker build -f docker/Dockerfile.wsl2.gpu.eval -t rl-engram:gpu-eval .
docker build -f docker/Dockerfile.wsl2.gpu.ocr -t rl-engram:gpu-ocr .
```

Remote:

```bash
REMOTE_HOST=my-server REMOTE_REPO='$HOME/RL/engram' \
  bash scripts/setup_ocr_like.sh
```

## 2) Run examples

Step 1 local:

```bash
bash scripts/run_long_context_compare_ddp.sh
```

Step 1 remote:

```bash
REMOTE_HOST=my-server REMOTE_REPO='$HOME/RL/engram' \
  bash scripts/run_long_context_compare_ddp.sh
```

Step 3 OCR synthetic train local:

```bash
GPU_ID=0 VARIANT=engram_noconv \
  bash scripts/run_train_deepseek_like_ocr_synth.sh
```

Step 3 OCR synthetic train remote:

```bash
REMOTE_HOST=my-server REMOTE_REPO='$HOME/RL/engram' GPU_ID=0 VARIANT=engram_noconv \
  bash scripts/run_train_deepseek_like_ocr_synth.sh
```

## 3) Harness setup

Local:

```bash
bash scripts/setup_eval_harnesses.sh
```

Remote:

```bash
REMOTE_HOST=my-server HARNESS_ROOT='$HOME/RL/harnesses' \
  bash scripts/setup_eval_harnesses.sh
```

## 4) Notes

- Use `REMOTE_HOST` only if execution must happen on another machine.
- `GPU_ID` and `CUDA_VISIBLE_DEVICES` controls are exposed by launchers.
- For strict reproducibility, pin external harness repos to commit hashes.
