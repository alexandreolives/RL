# Step 2 — LeJEPA on Text

Status: implementation `done`; remote CUDA smoke `passed`; paper-aligned
multi-seed evaluation `pending`

Scope:
- `engram_noconv + lejepa` paper-aligned auxiliary objective
- comparison vs `baseline` and `engram_noconv`
- same deterministic seed protocol

Implementation:
- independent masked text views are created before the encoder
- prediction uses the mean global-view center without stop-gradient
- SIGReg uses random normalized slices and the Epps-Pulley statistic
- `lambda` mixes prediction and SIGReg as in LeJEPA Algorithm 2
- no predictor or teacher network

The results in `notes/LEJEPA_RESULTS_2026-04-22.md` are historical proxy
results. They must not be attributed to the current `lejepa` mode; reproduce
them with `--jepa-mode lejepa_proxy`.

Folders:
- `notes/`: method notes and interpretation
- `scripts/run_real_lejepa_campaign.sh`: three-GPU deterministic training and
  paper-task evaluation using the archived control plans/checkpoints
- `configs/`: reserved for step-2 configs

Primary docs:
- `notes/REMOTE_GPU_VALIDATION_2026-07-15.md`
- `notes/LEJEPA_RESULTS_2026-04-22.md`
- `notes/DETERMINISTIC_PROXY_RUNBOOK.md`
