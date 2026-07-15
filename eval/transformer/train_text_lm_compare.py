from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset

from eval.transformer.common import resolve_device, set_seed
from eval.transformer.lejepa import lejepa_loss, make_masked_views
from eval.transformer.train_long_context_compare import build_train_model


class JEPAPredictor(nn.Module):
    def __init__(self, d_model: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _isotropy_loss(z: torch.Tensor) -> torch.Tensor:
    # Encourage near-isotropic latent dimensions (LeJEPA-style anti-collapse bias).
    # z: [B, D]
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + 1e-6)
    cov = (z.T @ z) / max(z.size(0) - 1, 1)
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return ((cov - eye) ** 2).mean()


def text_to_byte_tensor(
    text: str,
    *,
    seq_len: int,
    device: torch.device,
    start: int | None = None,
) -> torch.Tensor | None:
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) < seq_len:
        return None
    if start is None:
        start = 0
    start = max(0, min(start, len(raw) - seq_len))
    ids = torch.tensor(list(raw[start : start + seq_len]), dtype=torch.long, device=device)
    return ids


def build_sample_plan(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    seq_len: int,
    total_sequences: int,
    seed: int,
) -> list[tuple[int, int]]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    rows = ds["text"] if "text" in ds.column_names else ds[ds.column_names[0]]
    rng = torch.Generator()
    rng.manual_seed(seed)

    plan: list[tuple[int, int]] = []
    for row_idx, row in enumerate(rows):
        if not isinstance(row, str):
            continue
        raw = row.encode("utf-8", errors="ignore")
        if len(raw) < seq_len + 1:
            continue
        max_start = len(raw) - (seq_len + 1)
        start = 0 if max_start <= 0 else int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
        plan.append((row_idx, start))
        if len(plan) >= total_sequences:
            break

    if not plan:
        raise RuntimeError("Could not build sample plan: no usable rows found")
    return plan


def build_batches(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    seq_len: int,
    max_batches: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    sample_plan: list[tuple[int, int]] | None = None,
) -> list[torch.Tensor]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    rows = ds["text"] if "text" in ds.column_names else ds[ds.column_names[0]]

    sequences = []
    if sample_plan is None:
        sample_plan = build_sample_plan(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            seq_len=seq_len,
            total_sequences=max_batches * batch_size,
            seed=seed,
        )

    for row_idx, start in sample_plan:
        if row_idx < 0 or row_idx >= len(rows):
            continue
        row = rows[row_idx]
        if not isinstance(row, str):
            continue
        t = text_to_byte_tensor(row, seq_len=seq_len + 1, device=device, start=start)
        if t is None:
            continue
        sequences.append(t)
        if len(sequences) >= max_batches * batch_size:
            break

    if len(sequences) < batch_size:
        raise RuntimeError("Not enough sequences to build a single batch")

    batches = []
    for i in range(0, len(sequences) - batch_size + 1, batch_size):
        chunk = sequences[i : i + batch_size]
        if len(chunk) < batch_size:
            break
        batches.append(torch.stack(chunk, dim=0))
        if len(batches) >= max_batches:
            break
    return batches


def _jepa_latent_loss(
    hidden: torch.Tensor,
    predictor: JEPAPredictor,
    *,
    mask_ratio: float,
) -> torch.Tensor:
    # JEPA-style latent prediction:
    # - context: visible tokens (unmasked)
    # - target: masked tokens (stop-grad target representation)
    bsz, seqlen, _ = hidden.shape
    device = hidden.device
    mask = torch.rand((bsz, seqlen), device=device) < mask_ratio
    # Prevent degenerate all-masked/all-visible per sample.
    for i in range(bsz):
        if not mask[i].any():
            mask[i, torch.randint(0, seqlen, (1,), device=device)] = True
        if mask[i].all():
            mask[i, torch.randint(0, seqlen, (1,), device=device)] = False

    visible = (~mask).unsqueeze(-1)
    target = mask.unsqueeze(-1)
    ctx = (hidden * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
    tgt = (hidden.detach() * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
    pred = predictor(ctx)
    return F.mse_loss(pred, tgt)


def _lejepa_proxy_latent_loss(
    hidden: torch.Tensor,
    predictor: JEPAPredictor,
    *,
    mask_ratio: float,
    isotropy_weight: float,
) -> torch.Tensor:
    bsz, seqlen, _ = hidden.shape
    device = hidden.device
    mask = torch.rand((bsz, seqlen), device=device) < mask_ratio
    for i in range(bsz):
        if not mask[i].any():
            mask[i, torch.randint(0, seqlen, (1,), device=device)] = True
        if mask[i].all():
            mask[i, torch.randint(0, seqlen, (1,), device=device)] = False

    visible = (~mask).unsqueeze(-1)
    target = mask.unsqueeze(-1)
    ctx = (hidden * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1.0)
    tgt = (hidden.detach() * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
    pred = predictor(ctx)
    pred = F.layer_norm(pred, pred.shape[-1:])
    tgt = F.layer_norm(tgt, tgt.shape[-1:])
    pred_loss = F.mse_loss(pred, tgt)
    iso_loss = _isotropy_loss(pred)
    return pred_loss + (isotropy_weight * iso_loss)


def _forward_hidden(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if model.config.use_byte_first:
        return model(byte_ids=input_ids, return_hidden=True)
    return model(token_ids=input_ids, return_hidden=True)


def _align_lm_targets(model: torch.nn.Module, targets: torch.Tensor, logits_length: int) -> torch.Tensor:
    """Align next-byte targets when byte patching shortens the sequence."""
    if targets.size(1) == logits_length:
        return targets
    patch_size = int(model.config.bytes.patch_size)
    indices = list(range(patch_size - 1, targets.size(1), patch_size))
    if len(indices) < logits_length:
        indices.append(targets.size(1) - 1)
    aligned = targets[:, indices[:logits_length]]
    if aligned.size(1) != logits_length:
        raise RuntimeError("Could not align LM targets with byte-patched hidden states")
    return aligned


def step_loss(
    model: torch.nn.Module,
    batch_ids: torch.Tensor,
    *,
    jepa_predictor: JEPAPredictor | None = None,
    jepa_mode: str = "none",
    jepa_weight: float = 0.0,
    jepa_mask_ratio: float = 0.4,
    lejepa_isotropy_weight: float = 0.1,
    lejepa_lambda: float = 0.1,
    lejepa_num_views: int = 2,
    lejepa_num_slices: int = 256,
    lejepa_num_knots: int = 17,
    lejepa_t_max: float = 5.0,
    view_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inp = batch_ids[:, :-1]
    tgt = batch_ids[:, 1:]
    logits, hidden = _forward_hidden(model, inp)
    tgt = _align_lm_targets(model, tgt, logits.size(1))
    lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    if jepa_weight <= 0.0 or jepa_mode == "none":
        zero = lm_loss.new_zeros(())
        return lm_loss, zero, lm_loss
    if jepa_mode == "jepa":
        if jepa_predictor is None:
            raise ValueError("jepa mode requires a predictor")
        jepa_loss = _jepa_latent_loss(hidden, jepa_predictor, mask_ratio=jepa_mask_ratio)
    elif jepa_mode == "lejepa_proxy":
        if jepa_predictor is None:
            raise ValueError("lejepa_proxy mode requires a predictor")
        jepa_loss = _lejepa_proxy_latent_loss(
            hidden,
            jepa_predictor,
            mask_ratio=jepa_mask_ratio,
            isotropy_weight=lejepa_isotropy_weight,
        )
    elif jepa_mode == "lejepa":
        # Bytes 256..259 are reserved by the 260-entry byte/symbol vocabulary,
        # so 256 is an unambiguous mask token for raw UTF-8 input.
        views = make_masked_views(
            inp,
            num_views=lejepa_num_views,
            mask_ratio=jepa_mask_ratio,
            mask_token_id=256,
            seed=view_seed,
        )
        embeddings = [_forward_hidden(model, view)[1].mean(dim=1) for view in views]
        jepa_loss = lejepa_loss(
            embeddings,
            lambd=lejepa_lambda,
            global_step=view_seed,
            num_slices=lejepa_num_slices,
            num_knots=lejepa_num_knots,
            t_max=lejepa_t_max,
        ).total
    else:
        raise ValueError(f"Unknown jepa_mode: {jepa_mode}")
    total = lm_loss + (jepa_weight * jepa_loss)
    return lm_loss, jepa_loss, total


def train_variant(
    *,
    variant: str,
    dataset_name: str,
    dataset_config: str | None,
    seq_len: int,
    batch_size: int,
    train_steps: int,
    eval_steps: int,
    lr: float,
    seed: int,
    device: torch.device,
    out_dir: Path,
    input_mode: str,
    byte_patching: bool,
    byte_patch_size: int,
    jepa_mode: str,
    jepa_loss_weight: float,
    jepa_mask_ratio: float,
    jepa_proj_dim: int,
    lejepa_isotropy_weight: float,
    lejepa_lambda: float,
    lejepa_num_views: int,
    lejepa_num_slices: int,
    lejepa_num_knots: int,
    lejepa_t_max: float,
    train_plan: list[tuple[int, int]] | None = None,
    eval_plan: list[tuple[int, int]] | None = None,
) -> dict:
    set_seed(seed)
    model = build_train_model(
        variant,
        device,
        model_size="tiny",
        input_mode=input_mode,
        attention_backend="auto",
        byte_patching=byte_patching,
        byte_patch_size=byte_patch_size,
    )
    jepa_predictor = None
    if jepa_mode in {"jepa", "lejepa_proxy"} and jepa_loss_weight > 0.0:
        jepa_predictor = JEPAPredictor(model.config.d_model, proj_dim=jepa_proj_dim).to(device)
    model.train()

    train_batches = build_batches(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="train",
        seq_len=seq_len,
        max_batches=max(train_steps, 64),
        batch_size=batch_size,
        device=device,
        seed=seed,
        sample_plan=train_plan,
    )
    eval_batches = build_batches(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="validation",
        seq_len=seq_len,
        max_batches=max(eval_steps, 16),
        batch_size=batch_size,
        device=device,
        seed=seed + 10_000,
        sample_plan=eval_plan,
    )

    params = list(model.parameters()) + (list(jepa_predictor.parameters()) if jepa_predictor is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    train_losses: list[float] = []
    train_lm_losses: list[float] = []
    train_jepa_losses: list[float] = []
    for i in range(train_steps):
        batch = train_batches[i % len(train_batches)]
        opt.zero_grad(set_to_none=True)
        lm_loss, jepa_loss, loss = step_loss(
            model,
            batch,
            jepa_predictor=jepa_predictor,
            jepa_mode=jepa_mode,
            jepa_weight=jepa_loss_weight,
            jepa_mask_ratio=jepa_mask_ratio,
            lejepa_isotropy_weight=lejepa_isotropy_weight,
            lejepa_lambda=lejepa_lambda,
            lejepa_num_views=lejepa_num_views,
            lejepa_num_slices=lejepa_num_slices,
            lejepa_num_knots=lejepa_num_knots,
            lejepa_t_max=lejepa_t_max,
            view_seed=seed * 1_000_003 + i,
        )
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        train_lm_losses.append(float(lm_loss.item()))
        train_jepa_losses.append(float(jepa_loss.item()))

    model.eval()
    eval_losses: list[float] = []
    eval_lm_losses: list[float] = []
    eval_jepa_losses: list[float] = []
    with torch.no_grad():
        for i in range(eval_steps):
            batch = eval_batches[i % len(eval_batches)]
            lm_loss, jepa_loss, loss = step_loss(
                model,
                batch,
                jepa_predictor=jepa_predictor,
                jepa_mode=jepa_mode,
                jepa_weight=jepa_loss_weight,
                jepa_mask_ratio=jepa_mask_ratio,
                lejepa_isotropy_weight=lejepa_isotropy_weight,
                lejepa_lambda=lejepa_lambda,
                lejepa_num_views=lejepa_num_views,
                lejepa_num_slices=lejepa_num_slices,
                lejepa_num_knots=lejepa_num_knots,
                lejepa_t_max=lejepa_t_max,
                view_seed=seed * 1_000_003 + 10_000_019 + i,
            )
            eval_losses.append(float(loss.item()))
            eval_lm_losses.append(float(lm_loss.item()))
            eval_jepa_losses.append(float(jepa_loss.item()))

    run_dir = out_dir / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "model.pt"
    metrics_path = run_dir / "metrics.json"
    torch.save(
        {
            "variant": variant,
            "seed": seed,
            "state_dict": model.state_dict(),
            "jepa_predictor_state_dict": (
                jepa_predictor.state_dict() if jepa_predictor is not None else None
            ),
            "optimizer_state_dict": opt.state_dict(),
            "config": model.config,
            "objective": {
                "jepa_mode": jepa_mode,
                "jepa_loss_weight": float(jepa_loss_weight),
                "jepa_mask_ratio": float(jepa_mask_ratio),
                "jepa_proj_dim": int(jepa_proj_dim),
                "lejepa_isotropy_weight": float(lejepa_isotropy_weight),
                "lejepa_lambda": float(lejepa_lambda),
                "lejepa_num_views": int(lejepa_num_views),
                "lejepa_num_slices": int(lejepa_num_slices),
                "lejepa_num_knots": int(lejepa_num_knots),
                "lejepa_t_max": float(lejepa_t_max),
            },
            "completed_steps": int(train_steps),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        ckpt_path,
    )
    metrics = {
        "variant": variant,
        "seed": seed,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "input_mode": input_mode,
        "byte_patching": bool(byte_patching),
        "byte_patch_size": int(byte_patch_size),
        "jepa_mode": jepa_mode,
        "jepa_loss_weight": float(jepa_loss_weight),
        "jepa_mask_ratio": float(jepa_mask_ratio),
        "jepa_proj_dim": int(jepa_proj_dim),
        "lejepa_isotropy_weight": float(lejepa_isotropy_weight),
        "lejepa_lambda": float(lejepa_lambda),
        "lejepa_num_views": int(lejepa_num_views),
        "lejepa_num_slices": int(lejepa_num_slices),
        "lejepa_num_knots": int(lejepa_num_knots),
        "lejepa_t_max": float(lejepa_t_max),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "train_loss_last": train_losses[-1],
        "train_loss_mean_last10": mean(train_losses[-10:]),
        "train_lm_loss_mean_last10": mean(train_lm_losses[-10:]),
        "train_jepa_loss_mean_last10": mean(train_jepa_losses[-10:]),
        "eval_loss_mean": mean(eval_losses),
        "eval_lm_loss_mean": mean(eval_lm_losses),
        "eval_jepa_loss_mean": mean(eval_jepa_losses),
        "eval_ppl": float(torch.exp(torch.tensor(mean(eval_lm_losses))).item()),
        "checkpoint": str(ckpt_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext"])
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--variants", nargs="+", default=["baseline", "engram"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/text_lm_compare"))
    parser.add_argument("--input-mode", choices=["byte", "symbolic"], default="byte")
    parser.add_argument(
        "--byte-patching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable byte patching when input-mode=byte.",
    )
    parser.add_argument("--byte-patch-size", type=int, default=1)
    parser.add_argument(
        "--batch-plan-in",
        type=Path,
        default=None,
        help="JSON file containing precomputed train/eval sample plans.",
    )
    parser.add_argument(
        "--batch-plan-out",
        type=Path,
        default=None,
        help="Write the generated deterministic train/eval sample plans to this JSON file.",
    )
    parser.add_argument(
        "--jepa-mode",
        choices=["none", "jepa", "lejepa_proxy", "lejepa"],
        default="none",
        help="lejepa is the SIGReg implementation; lejepa_proxy preserves the historical proxy.",
    )
    parser.add_argument("--jepa-loss-weight", type=float, default=0.0)
    parser.add_argument("--jepa-mask-ratio", type=float, default=0.4)
    parser.add_argument("--jepa-proj-dim", type=int, default=256)
    parser.add_argument("--lejepa-isotropy-weight", type=float, default=0.1)
    parser.add_argument("--lejepa-lambda", type=float, default=0.1)
    parser.add_argument("--lejepa-num-views", type=int, default=2)
    parser.add_argument("--lejepa-num-slices", type=int, default=256)
    parser.add_argument("--lejepa-num-knots", type=int, default=17)
    parser.add_argument("--lejepa-t-max", type=float, default=5.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.dataset == "wikitext":
        # Datasets 5 no longer resolves the legacy un-namespaced alias.
        dataset_name = "Salesforce/wikitext"
        dataset_config = args.dataset_config
    else:
        dataset_name = args.dataset
        dataset_config = None

    train_total = max(args.train_steps, 64) * args.batch_size
    eval_total = max(args.eval_steps, 16) * args.batch_size

    if args.batch_plan_in is not None:
        plan_payload = json.loads(args.batch_plan_in.read_text())
        train_plan = [(int(a), int(b)) for a, b in plan_payload["train"]]
        eval_plan = [(int(a), int(b)) for a, b in plan_payload["eval"]]
    else:
        train_plan = build_sample_plan(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split="train",
            seq_len=args.seq_len,
            total_sequences=train_total,
            seed=args.seed,
        )
        eval_plan = build_sample_plan(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split="validation",
            seq_len=args.seq_len,
            total_sequences=eval_total,
            seed=args.seed + 10_000,
        )
        if args.batch_plan_out is not None:
            args.batch_plan_out.parent.mkdir(parents=True, exist_ok=True)
            args.batch_plan_out.write_text(json.dumps({"train": train_plan, "eval": eval_plan}, indent=2))

    results = []
    for variant in args.variants:
        metrics = train_variant(
            variant=variant,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            lr=args.lr,
            seed=args.seed,
            device=device,
            out_dir=args.out_dir,
            input_mode=args.input_mode,
            byte_patching=args.byte_patching,
            byte_patch_size=args.byte_patch_size,
            jepa_mode=args.jepa_mode,
            jepa_loss_weight=args.jepa_loss_weight,
            jepa_mask_ratio=args.jepa_mask_ratio,
            jepa_proj_dim=args.jepa_proj_dim,
            lejepa_isotropy_weight=args.lejepa_isotropy_weight,
            lejepa_lambda=args.lejepa_lambda,
            lejepa_num_views=args.lejepa_num_views,
            lejepa_num_slices=args.lejepa_num_slices,
            lejepa_num_knots=args.lejepa_num_knots,
            lejepa_t_max=args.lejepa_t_max,
            train_plan=train_plan,
            eval_plan=eval_plan,
        )
        results.append(metrics)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
