from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from eval.transformer.common import resolve_device, set_seed
from eval.transformer.long_context_accuracy import (
    TASK_FNS,
    _forward_answer_logits,
    make_multi_query_batch,
    make_multi_query_token_batch,
    make_passkey_batch,
    make_passkey_token_batch,
    make_variable_tracking_batch,
    make_variable_tracking_token_batch,
)
from models.atoms.norms import RMSNorm
from models.example import (
    apply_model_size,
    build_config,
    build_deepseek_v4_v4_config,
    build_deepseek_v4_v5_config,
    build_deepseek_v4_v6_config,
    build_model,
)
from models.molecules import TransformerMolecule


TRAIN_BATCH_FNS = {
    "passkey": make_passkey_batch,
    "multi_query": make_multi_query_batch,
    "variable_tracking": make_variable_tracking_batch,
}

TRAIN_TOKEN_BATCH_FNS = {
    "passkey": make_passkey_token_batch,
    "multi_query": make_multi_query_token_batch,
    "variable_tracking": make_variable_tracking_token_batch,
}


def init_distributed(requested_device: str) -> tuple[torch.device, bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or "1")
    rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1 or "LOCAL_RANK" in os.environ

    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    else:
        device = resolve_device(requested_device)

    if is_distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend=backend)
    return device, is_distributed, rank, world_size


def unwrap_model(model: TransformerMolecule | DistributedDataParallel) -> TransformerMolecule:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def forward_answer_logits_train(
    model: TransformerMolecule | DistributedDataParallel,
    *,
    seq_ids: torch.Tensor,
    modality_ids: torch.Tensor | None,
) -> torch.Tensor:
    base_model = unwrap_model(model)
    if base_model.config.use_byte_first:
        return model(byte_ids=seq_ids, modality_ids=modality_ids)[:, -1, :]
    return model(token_ids=seq_ids, modality_ids=modality_ids)[:, -1, :]


def sync_mean(value: float, *, device: torch.device, enabled: bool) -> float:
    if not enabled:
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def build_optimizer(
    model: TransformerMolecule,
    *,
    lr: float,
    weight_decay: float,
    engram_lr_scale: float,
    engram_embedding_weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    engram_decay_params = []
    engram_no_decay_params = []
    seen = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))

            full_name = f"{module_name}.{param_name}" if module_name else param_name
            is_engram = ".engram." in f".{full_name}."
            is_engram_embedding = is_engram and isinstance(module, nn.Embedding)
            is_bias_or_norm = param_name.endswith("bias") or isinstance(module, (nn.LayerNorm, RMSNorm))

            if is_engram_embedding:
                engram_no_decay_params.append(param)
            elif is_engram:
                if is_bias_or_norm:
                    engram_no_decay_params.append(param)
                else:
                    engram_decay_params.append(param)
            else:
                if is_bias_or_norm:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    if engram_decay_params:
        param_groups.append({"params": engram_decay_params, "lr": lr * engram_lr_scale, "weight_decay": weight_decay})
    if engram_no_decay_params:
        param_groups.append(
            {
                "params": engram_no_decay_params,
                "lr": lr * engram_lr_scale,
                "weight_decay": engram_embedding_weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)


def sample_batch(task: str, batch: int, seq_len: int, *, device: torch.device, input_mode: str):
    if input_mode == "byte":
        return TRAIN_BATCH_FNS[task](batch, seq_len, device=device)
    return TRAIN_TOKEN_BATCH_FNS[task](batch, seq_len, device=device)


def build_train_cache(
    task: str,
    batch: int,
    seq_len: int,
    *,
    device: torch.device,
    input_mode: str,
    cache_size: int,
):
    if cache_size <= 0:
        return None
    return [
        sample_batch(task, batch, seq_len, device=device, input_mode=input_mode)
        for _ in range(cache_size)
    ]


def build_train_model(
    name: str,
    device: torch.device,
    *,
    model_size: str,
    input_mode: str,
    attention_backend: str,
    byte_patching: bool = False,
    byte_patch_size: int = 1,
    activation: str = "gelu",
) -> TransformerMolecule:
    cfg = build_config(
        use_engram=True,
        use_dsa=False,
        use_mhc=False,
        use_moe=False,
        activation=activation,
        attention_backend=attention_backend,
    )
    cfg.multimodal.enabled = False
    if input_mode == "byte":
        cfg.use_byte_first = True
        cfg.bytes.use_byte_patching = False
        cfg.bytes.patch_size = 1
    elif input_mode == "symbolic":
        cfg.use_byte_first = False
        cfg.vocab_size = 260
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    if model_size == "tiny":
        cfg.d_model = 192
        cfg.depth = 4
        cfg.mlp_ratio = 2.0
        cfg.attention.num_heads = 4
        cfg.attention.num_kv_heads = 1
        cfg.attention.local_window = 128
        cfg.engram.heads = 4
        cfg.engram.insert_layers = (1, 3)
    elif model_size == "small":
        cfg.d_model = 256
        cfg.depth = 6
        cfg.mlp_ratio = 2.5
        cfg.attention.num_heads = 4
        cfg.attention.num_kv_heads = 1
        cfg.attention.local_window = 192
        cfg.engram.heads = 4
        cfg.engram.insert_layers = (1, 4)
    elif model_size != "base":
        raise ValueError(f"Unknown model_size: {model_size}")

    if name == "baseline":
        cfg.engram.enabled = False
    elif name == "attnres":
        cfg.engram.enabled = False
        cfg.use_attnres = True
    elif name == "mhc":
        cfg.engram.enabled = False
        cfg.use_mhc = True
        cfg.use_mhc_streams = True
        cfg.hc_mult = 4
        cfg.residual_branches = 4
    elif name == "mhc_attnres":
        cfg.engram.enabled = False
        cfg.use_mhc = True
        cfg.use_mhc_streams = True
        cfg.hc_mult = 4
        cfg.residual_branches = 4
        cfg.use_attnres = True
    elif name == "engram":
        pass
    elif name == "engram_layerhash":
        cfg.engram.use_layerwise_hash = True
    elif name == "engram_compress":
        cfg.engram.compressed_vocab_size = 128
        cfg.engram.compression_reserved_ids = 20
    elif name == "engram_official_gate":
        cfg.engram.official_gating = True
    elif name == "engram_noconv":
        cfg.engram.conv_enabled = False
    elif name == "engram_attnres":
        cfg.use_attnres = True
    elif name == "engram_noconv_attnres":
        cfg.engram.conv_enabled = False
        cfg.use_attnres = True
    elif name == "engram_noconv_attnres_v1":
        cfg.engram.conv_enabled = False
        cfg.use_attnres = True
        cfg.attnres_engram_mode = "fused"
    elif name == "engram_noconv_attnres_v2":
        cfg.engram.conv_enabled = False
        cfg.use_attnres = True
        cfg.attnres_engram_mode = "bypass"
    elif name == "engram_noconv_attnres_v3":
        cfg.engram.conv_enabled = False
        cfg.use_attnres = True
        cfg.attnres_engram_mode = "bounded_bypass"
        cfg.attnres_engram_gate_init = 0.1
    elif name == "engram_fullconv":
        cfg.engram.long_conv_enabled = True
    elif name == "full":
        cfg = build_config(
            use_engram=True,
            use_dsa=True,
            use_mhc=True,
            use_moe=True,
            activation="swiglu",
            attention_backend=attention_backend,
        )
        cfg.multimodal.enabled = False
        if input_mode == "byte":
            cfg.use_byte_first = True
            cfg.bytes.use_byte_patching = False
            cfg.bytes.patch_size = 1
        elif input_mode == "symbolic":
            cfg.use_byte_first = False
            cfg.vocab_size = 260
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
    elif name in {"deepseek_v4_like", "v1", "deepseek_v4_v1"}:
        cfg.engram.enabled = False
        cfg.use_mhc = True
        cfg.use_dsa = True
        cfg.use_multibranch_residual = False
        cfg.residual_branches = 4
        cfg.activation = "swiglu"
        cfg.attention.qk_norm = True
        cfg.attention.num_kv_heads = 1
        cfg.attention.local_window = 128
        cfg.attention.dsa_top_k = 64
        cfg.attention.dsa_indexer_hidden = 128
        cfg.moe.enabled = True
        cfg.moe.num_experts = 8
        cfg.moe.top_k = 4
        cfg.moe.shared_expert = True
        cfg.moe.router_jitter = 0.0
    elif name in {"v2", "deepseek_v4_v2"}:
        cfg = build_config(
            use_engram=True,
            use_dsa=True,
            use_mhc=True,
            use_moe=True,
            activation="swiglu",
            attention_backend=attention_backend,
        )
        cfg.multimodal.enabled = False
        if input_mode == "byte":
            cfg.use_byte_first = True
            cfg.bytes.use_byte_patching = False
            cfg.bytes.patch_size = 1
        elif input_mode == "symbolic":
            cfg.use_byte_first = False
            cfg.vocab_size = 260
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        cfg.attention.qk_norm = True
        cfg.attention.num_kv_heads = 1
        cfg.attention.local_window = 128
        cfg.attention.dsa_top_k = 64
        cfg.attention.dsa_indexer_hidden = 128
        cfg.use_multibranch_residual = False
        cfg.residual_branches = 4
        cfg.moe.num_experts = 8
        cfg.moe.top_k = 4
        cfg.moe.shared_expert = True
        cfg.moe.router_jitter = 0.0
        cfg.engram.use_layerwise_hash = True
        cfg.engram.conv_enabled = False
        cfg.engram.long_conv_enabled = False
        cfg.engram.insert_layers = (0, 2, 4)
    elif name in {"v3", "deepseek_v4_v3", "deepseek_v4_public_like"}:
        cfg = build_config(
            use_engram=True,
            use_dsa=True,
            use_mhc=True,
            use_moe=True,
            activation="swiglu",
            attention_backend=attention_backend,
        )
        cfg.multimodal.enabled = False
        if input_mode == "byte":
            cfg.use_byte_first = True
            cfg.bytes.use_byte_patching = False
            cfg.bytes.patch_size = 1
        elif input_mode == "symbolic":
            cfg.use_byte_first = False
            cfg.vocab_size = 260
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        cfg.layer_types = (
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "heavily_compressed_attention",
        )
        cfg.attention.qk_norm = True
        cfg.attention.num_kv_heads = 1
        cfg.attention.local_window = 128
        cfg.attention.dsa_top_k = 64
        cfg.attention.dsa_indexer_hidden = 128
        cfg.use_multibranch_residual = False
        cfg.residual_branches = 4
        cfg.moe.num_experts = 8
        cfg.moe.top_k = 6
        cfg.moe.shared_expert = True
        cfg.moe.scoring_func = "sqrtsoftplus"
        cfg.moe.norm_topk_prob = True
        cfg.moe.routed_scaling_factor = 2.5
        cfg.engram.use_layerwise_hash = True
        cfg.engram.conv_enabled = False
        cfg.engram.long_conv_enabled = False
        cfg.engram.insert_layers = (0, 1, 2)
    elif name in {"v4", "deepseek_v4_v4"}:
        cfg = build_deepseek_v4_v4_config(attention_backend=attention_backend, input_mode=input_mode)
    elif name in {"v5", "deepseek_v4_v5"}:
        cfg = build_deepseek_v4_v5_config(attention_backend=attention_backend, input_mode=input_mode)
    elif name in {
        "v6",
        "deepseek_v4_v6",
        "deepseek_v4_reference",
        "deepseek_v4_public_exact",
        "deepseek_v4_public_solid",
    }:
        cfg = build_deepseek_v4_v6_config(attention_backend=attention_backend, input_mode=input_mode)
    elif name in {"v6_engram", "deepseek_v4_v6_engram"}:
        cfg = build_deepseek_v4_v6_config(attention_backend=attention_backend, input_mode=input_mode)
        cfg.implementation = "native"
        cfg.engram.enabled = True
        cfg.engram.conv_enabled = False
        cfg.engram.insert_layers = (1, 4)
    elif name in {"v6_engram_attnres", "deepseek_v4_v6_engram_attnres"}:
        cfg = build_deepseek_v4_v6_config(attention_backend=attention_backend, input_mode=input_mode)
        cfg.implementation = "native"
        cfg.engram.enabled = True
        cfg.engram.conv_enabled = False
        cfg.engram.insert_layers = (1, 4)
        cfg.use_attnres = True
    elif name == "full_noconv":
        cfg = build_config(
            use_engram=True,
            use_dsa=True,
            use_mhc=True,
            use_moe=True,
            activation="swiglu",
            attention_backend=attention_backend,
        )
        cfg.multimodal.enabled = False
        if input_mode == "byte":
            cfg.use_byte_first = True
            cfg.bytes.use_byte_patching = False
            cfg.bytes.patch_size = 1
        elif input_mode == "symbolic":
            cfg.use_byte_first = False
            cfg.vocab_size = 260
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        cfg.engram.conv_enabled = False
    else:
        raise ValueError(f"Unknown variant: {name}")

    cfg = apply_model_size(cfg, model_size, input_mode=input_mode)
    if input_mode == "byte":
        # This must happen before model construction so BytePatcher is created.
        cfg.bytes.use_byte_patching = bool(byte_patching)
        cfg.bytes.patch_size = int(byte_patch_size)
    return build_model(cfg).to(device)


def train_variant(
    name: str,
    *,
    train_task: str,
    eval_tasks: list[str],
    seq_len: int,
    batch: int,
    train_steps: int,
    eval_steps: int,
    lr: float,
    weight_decay: float,
    grad_accum: int,
    model_size: str,
    input_mode: str,
    attention_backend: str,
    engram_lr_scale: float,
    engram_embedding_weight_decay: float,
    seed: int,
    device: torch.device,
    fixed_batch: bool,
    train_cache_size: int,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict:
    set_seed(seed + rank * 1000)
    model = build_train_model(name, device, model_size=model_size, input_mode=input_mode, attention_backend=attention_backend)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    model.train()
    optimizer = build_optimizer(
        unwrap_model(model),
        lr=lr,
        weight_decay=weight_decay,
        engram_lr_scale=engram_lr_scale,
        engram_embedding_weight_decay=engram_embedding_weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    effective_cache_size = max(train_cache_size, 1 if fixed_batch else 0)
    train_cache = build_train_cache(
        train_task,
        batch,
        seq_len,
        device=device,
        input_mode=input_mode,
        cache_size=effective_cache_size,
    )

    loss_history = []
    acc_history = []
    cache_cursor = 0
    for _step in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        step_accs = []
        for _micro in range(grad_accum):
            if train_cache is None:
                seq_ids, modality_ids, answers = sample_batch(train_task, batch, seq_len, device=device, input_mode=input_mode)
            else:
                seq_ids, modality_ids, answers = train_cache[cache_cursor]
                cache_cursor = (cache_cursor + 1) % len(train_cache)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                answer_logits = forward_answer_logits_train(model, seq_ids=seq_ids, modality_ids=modality_ids)
                loss = F.cross_entropy(answer_logits, answers) / grad_accum

            scaler.scale(loss).backward()
            step_losses.append(loss.item() * grad_accum)
            step_accs.append((answer_logits.argmax(dim=-1) == answers).float().mean().item())

        scaler.step(optimizer)
        scaler.update()

        loss_history.append(sync_mean(mean(step_losses), device=device, enabled=distributed))
        acc_history.append(sync_mean(mean(step_accs), device=device, enabled=distributed))

    base_model = unwrap_model(model)
    if distributed:
        dist.barrier()
    if rank != 0:
        return {}

    base_model.eval()
    eval_results = {}
    for task in eval_tasks:
        metrics = TASK_FNS[task](base_model, batch=batch, seq_len=seq_len, steps=eval_steps, device=device, input_mode=input_mode)
        eval_results[task] = {k: round(v, 6) for k, v in metrics.items()}

    return {
        "variant": name,
        "train_task": train_task,
        "seq_len": seq_len,
        "batch": batch,
        "grad_accum": grad_accum,
        "model_size": model_size,
        "input_mode": input_mode,
        "attention_backend": attention_backend,
        "engram_lr_scale": engram_lr_scale,
        "engram_embedding_weight_decay": engram_embedding_weight_decay,
        "seed": seed,
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "fixed_batch": fixed_batch,
        "train_cache_size": effective_cache_size,
        "world_size": world_size,
        "rank": rank,
        "device": str(device),
        "train_loss_final": round(loss_history[-1], 6),
        "train_loss_mean_last10": round(mean(loss_history[-10:]), 6),
        "train_acc_mean_last10": round(mean(acc_history[-10:]), 6),
        "eval": eval_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--train-task", default="passkey", choices=list(TRAIN_BATCH_FNS))
    parser.add_argument("--input-mode", default="byte", choices=["byte", "symbolic"])
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "eager", "sdpa", "flash"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=["passkey", "multi_query", "variable_tracking"],
        choices=list(TRAIN_BATCH_FNS),
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--engram-lr-scale", type=float, default=5.0)
    parser.add_argument("--engram-embedding-weight-decay", type=float, default=0.0)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--train-cache-size", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "engram", "engram_layerhash", "v1", "v2", "v3", "v4", "v5", "v6"],
    )
    args = parser.parse_args()

    device, distributed, rank, world_size = init_distributed(args.device)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    results = []
    for name in args.variants:
        variant_runs = [
            train_variant(
                name,
                train_task=args.train_task,
                eval_tasks=args.eval_tasks,
                seq_len=args.seq_len,
                batch=args.batch,
                train_steps=args.train_steps,
                eval_steps=args.eval_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_accum=args.grad_accum,
                model_size=args.model_size,
                input_mode=args.input_mode,
                attention_backend=args.attention_backend,
                engram_lr_scale=args.engram_lr_scale,
                engram_embedding_weight_decay=args.engram_embedding_weight_decay,
                seed=seed,
                device=device,
                fixed_batch=args.fixed_batch,
                train_cache_size=args.train_cache_size,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )
            for seed in seeds
        ]
        if rank != 0:
            continue
        if len(variant_runs) == 1:
            results.append(variant_runs[0])
            continue

        eval_summary = {}
        for task in args.eval_tasks:
            task_metrics = variant_runs[0]["eval"][task].keys()
            eval_summary[task] = {
                metric: round(mean(run["eval"][task][metric] for run in variant_runs), 6)
                for metric in task_metrics
            }

        results.append(
            {
                "variant": name,
                "train_task": args.train_task,
                "seq_len": args.seq_len,
                "batch": args.batch,
                "grad_accum": args.grad_accum,
                "model_size": args.model_size,
                "input_mode": args.input_mode,
                "attention_backend": args.attention_backend,
                "engram_lr_scale": args.engram_lr_scale,
                "engram_embedding_weight_decay": args.engram_embedding_weight_decay,
                "seeds": seeds,
                "train_steps": args.train_steps,
                "eval_steps": args.eval_steps,
                "fixed_batch": args.fixed_batch,
                "train_cache_size": max(args.train_cache_size, 1 if args.fixed_batch else 0),
                "world_size": world_size,
                "device": str(device),
                "train_loss_final_mean": round(mean(run["train_loss_final"] for run in variant_runs), 6),
                "train_loss_mean_last10_mean": round(mean(run["train_loss_mean_last10"] for run in variant_runs), 6),
                "train_acc_mean_last10_mean": round(mean(run["train_acc_mean_last10"] for run in variant_runs), 6),
                "eval": eval_summary,
                "runs": variant_runs,
            }
        )
    if rank == 0:
        print(json.dumps(results, indent=2))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
