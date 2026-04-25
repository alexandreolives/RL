from __future__ import annotations

import argparse
import json
from statistics import mean

import torch

from eval.transformer.common import resolve_device, set_seed
from models.example import build_config
from models.molecules import TransformerMolecule


PASSKEY_MARKER = 250
QUERY_MARKER = 251
ANSWER_MARKER = 252
ASSIGN_MARKER = 253
TRACK_MARKER = 254
FILLER_MAX = 200


def build_long_context_model(
    name: str,
    device: torch.device,
    *,
    input_mode: str = "byte",
    attention_backend: str = "auto",
) -> TransformerMolecule:
    cfg = build_config(
        use_engram=True,
        use_dsa=False,
        use_mhc=False,
        use_moe=False,
        activation="gelu",
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
        cfg.multimodal.enabled = False
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    if name == "baseline":
        cfg.engram.enabled = False
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
    elif name == "engram_fullconv":
        cfg.engram.long_conv_enabled = True
    else:
        raise ValueError(f"Unknown variant: {name}")

    return TransformerMolecule(cfg).to(device).eval()


def make_passkey_batch(batch: int, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_len < 16:
        raise ValueError("seq_len must be at least 16 for the passkey task")

    byte_ids = torch.randint(0, FILLER_MAX, (batch, seq_len), device=device)
    answers = torch.randint(0, FILLER_MAX, (batch,), device=device)
    query_pos = seq_len - 3

    for i in range(batch):
        key_pos = seq_len // 3
        ans = answers[i]
        byte_ids[i, key_pos] = PASSKEY_MARKER
        byte_ids[i, key_pos + 1] = ans
        byte_ids[i, key_pos + 2] = ANSWER_MARKER
        byte_ids[i, query_pos] = QUERY_MARKER
        byte_ids[i, query_pos + 1] = PASSKEY_MARKER
        byte_ids[i, query_pos + 2] = 0

    modality_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    return byte_ids, modality_ids, answers


def make_passkey_token_batch(batch: int, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor, None, torch.Tensor]:
    token_ids, _, answers = make_passkey_batch(batch, seq_len, device=device)
    return token_ids, None, answers


def _forward_answer_logits(model: TransformerMolecule, *, seq_ids: torch.Tensor, modality_ids: torch.Tensor | None) -> torch.Tensor:
    if model.config.use_byte_first:
        return model(byte_ids=seq_ids, modality_ids=modality_ids)[:, -1, :]
    return model(token_ids=seq_ids, modality_ids=modality_ids)[:, -1, :]


def evaluate_passkey(model: TransformerMolecule, *, batch: int, seq_len: int, steps: int, device: torch.device, input_mode: str = "byte") -> dict[str, float]:
    accuracies = []
    target_probs = []
    target_ranks = []

    with torch.no_grad():
        for _ in range(steps):
            if input_mode == "byte":
                seq_ids, modality_ids, answers = make_passkey_batch(batch, seq_len, device=device)
            else:
                seq_ids, modality_ids, answers = make_passkey_token_batch(batch, seq_len, device=device)
            answer_logits = _forward_answer_logits(model, seq_ids=seq_ids, modality_ids=modality_ids)
            pred = answer_logits.argmax(dim=-1)
            accuracies.append((pred == answers).float().mean().item())

            probs = torch.softmax(answer_logits, dim=-1)
            target_probs.append(probs[torch.arange(batch, device=device), answers].mean().item())

            greater = (answer_logits > answer_logits[torch.arange(batch, device=device), answers].unsqueeze(-1)).sum(dim=-1)
            target_ranks.append((greater + 1).float().mean().item())

    chance = 1.0 / model.output_vocab_size
    return {
        "accuracy": mean(accuracies),
        "target_prob": mean(target_probs),
        "target_rank": mean(target_ranks),
        "chance_accuracy": chance,
    }


def summarize_logits(answer_logits: torch.Tensor, answers: torch.Tensor, *, vocab_size: int) -> dict[str, float]:
    pred = answer_logits.argmax(dim=-1)
    accuracy = (pred == answers).float().mean().item()
    probs = torch.softmax(answer_logits, dim=-1)
    target_prob = probs[torch.arange(answer_logits.size(0), device=answer_logits.device), answers].mean().item()
    greater = (answer_logits > answer_logits[torch.arange(answer_logits.size(0), device=answer_logits.device), answers].unsqueeze(-1)).sum(dim=-1)
    target_rank = (greater + 1).float().mean().item()
    return {
        "accuracy": accuracy,
        "target_prob": target_prob,
        "target_rank": target_rank,
        "chance_accuracy": 1.0 / vocab_size,
    }


def make_multi_query_batch(batch: int, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_len < 32:
        raise ValueError("seq_len must be at least 32 for multi_query")

    byte_ids = torch.randint(0, FILLER_MAX, (batch, seq_len), device=device)
    answers = torch.randint(0, FILLER_MAX, (batch,), device=device)
    query_pos = seq_len - 4
    slot_positions = [seq_len // 5, (2 * seq_len) // 5, (3 * seq_len) // 5]

    for i in range(batch):
        query_slot = torch.randint(0, len(slot_positions), (1,), device=device).item()
        for slot_idx, pos in enumerate(slot_positions):
            key_id = 40 + slot_idx
            value = answers[i] if slot_idx == query_slot else torch.randint(0, FILLER_MAX, (1,), device=device)[0]
            byte_ids[i, pos] = PASSKEY_MARKER
            byte_ids[i, pos + 1] = key_id
            byte_ids[i, pos + 2] = value
            byte_ids[i, pos + 3] = ANSWER_MARKER
        byte_ids[i, query_pos] = QUERY_MARKER
        byte_ids[i, query_pos + 1] = 40 + query_slot
        byte_ids[i, query_pos + 2] = PASSKEY_MARKER
        byte_ids[i, query_pos + 3] = 0

    modality_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    return byte_ids, modality_ids, answers


def make_multi_query_token_batch(batch: int, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor, None, torch.Tensor]:
    token_ids, _, answers = make_multi_query_batch(batch, seq_len, device=device)
    return token_ids, None, answers


def evaluate_multi_query(model: TransformerMolecule, *, batch: int, seq_len: int, steps: int, device: torch.device, input_mode: str = "byte") -> dict[str, float]:
    metrics = []
    with torch.no_grad():
        for _ in range(steps):
            if input_mode == "byte":
                seq_ids, modality_ids, answers = make_multi_query_batch(batch, seq_len, device=device)
            else:
                seq_ids, modality_ids, answers = make_multi_query_token_batch(batch, seq_len, device=device)
            answer_logits = _forward_answer_logits(model, seq_ids=seq_ids, modality_ids=modality_ids)
            metrics.append(summarize_logits(answer_logits, answers, vocab_size=model.output_vocab_size))
    return {key: mean(item[key] for item in metrics) for key in metrics[0]}


def make_variable_tracking_batch(batch: int, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_len < 40:
        raise ValueError("seq_len must be at least 40 for variable_tracking")

    byte_ids = torch.randint(0, FILLER_MAX, (batch, seq_len), device=device)
    answers = torch.randint(0, FILLER_MAX, (batch,), device=device)
    query_pos = seq_len - 4

    for i in range(batch):
        pos = seq_len // 6
        symbols = [60, 61, 62]
        value = answers[i]
        byte_ids[i, pos] = ASSIGN_MARKER
        byte_ids[i, pos + 1] = symbols[0]
        byte_ids[i, pos + 2] = value
        byte_ids[i, pos + 3] = ANSWER_MARKER

        pos2 = (2 * seq_len) // 6
        byte_ids[i, pos2] = ASSIGN_MARKER
        byte_ids[i, pos2 + 1] = symbols[1]
        byte_ids[i, pos2 + 2] = symbols[0]
        byte_ids[i, pos2 + 3] = ANSWER_MARKER

        pos3 = (3 * seq_len) // 6
        byte_ids[i, pos3] = ASSIGN_MARKER
        byte_ids[i, pos3 + 1] = symbols[2]
        byte_ids[i, pos3 + 2] = symbols[1]
        byte_ids[i, pos3 + 3] = ANSWER_MARKER

        byte_ids[i, query_pos] = TRACK_MARKER
        byte_ids[i, query_pos + 1] = symbols[2]
        byte_ids[i, query_pos + 2] = QUERY_MARKER
        byte_ids[i, query_pos + 3] = 0

    modality_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    return byte_ids, modality_ids, answers


def make_variable_tracking_token_batch(batch: int, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor, None, torch.Tensor]:
    token_ids, _, answers = make_variable_tracking_batch(batch, seq_len, device=device)
    return token_ids, None, answers


def evaluate_variable_tracking(model: TransformerMolecule, *, batch: int, seq_len: int, steps: int, device: torch.device, input_mode: str = "byte") -> dict[str, float]:
    metrics = []
    with torch.no_grad():
        for _ in range(steps):
            if input_mode == "byte":
                seq_ids, modality_ids, answers = make_variable_tracking_batch(batch, seq_len, device=device)
            else:
                seq_ids, modality_ids, answers = make_variable_tracking_token_batch(batch, seq_len, device=device)
            answer_logits = _forward_answer_logits(model, seq_ids=seq_ids, modality_ids=modality_ids)
            metrics.append(summarize_logits(answer_logits, answers, vocab_size=model.output_vocab_size))
    return {key: mean(item[key] for item in metrics) for key in metrics[0]}


TASK_FNS = {
    "passkey": evaluate_passkey,
    "multi_query": evaluate_multi_query,
    "variable_tracking": evaluate_variable_tracking,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--input-mode", default="byte", choices=["byte", "symbolic"])
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "eager", "sdpa", "flash"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="passkey", choices=["passkey", "multi_query", "variable_tracking"])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "engram", "engram_noconv", "engram_fullconv"],
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    task_fn = TASK_FNS[args.task]
    results = []
    for name in args.variants:
        model = build_long_context_model(name, device, input_mode=args.input_mode, attention_backend=args.attention_backend)
        metrics = task_fn(model, batch=args.batch, seq_len=args.seq_len, steps=args.steps, device=device, input_mode=args.input_mode)
        results.append(
            {
                "variant": name,
                "input_mode": args.input_mode,
                "attention_backend": args.attention_backend,
                "seed": args.seed,
                "task": args.task,
                "seq_len": args.seq_len,
                "batch": args.batch,
                "steps": args.steps,
                "device": str(device),
                "accuracy": round(metrics["accuracy"], 6),
                "target_prob": round(metrics["target_prob"], 6),
                "target_rank": round(metrics["target_rank"], 3),
                "chance_accuracy": round(metrics["chance_accuracy"], 6),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
