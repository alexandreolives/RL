from __future__ import annotations

import argparse
import json
from dataclasses import is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from eval.transformer.common import resolve_device
from models.atoms.attention import flash_attn_func
from models.molecules import TransformerMolecule


def safe_load_dataset(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except Exception:
        return None


def load_model(ckpt_path: Path, device: torch.device) -> TransformerMolecule:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if is_dataclass(cfg):
        cfg = cfg
    elif isinstance(cfg, dict):
        raise RuntimeError("Checkpoint config must be dataclass object, got dict")
    if cfg.attention.backend == "flash" and flash_attn_func is None:
        cfg.attention.backend = "auto"
    model = TransformerMolecule(cfg).to(device)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    # Checkpoints produced before f94aaef do not contain the Lightning Indexer
    # parameters. They are dormant for the baseline/Engram controls because
    # those configs do not use compressed attention. Reject every other schema
    # mismatch so compatibility does not silently hide a real model change.
    allowed_missing_fragments = (
        ".attn.lightning_indexer.",
        ".attn.lightning_q.",
        ".attn.lightning_k.",
    )
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not any(fragment in key for fragment in allowed_missing_fragments)
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Unsupported checkpoint schema mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def byte_ids(text: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long, device=device)


def choice_scores(
    model: TransformerMolecule,
    context: str,
    choice: str,
    *,
    max_len: int,
    device: torch.device,
) -> tuple[float, float]:
    c = byte_ids(context, device)
    y = byte_ids(choice, device)
    if c.numel() == 0 or y.numel() == 0:
        return float("-inf"), float("-inf")
    full = torch.cat([c, y], dim=0)
    if full.numel() < 2:
        return float("-inf"), float("-inf")
    truncated_prefix = max(0, full.numel() - max_len)
    if full.numel() > max_len:
        full = full[-max_len:]
    inp = full[:-1].unsqueeze(0)
    tgt = full[1:].unsqueeze(0)
    # Target positions that correspond to choice bytes in the shifted target.
    # If we left-truncate, the boundary between context and choice shifts too.
    c_len_after_trunc = max(0, c.numel() - truncated_prefix)
    start = max(0, c_len_after_trunc - 1)
    with torch.no_grad():
        logits = model(byte_ids=inp)
        logp = F.log_softmax(logits, dim=-1)
        tok = torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
        choice_tok = tok[:, start:]
        total = float(choice_tok.sum().item())
        normalized = float(choice_tok.mean().item())
        return total, normalized


def eval_arc(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    correct = 0
    total = 0
    for row in ds:
        question = row["question"]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer = row["answerKey"]
        if not labels or not texts:
            continue
        scores = []
        for txt in texts:
            _raw, norm = choice_scores(model, question + "\nAnswer:", " " + txt, max_len=max_len, device=device)
            scores.append(norm)
        pred = labels[int(torch.tensor(scores).argmax().item())]
        correct += int(pred == answer)
        total += 1
        if total >= limit:
            break
    return {"accuracy": correct / max(total, 1), "n": total}


def eval_arc_easy(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    correct = 0
    total = 0
    for row in ds:
        question = row["question"]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer = row["answerKey"]
        if not labels or not texts:
            continue
        scores = []
        for txt in texts:
            _raw, norm = choice_scores(model, question + "\nAnswer:", " " + txt, max_len=max_len, device=device)
            scores.append(norm)
        pred = labels[int(torch.tensor(scores).argmax().item())]
        correct += int(pred == answer)
        total += 1
        if total >= limit:
            break
    return {"accuracy": correct / max(total, 1), "n": total}


def eval_hellaswag(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    correct = 0
    total = 0
    for row in ds:
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        scores = []
        for e in endings:
            _raw, norm = choice_scores(model, ctx, " " + e, max_len=max_len, device=device)
            scores.append(norm)
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == label)
        total += 1
        if total >= limit:
            break
    return {"accuracy": correct / max(total, 1), "n": total}


def load_mmlu_split():
    # try common dataset names used in practice.
    try:
        return load_dataset("cais/mmlu", "all", split="validation")
    except Exception:
        return load_dataset("hendrycks_test", "abstract_algebra", split="test")


def eval_mmlu(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    ds = load_mmlu_split()
    correct = 0
    total = 0
    for row in ds:
        if "question" not in row or "choices" not in row:
            continue
        q = row["question"]
        choices = row["choices"]
        answer = row.get("answer")
        if answer is None:
            # hendrycks_test style could use "answer" letter index not guaranteed.
            continue
        scores = []
        for c in choices:
            _raw, norm = choice_scores(model, q + "\nAnswer:", " " + c, max_len=max_len, device=device)
            scores.append(norm)
        pred = int(torch.tensor(scores).argmax().item())
        try:
            gt = int(answer)
        except Exception:
            continue
        correct += int(pred == gt)
        total += 1
        if total >= limit:
            break
    return {"accuracy": correct / max(total, 1), "n": total}


def eval_piqa(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    ds = safe_load_dataset("ybisk/piqa", split="validation")
    if ds is None:
        return {"accuracy": float("nan"), "n": 0, "skipped": "dataset_unavailable"}
    correct = 0
    total = 0
    for row in ds:
        question = row["goal"]
        choices = [row["sol1"], row["sol2"]]
        label = int(row["label"])
        scores = []
        for c in choices:
            _raw, norm = choice_scores(model, question + "\nAnswer:", " " + c, max_len=max_len, device=device)
            scores.append(norm)
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == label)
        total += 1
        if total >= limit:
            break
    return {"accuracy": correct / max(total, 1), "n": total}


def eval_winogrande(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    ds = safe_load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    if ds is None:
        return {"accuracy": float("nan"), "n": 0, "skipped": "dataset_unavailable"}
    correct = 0
    total = 0
    for row in ds:
        sentence = row["sentence"]
        opt1 = row["option1"]
        opt2 = row["option2"]
        answer = int(row["answer"]) - 1
        scores = []
        for opt in (opt1, opt2):
            completed = sentence.replace("_", opt)
            _raw, norm = choice_scores(model, "", completed, max_len=max_len, device=device)
            scores.append(norm)
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == answer)
        total += 1
        if total >= limit:
            break
    return {"accuracy": correct / max(total, 1), "n": total}


def eval_openbookqa(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    ds = safe_load_dataset("allenai/openbookqa", "main", split="validation")
    if ds is None:
        return {"accuracy": float("nan"), "n": 0, "skipped": "dataset_unavailable"}
    correct = 0
    total = 0
    for row in ds:
        stem = row["question_stem"]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer = row["answerKey"]
        scores = []
        for txt in texts:
            _raw, norm = choice_scores(model, stem + "\nAnswer:", " " + txt, max_len=max_len, device=device)
            scores.append(norm)
        pred = labels[int(torch.tensor(scores).argmax().item())]
        correct += int(pred == answer)
        total += 1
        if total >= limit:
            break
    return {"accuracy": correct / max(total, 1), "n": total}


def run_suite(model: TransformerMolecule, *, limit: int, max_len: int, device: torch.device) -> dict:
    return {
        "arc_challenge": eval_arc(model, limit=limit, max_len=max_len, device=device),
        "arc_easy": eval_arc_easy(model, limit=limit, max_len=max_len, device=device),
        "hellaswag": eval_hellaswag(model, limit=limit, max_len=max_len, device=device),
        "mmlu": eval_mmlu(model, limit=limit, max_len=max_len, device=device),
        "piqa": eval_piqa(model, limit=limit, max_len=max_len, device=device),
        "winogrande": eval_winogrande(model, limit=limit, max_len=max_len, device=device),
        "openbookqa": eval_openbookqa(model, limit=limit, max_len=max_len, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--baseline-ckpt", type=Path, required=True)
    parser.add_argument("--engram-ckpt", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("artifacts/paper_tasks_compare.json"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    baseline = load_model(args.baseline_ckpt, device)
    engram = load_model(args.engram_ckpt, device)

    base_res = run_suite(baseline, limit=args.limit, max_len=args.max_len, device=device)
    eng_res = run_suite(engram, limit=args.limit, max_len=args.max_len, device=device)

    delta = {}
    for k in base_res:
        if base_res[k].get("n", 0) == 0 or eng_res[k].get("n", 0) == 0:
            delta[k] = {"accuracy_delta": float("nan"), "n": 0}
            continue
        delta[k] = {
            "accuracy_delta": eng_res[k]["accuracy"] - base_res[k]["accuracy"],
            "n": min(base_res[k]["n"], eng_res[k]["n"]),
        }

    result = {
        "limit": args.limit,
        "max_len": args.max_len,
        "baseline_ckpt": str(args.baseline_ckpt),
        "engram_ckpt": str(args.engram_ckpt),
        "baseline": base_res,
        "engram": eng_res,
        "delta": delta,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
