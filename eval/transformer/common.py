from __future__ import annotations

import time
from statistics import mean

import torch

from models.example import build_variant


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_forward(model, *, byte_ids, modality_ids, steps: int = 3):
    times = []
    device = byte_ids.device
    with torch.no_grad():
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            logits = model(byte_ids=byte_ids, modality_ids=modality_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
    return logits, {
        "mean_sec": mean(times),
        "tok_per_s": (byte_ids.numel() * steps) / sum(times),
    }


def make_batch(batch: int, seq_len: int, patch_size: int, *, device: torch.device | None = None):
    byte_ids = torch.randint(0, 260, (batch, seq_len), device=device)
    patch_len = (seq_len + patch_size - 1) // patch_size
    modality_ids = torch.zeros((batch, patch_len), dtype=torch.long, device=device)
    return byte_ids, modality_ids


def make_model(name: str, *, device: torch.device | None = None, attention_backend: str = "auto"):
    model = build_variant(name, attention_backend=attention_backend).to(device).eval()
    return model
