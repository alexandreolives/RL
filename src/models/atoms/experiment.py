"""Minimal run metadata helpers for auditable modular experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from torch import nn


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


@dataclass(frozen=True)
class RunRecord:
    seed: int
    variant: str
    parameters: int
    trainable_parameters: int
    active_flops: float | None = None
    wall_time_seconds: float | None = None
    peak_memory_bytes: int | None = None
    energy_joules: float | None = None

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
