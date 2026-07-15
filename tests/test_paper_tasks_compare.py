from __future__ import annotations

from pathlib import Path

import torch

from eval.transformer.paper_tasks_compare import load_model
from eval.transformer.train_long_context_compare import build_train_model


def test_load_model_accepts_only_dormant_historical_lightning_keys(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = build_train_model(
        "baseline",
        device,
        model_size="tiny",
        input_mode="byte",
        attention_backend="eager",
    ).eval()
    historical_state = {
        key: value
        for key, value in model.state_dict().items()
        if ".attn.lightning_indexer." not in key
        and ".attn.lightning_q." not in key
        and ".attn.lightning_k." not in key
    }
    model.config.attention.backend = "flash"
    checkpoint = tmp_path / "historical.pt"
    torch.save({"config": model.config, "state_dict": historical_state}, checkpoint)

    loaded = load_model(checkpoint, device)
    assert loaded.config.attention.backend in {"auto", "flash"}
    ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        expected = model(byte_ids=ids)
        actual = loaded(byte_ids=ids)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
