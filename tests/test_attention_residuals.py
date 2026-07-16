from __future__ import annotations

import torch

from eval.transformer.train_long_context_compare import build_train_model
from models.atoms.config import TransformerConfig
from models.atoms.residual import FullAttentionResidual
from models.molecules.transformer import TransformerMolecule


def test_zero_initialized_query_is_uniform_average() -> None:
    mixer = FullAttentionResidual(4)
    sources = [
        torch.full((2, 3, 4), 2.0),
        torch.full((2, 3, 4), 5.0),
        torch.full((2, 3, 4), 8.0),
    ]

    actual = mixer(sources)

    torch.testing.assert_close(actual, torch.full_like(actual, 5.0))
    torch.testing.assert_close(mixer.query, torch.zeros_like(mixer.query))


def test_attnres_variants_build_with_paired_backbones() -> None:
    device = torch.device("cpu")
    torch.manual_seed(7)
    baseline = build_train_model(
        "baseline", device, model_size="tiny", input_mode="symbolic", attention_backend="eager"
    )
    torch.manual_seed(7)
    attnres = build_train_model(
        "attnres", device, model_size="tiny", input_mode="symbolic", attention_backend="eager"
    )
    torch.manual_seed(7)
    engram = build_train_model(
        "engram_noconv", device, model_size="tiny", input_mode="symbolic", attention_backend="eager"
    )
    torch.manual_seed(7)
    combined = build_train_model(
        "engram_noconv_attnres",
        device,
        model_size="tiny",
        input_mode="symbolic",
        attention_backend="eager",
    )

    assert not baseline.config.use_attnres
    assert attnres.config.use_attnres and not attnres.config.engram.enabled
    assert not engram.config.use_attnres and engram.config.engram.enabled
    assert combined.config.use_attnres and combined.config.engram.enabled
    assert baseline.config.d_model == attnres.config.d_model == engram.config.d_model == combined.config.d_model
    assert baseline.config.depth == attnres.config.depth == engram.config.depth == combined.config.depth

    baseline_state = baseline.state_dict()
    attnres_state = attnres.state_dict()
    for name in baseline_state.keys() & attnres_state.keys():
        torch.testing.assert_close(baseline_state[name], attnres_state[name])

    engram_state = engram.state_dict()
    combined_state = combined.state_dict()
    for name in engram_state.keys() & combined_state.keys():
        torch.testing.assert_close(engram_state[name], combined_state[name])


def test_engram_attnres_forward_backward_reaches_query_and_memory() -> None:
    torch.manual_seed(0)
    model = build_train_model(
        "engram_noconv_attnres",
        torch.device("cpu"),
        model_size="tiny",
        input_mode="symbolic",
        attention_backend="eager",
    )
    token_ids = torch.randint(0, 260, (2, 12))

    logits = model(token_ids=token_ids, use_cache=False)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 260), token_ids[:, 1:].reshape(-1)
    )
    loss.backward()

    query_grad = model.final_attn_res.query.grad
    memory_grad = model.blocks[1].engram.order_embeddings["2"].weight.grad
    assert query_grad is not None and torch.isfinite(query_grad).all()
    assert memory_grad is not None and torch.isfinite(memory_grad).all()
    assert query_grad.abs().sum() > 0
    assert memory_grad.abs().sum() > 0


def test_attnres_full_forward_matches_incremental_cache() -> None:
    torch.manual_seed(3)
    model = build_train_model(
        "attnres",
        torch.device("cpu"),
        model_size="tiny",
        input_mode="symbolic",
        attention_backend="eager",
    ).eval()
    token_ids = torch.randint(0, 260, (1, 8))

    with torch.no_grad():
        full = model(token_ids=token_ids, use_cache=False)
        cache = None
        pieces = []
        for position in range(token_ids.size(1)):
            logits, cache = model(
                token_ids=token_ids[:, position : position + 1],
                past_key_values=cache,
                return_cache=True,
            )
            pieces.append(logits)

    incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, incremental, atol=2e-5, rtol=2e-5)
    assert cache.get_seq_length() == token_ids.size(1)


def test_attnres_rejects_mhc_combination() -> None:
    config = TransformerConfig(use_attnres=True, use_mhc=True)
    try:
        TransformerMolecule(config)
    except ValueError as exc:
        assert "mHC" in str(exc)
    else:
        raise AssertionError("AttnRes + mHC should require a separately defined composition")


def test_fused_v1_keeps_engram_inside_attention_source() -> None:
    torch.manual_seed(11)
    model = build_train_model(
        "engram_noconv_attnres_v1",
        torch.device("cpu"),
        model_size="tiny",
        input_mode="symbolic",
        attention_backend="eager",
    )
    captured = {}

    def capture_sources(_module, args) -> None:
        captured["count"] = len(args[0])

    handle = model.final_attn_res.register_forward_pre_hook(capture_sources)
    try:
        token_ids = torch.randint(0, 260, (2, 12))
        logits = model(token_ids=token_ids, use_cache=False)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 260), token_ids[:, 1:].reshape(-1)
        )
        loss.backward()
    finally:
        handle.remove()

    # Embedding + two standard sublayers for each of four blocks. Engram is
    # fused into the current attention source and does not add two extra slots.
    assert captured["count"] == 9
    memory_grad = model.blocks[1].engram.order_embeddings["2"].weight.grad
    assert memory_grad is not None and memory_grad.abs().sum() > 0
