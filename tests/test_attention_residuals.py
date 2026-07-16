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


def test_mhc_attnres_streams_build_and_backward() -> None:
    torch.manual_seed(19)
    model = build_train_model("mhc_attnres", torch.device("cpu"), model_size="tiny", input_mode="symbolic", attention_backend="eager")
    assert model.config.use_mhc_streams and model.config.hc_mult == 4
    token_ids = torch.randint(0, 260, (2, 12))
    logits = model(token_ids=token_ids, use_cache=False)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 260), token_ids[:, 1:].reshape(-1))
    loss.backward()
    assert model.final_attn_res.query.grad is not None


def test_v6_engram_uses_native_mhc_four_stream_backbone() -> None:
    model = build_train_model("v6_engram", torch.device("cpu"), model_size="tiny", input_mode="symbolic", attention_backend="eager")
    assert model.config.implementation == "native"
    assert model.config.use_mhc_streams and model.config.hc_mult == 4
    assert model.config.engram.enabled
    assert any(block.engram is not None for block in model.blocks)


def test_v6_engram_attnres_forward_backward() -> None:
    torch.manual_seed(23)
    model = build_train_model("v6_engram_attnres", torch.device("cpu"), model_size="tiny", input_mode="symbolic", attention_backend="eager")
    assert model.config.use_attnres and model.config.use_mhc_streams
    token_ids = torch.randint(0, 260, (2, 12))
    logits = model(token_ids=token_ids, use_cache=False)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 260), token_ids[:, 1:].reshape(-1))
    loss.backward()
    assert model.final_attn_res.query.grad is not None
    assert model.blocks[1].engram.order_embeddings["2"].weight.grad is not None


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


def test_v2_routes_engram_through_separate_gated_bypass() -> None:
    torch.manual_seed(13)
    model = build_train_model(
        "engram_noconv_attnres_v2",
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

    assert model.config.attnres_engram_mode == "bypass"
    assert captured["count"] == 9
    for layer_idx in model.config.engram.insert_layers:
        block = model.blocks[layer_idx]
        torch.testing.assert_close(block.engram_bypass_gate, torch.ones_like(block.engram_bypass_gate))
        assert block.engram_bypass_gate.grad is not None
        assert block.engram_bypass_gate.grad.abs().sum() > 0
        memory_grad = block.engram.order_embeddings["2"].weight.grad
        assert memory_grad is not None and memory_grad.abs().sum() > 0


def test_v3_uses_bounded_low_gain_bypass() -> None:
    torch.manual_seed(17)
    model = build_train_model("engram_noconv_attnres_v3", torch.device("cpu"), model_size="tiny", input_mode="symbolic", attention_backend="eager")
    block = model.blocks[1]
    assert model.config.attnres_engram_mode == "bounded_bypass"
    torch.testing.assert_close(torch.sigmoid(block.engram_bypass_gate_logit), torch.full((192,), 0.1), atol=1e-6, rtol=1e-6)
    token_ids = torch.randint(0, 260, (2, 12))
    logits = model(token_ids=token_ids, use_cache=False)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 260), token_ids[:, 1:].reshape(-1))
    loss.backward()
    assert block.engram_bypass_gate_logit.grad is not None
