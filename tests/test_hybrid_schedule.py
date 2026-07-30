import torch

from models.atoms.hybrid import ConfigurableHybridCore, AdaptiveHybridCore, AnchoredKDALoopMacroblock, CausalGatedFourierBlock, HybridStage
from agents.modular_agent import ModularMultimodalAgent


def test_configurable_schedule_and_trace():
    torch.manual_seed(0)
    core = ConfigurableHybridCore(16, stages=[HybridStage("fourier"), HybridStage("kda"), HybridStage("attention"), HybridStage("loop")])
    x = torch.randn(2, 5, 16)
    y, trace = core(x, iterations=2, return_trace=True)
    assert y.shape == x.shape and len(trace) == 8
    assert [row["kind"] for row in trace[:4]] == ["fourier", "kda", "attention", "loop"]


def test_agent_accepts_hybrid_schedule():
    model = ModularMultimodalAgent(hybrid_stages=["fourier", HybridStage("kda", 2), "attention"], hybrid_iterations=2)
    args = {"image": torch.rand(2, 1, 8, 8), "bytes_view": torch.randint(0, 256, (2, 8)), "symbolic": torch.rand(2, 4), "phase": torch.rand(2, 1)}
    logits, values, latent, _ = model(**args)
    assert logits.shape == (2, 2) and values.shape == (2,) and torch.isfinite(latent).all()


def test_adaptive_schedule_records_selected_path():
    core = AdaptiveHybridCore(16, fast_stages=["fourier"], full_stages=["fourier", "kda", "attention"])
    y, trace = core(torch.randn(2, 4, 16), return_trace=True)
    assert y.shape == (2, 4, 16)
    assert trace and all("adaptive_full" in item and "uncertainty" in item for item in trace)


def test_kda_state_can_be_carried_between_loop_iterations():
    core = ConfigurableHybridCore(16, stages=["kda", "loop"], carry_kda_state=True)
    y = core(torch.randn(2, 4, 16), iterations=3)
    assert y.shape == (2, 4, 16) and torch.isfinite(y).all()


def test_causal_fourier_does_not_see_future_tokens():
    block = CausalGatedFourierBlock(8, kernel_size=5)
    x = torch.randn(1, 12, 8)
    y = block(x)
    changed = x.clone(); changed[:, 8:] += 100.0
    y_changed = block(changed)
    torch.testing.assert_close(y[:, :8], y_changed[:, :8], atol=1e-5, rtol=1e-5)


def test_anchored_kda_loop_macroblock():
    block = AnchoredKDALoopMacroblock(16, loop_repeats=2, post_kda=True)
    y, trace = block(torch.randn(2, 6, 16))
    assert y.shape == (2, 6, 16)
    assert trace.count("loop_kda") == 2 and "mla_anchor" in trace
