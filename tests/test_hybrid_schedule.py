import torch

from models.atoms.hybrid import ConfigurableHybridCore, HybridStage
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
