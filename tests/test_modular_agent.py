import torch

from agents.modular_agent import ModularMultimodalAgent


def test_modular_agent_composes_optional_m1_blocks():
    model = ModularMultimodalAgent(use_engram=True, use_jepa=True, use_loop=True, use_spectral=False)
    args = {
        "image": torch.rand(2, 1, 8, 8),
        "bytes_view": torch.randint(0, 256, (2, 8)),
        "symbolic": torch.rand(2, 4),
        "phase": torch.rand(2, 1),
        "action": torch.tensor([0, 1]),
    }
    logits, values, latent, predicted = model(**args)
    assert logits.shape == (2, 2)
    assert values.shape == (2,)
    assert latent.shape == predicted.shape == (2, 64)
