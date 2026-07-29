import torch

from agents.memory import EngramLatentAdapter


def test_engram_latent_adapter_is_shape_preserving():
    adapter = EngramLatentAdapter(d_model=16, slots=31, heads=2)
    latent = torch.randn(2, 5, 16)
    output = adapter(latent, torch.randint(0, 32, (2, 5)))
    assert output.shape == latent.shape
