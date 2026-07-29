import torch

from agents.spectral_loop import FourierMixer, SpectralAttentionLoop


def test_fourier_mixer_preserves_shape_and_gradients():
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = FourierMixer()(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None


def test_spectral_loop_has_attention_fallback():
    x = torch.randn(2, 5, 16)
    y, used, iterations = SpectralAttentionLoop(16, heads=4)(x, force_attention=True)
    assert y.shape == x.shape
    assert used and iterations == 2
