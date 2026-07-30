import torch

from models.atoms.quantization import MXFP4FakeQuant, MXFP8FakeQuant


def test_mxfp_fake_quant_is_finite_and_has_ste_gradient():
    x = torch.randn(2, 32, requires_grad=True)
    y = MXFP4FakeQuant()(x)
    y.sum().backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mxfp8_preserves_more_levels():
    x = torch.linspace(-1, 1, 32)
    y4, y8 = MXFP4FakeQuant()(x), MXFP8FakeQuant()(x)
    assert torch.unique(y8).numel() >= torch.unique(y4).numel()
