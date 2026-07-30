import torch

from models.atoms.kda import KDA, HybridKDA
from models.atoms.latent_moe import LatentMoE


def test_kda_streaming_shapes_and_state():
    torch.manual_seed(0)
    m = KDA(16, heads=4)
    x = torch.randn(2, 7, 16)
    y, state = m(x)
    assert y.shape == x.shape and state.value.shape == (2, 4, 4, 4)
    y2, _ = m(x[:, :3], m.init_state(x))
    assert torch.isfinite(y).all() and torch.isfinite(y2).all()
    y_a, st = m(x[:, :3])
    y_b, _ = m(x[:, 3:], st)
    assert torch.allclose(torch.cat((y_a, y_b), 1), y, atol=1e-5, rtol=1e-5)


def test_hybrid_and_latent_moe():
    x = torch.randn(2, 5, 16)
    assert HybridKDA(16)(x).shape == x.shape
    assert len(HybridKDA(16, kda_blocks=3, num_cycles=2).kda) == 6
    moe = LatentMoE(16, 8, num_experts=4, top_k=2)
    y = moe(x)
    assert y.shape == x.shape and torch.isfinite(moe.last_aux_loss)
    shared = LatentMoE(16, 8, num_experts=4, top_k=2, shared_experts=2)
    assert shared(x).shape == x.shape
