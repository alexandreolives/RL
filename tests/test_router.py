import torch

from models.atoms.router import TopKRouter, build_single_expert_router


def test_topk_router_returns_normalized_dispatch():
    router = TopKRouter(d_model=8, num_experts=4, top_k=2)
    decision = router(torch.randn(2, 3, 8))
    assert decision.expert_indices.shape == (2, 3, 2)
    assert torch.allclose(decision.expert_weights.sum(dim=-1), torch.ones(2, 3))


def test_single_expert_router_is_valid_control():
    decision = build_single_expert_router(8)(torch.randn(1, 4, 8))
    assert torch.equal(decision.expert_indices, torch.zeros(1, 4, 1, dtype=torch.long))
    assert torch.allclose(decision.expert_weights, torch.ones(1, 4, 1))
