import pytest
import torch

from agents.loop_transformer import AdaptiveLoopCore, LoopTransformerCore, StatefulLoopCore


def test_shared_loop_returns_each_iteration_state():
    core = LoopTransformerCore(d_model=16, heads=4, max_iterations=3)
    output, states = core(torch.randn(2, 5, 16), iterations=2)
    assert output.shape == (2, 5, 16)
    assert len(states) == 2
    assert not torch.equal(states[0], states[1])


def test_loop_rejects_invalid_depth():
    with pytest.raises(ValueError):
        LoopTransformerCore(d_model=16, heads=4, max_iterations=2)(torch.randn(1, 2, 16), iterations=3)


def test_stateful_loop_carries_previous_latent():
    core = StatefulLoopCore(d_model=16, heads=4)
    current = torch.randn(2, 16)
    first, state = core(current)
    second, next_state = core(current * 0, state)
    assert first.shape == second.shape == (2, 16)
    assert not torch.equal(state, next_state)


def test_adaptive_loop_respects_hard_budget():
    core = AdaptiveLoopCore(d_model=16, heads=4, max_iterations=3)
    output, used = core(torch.randn(2, 5, 16))
    assert output.shape == (2, 5, 16)
    assert 1 <= int(used) <= 3
