import pytest
import torch

from agents.loop_transformer import LoopTransformerCore


def test_shared_loop_returns_each_iteration_state():
    core = LoopTransformerCore(d_model=16, heads=4, max_iterations=3)
    output, states = core(torch.randn(2, 5, 16), iterations=2)
    assert output.shape == (2, 5, 16)
    assert len(states) == 2
    assert not torch.equal(states[0], states[1])


def test_loop_rejects_invalid_depth():
    with pytest.raises(ValueError):
        LoopTransformerCore(d_model=16, heads=4, max_iterations=2)(torch.randn(1, 2, 16), iterations=3)
