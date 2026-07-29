import pytest
import torch

from models.atoms.contracts import ModularAgentConfig, RouterDecision


def test_baseline_config_disables_optional_blocks():
    config = ModularAgentConfig.baseline()
    config.validate()
    assert not any((config.use_engram, config.use_jepa, config.use_loop, config.use_router))
    assert config.to_dict()["use_ternary_qat"] is False


def test_router_decision_validates_shapes():
    decision = RouterDecision(
        expert_indices=torch.zeros(2, 5, 2, dtype=torch.long),
        expert_weights=torch.ones(2, 5, 2),
    )
    decision.validate(batch_size=2, sequence_length=5, top_k=2)


def test_router_decision_rejects_mismatched_weights():
    decision = RouterDecision(
        expert_indices=torch.zeros(2, 5, 2, dtype=torch.long),
        expert_weights=torch.ones(2, 5, 1),
    )
    with pytest.raises(ValueError, match="identical shapes"):
        decision.validate(batch_size=2, sequence_length=5, top_k=2)
