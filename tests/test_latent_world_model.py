import pytest
import torch

from agents.latent_world_model import ActionConditionedLatentPredictor, latent_prediction_loss, variance_covariance_regularizer


def test_action_conditioned_predictor_and_loss():
    predictor = ActionConditionedLatentPredictor(latent_dim=16)
    predicted = predictor(torch.randn(3, 16), torch.tensor([0, 1, 0]))
    assert predicted.shape == (3, 16)
    assert latent_prediction_loss(predicted, torch.zeros_like(predicted)).ndim == 0


def test_predictor_rejects_bad_batch_shape():
    with pytest.raises(ValueError):
        ActionConditionedLatentPredictor()(torch.randn(2, 64), torch.zeros(1, dtype=torch.long))


def test_variance_covariance_regularizer_is_finite():
    loss = variance_covariance_regularizer(torch.randn(8, 16))
    assert torch.isfinite(loss)
