"""Controlled environments and agent baselines for the multimodal roadmap."""

from .multimodal_env import MultimodalMemoryEnv
from .multimodal_baseline import MultimodalActorCritic, RecurrentMultimodalActorCritic
from .latent_world_model import ActionConditionedLatentPredictor, latent_prediction_loss
from .loop_transformer import LoopTransformerCore

__all__ = [
    "MultimodalMemoryEnv",
    "MultimodalActorCritic",
    "RecurrentMultimodalActorCritic",
    "ActionConditionedLatentPredictor",
    "latent_prediction_loss",
    "LoopTransformerCore",
]
