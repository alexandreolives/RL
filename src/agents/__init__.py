"""Controlled environments and agent baselines for the multimodal roadmap."""

from .multimodal_env import MultimodalMemoryEnv
from .multimodal_baseline import MultimodalActorCritic, RecurrentMultimodalActorCritic
from .latent_world_model import ActionConditionedLatentPredictor, latent_prediction_loss, variance_covariance_regularizer
from .loop_transformer import AdaptiveLoopCore, LoopTransformerCore, StatefulLoopCore
from .spectral_loop import FourierMixer, SpectralAttentionLoop
from .memory import EngramLatentAdapter
from .modular_agent import ModularMultimodalAgent

__all__ = [
    "MultimodalMemoryEnv",
    "MultimodalActorCritic",
    "RecurrentMultimodalActorCritic",
    "ActionConditionedLatentPredictor",
    "latent_prediction_loss",
    "variance_covariance_regularizer",
    "LoopTransformerCore",
    "StatefulLoopCore",
    "AdaptiveLoopCore",
    "FourierMixer",
    "SpectralAttentionLoop",
    "EngramLatentAdapter",
    "ModularMultimodalAgent",
]
