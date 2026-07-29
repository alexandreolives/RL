import numpy as np
import torch

from agents.multimodal_baseline import MultimodalActorCritic, RecurrentMultimodalActorCritic
from agents.multimodal_env import MultimodalMemoryEnv


def test_environment_is_seed_reproducible_and_has_memory_horizon():
    env_a, env_b = MultimodalMemoryEnv(seed=7), MultimodalMemoryEnv(seed=7)
    obs_a, info_a = env_a.reset(seed=7)
    obs_b, info_b = env_b.reset(seed=7)
    assert info_a["cue"] == info_b["cue"]
    assert np.array_equal(obs_a["image"], obs_b["image"])
    for _ in range(7):
        obs_a, reward, terminated, _, _ = env_a.step(0)
    assert terminated
    assert reward in (0.0, 1.0)


def test_multimodal_baseline_fuses_all_views():
    env = MultimodalMemoryEnv(seed=0)
    obs, _ = env.reset(seed=0)
    model = MultimodalActorCritic()
    batch = {
        "image": torch.from_numpy(obs["image"]).unsqueeze(0),
        "bytes_view": torch.from_numpy(obs["bytes"]).unsqueeze(0),
        "symbolic": torch.from_numpy(obs["symbolic"]).unsqueeze(0),
        "phase": torch.from_numpy(obs["phase"]).unsqueeze(0),
    }
    logits, value, latent = model(**batch)
    assert logits.shape == (1, 2)
    assert value.shape == (1,)
    assert latent.shape == (1, 64)


def test_recurrent_baseline_keeps_state_across_steps():
    env = MultimodalMemoryEnv(seed=2)
    obs, _ = env.reset(seed=2)
    model = RecurrentMultimodalActorCritic()
    state = None
    for _ in range(2):
        batch = {
            "image": torch.from_numpy(obs["image"]).unsqueeze(0),
            "bytes_view": torch.from_numpy(obs["bytes"]).unsqueeze(0),
            "symbolic": torch.from_numpy(obs["symbolic"]).unsqueeze(0),
            "phase": torch.from_numpy(obs["phase"]).unsqueeze(0),
            "state": state,
        }
        _, _, state = model(**batch)
        obs, _, _, _, _ = env.step(0)
    assert state.shape == (1, 64)
