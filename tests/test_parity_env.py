import numpy as np

from agents.multimodal_env import MultimodalParityEnv


def test_parity_environment_is_reproducible_and_delayed():
    a, b = MultimodalParityEnv(sequence_length=8, seed=3), MultimodalParityEnv(sequence_length=8, seed=3)
    obs_a, info_a = a.reset(seed=3); obs_b, info_b = b.reset(seed=3)
    assert info_a == info_b
    assert np.array_equal(obs_a["image"], obs_b["image"])
    for _ in range(7):
        obs_a, reward, done, _, _ = a.step(0)
    assert done and reward in (0.0, 1.0)
