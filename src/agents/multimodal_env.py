from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultimodalMemoryEnv(gym.Env):
    """Tiny cue-and-recall environment with image, byte and symbolic views.

    A binary cue is shown at reset and hidden after the first observation. The
    agent must remember it and emit the matching action at the final step.
    Every modality encodes the same cue, making cross-modal transfer measurable
    without requiring an external dataset.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, horizon: int = 8, image_size: int = 8, seed: int | None = None, reveal_each_step: bool = False) -> None:
        super().__init__()
        if horizon < 2 or image_size < 2:
            raise ValueError("horizon must be >= 2 and image_size must be >= 2")
        self.horizon = horizon
        self.image_size = image_size
        self.reveal_each_step = reveal_each_step
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0.0, 1.0, (1, image_size, image_size), dtype=np.float32),
                "bytes": spaces.Box(0, 255, (8,), dtype=np.int64),
                "symbolic": spaces.MultiBinary(4),
                "phase": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._cue = 0
        self._step = 0

    def _observation(self) -> dict[str, np.ndarray]:
        visible = self.reveal_each_step or self._step == 0
        cue = self._cue if visible else 0
        image = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        image[:, : max(1, self.image_size // 4), : max(1, self.image_size // 4)] = cue
        bytes_view = np.zeros((8,), dtype=np.int64)
        bytes_view[0] = cue
        symbolic = np.zeros((4,), dtype=np.int8)
        symbolic[0] = cue
        return {
            "image": image,
            "bytes": bytes_view,
            "symbolic": symbolic,
            "phase": np.asarray([self._step / (self.horizon - 1)], dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cue = int(self._rng.integers(0, 2))
        self._step = 0
        return self._observation(), {"cue": self._cue}

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        self._step += 1
        terminated = self._step >= self.horizon - 1
        reward = float(terminated and action == self._cue)
        return self._observation(), reward, terminated, False, {"cue": self._cue}


class MultimodalParityEnv(gym.Env):
    """Longer memory control: predict parity of a streamed binary sequence."""

    def __init__(self, *, sequence_length: int = 16, image_size: int = 8, seed: int | None = None) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0.0, 1.0, (1, image_size, image_size), dtype=np.float32),
            "bytes": spaces.Box(0, 255, (8,), dtype=np.int64),
            "symbolic": spaces.MultiBinary(4),
            "phase": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._sequence = np.zeros(sequence_length, dtype=np.int8)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sequence = self._rng.integers(0, 2, size=self.sequence_length, dtype=np.int8)
        self._step = 0
        return self._obs(), {"parity": int(self._sequence.sum() % 2)}

    def _obs(self):
        bit = int(self._sequence[self._step])
        image = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        image[:, :2, :2] = bit
        bytes_view = np.zeros(8, dtype=np.int64); bytes_view[0] = bit
        symbolic = np.zeros(4, dtype=np.int8); symbolic[0] = bit
        return {"image": image, "bytes": bytes_view, "symbolic": symbolic, "phase": np.asarray([self._step / (self.sequence_length - 1)], dtype=np.float32)}

    def step(self, action: int):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action: {action}")
        self._step += 1
        terminated = self._step >= self.sequence_length - 1
        target = int(self._sequence.sum() % 2)
        reward = float(terminated and int(action) == target)
        return self._obs(), reward, terminated, False, {"parity": target}
