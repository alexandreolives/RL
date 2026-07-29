from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from agents.loop_transformer import StatefulLoopCore
from agents.multimodal_baseline import MultimodalActorCritic, RecurrentMultimodalActorCritic
from agents.multimodal_env import MultimodalMemoryEnv, MultimodalParityEnv


def obs_tensor(obs):
    return {
        "image": torch.from_numpy(obs["image"]).unsqueeze(0),
        "bytes_view": torch.from_numpy(obs["bytes"]).unsqueeze(0),
        "symbolic": torch.from_numpy(obs["symbolic"]).unsqueeze(0),
        "phase": torch.from_numpy(obs["phase"]).unsqueeze(0),
    }


class LoopMemoryActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MultimodalActorCritic()
        self.loop = StatefulLoopCore()
        self.policy = nn.Linear(64, 2)
        self.value = nn.Linear(64, 1)

    def forward(self, obs, state=None):
        latent = self.encoder.encode(**obs)
        latent, state = self.loop(latent, state)
        return self.policy(latent), self.value(latent).squeeze(-1), state


def train(name: str, episodes: int, seed: int, task: str, sequence_length: int) -> float:
    torch.manual_seed(seed)
    model = RecurrentMultimodalActorCritic() if name == "gru" else LoopMemoryActor()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rewards = []
    for _ in range(episodes):
        env = (MultimodalMemoryEnv(horizon=8, seed=seed, reveal_each_step=False)
               if task == "cue" else MultimodalParityEnv(sequence_length=sequence_length, seed=seed))
        obs, _ = env.reset()
        state = None
        logs, values = [], []
        reward = 0.0
        total_steps = env.horizon - 1 if task == "cue" else env.sequence_length - 1
        for _ in range(total_steps):
            if name == "gru":
                logits, value, state = model(**obs_tensor(obs), state=state)
            else:
                logits, value, state = model(obs_tensor(obs), state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            obs, reward, done, _, _ = env.step(int(action.item()))
            logs.append(dist.log_prob(action).squeeze())
            values.append(value.squeeze())
            if done:
                break
        returns = torch.full((len(values),), float(reward))
        values_t = torch.stack(values)
        loss = -(torch.stack(logs) * (returns - values_t.detach())).mean() + 0.5 * (returns - values_t).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        rewards.append(float(reward))
    return float(np.mean(rewards[-20:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=["cue", "parity"], default="cue")
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args()
    result = {name: train(name, args.episodes, args.seed, args.task, args.sequence_length) for name in ("gru", "loop")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
