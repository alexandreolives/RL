from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from agents.multimodal_baseline import MultimodalActorCritic
from agents.multimodal_env import MultimodalMemoryEnv


def tensor_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.from_numpy(obs["image"]).unsqueeze(0),
        "bytes_view": torch.from_numpy(obs["bytes"]).unsqueeze(0),
        "symbolic": torch.from_numpy(obs["symbolic"]).unsqueeze(0),
        "phase": torch.from_numpy(obs["phase"]).unsqueeze(0),
    }


def run(seed: int, episodes: int, horizon: int) -> dict[str, float | int]:
    torch.manual_seed(seed)
    env = MultimodalMemoryEnv(horizon=horizon, seed=seed)
    model = MultimodalActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rewards: list[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        log_probs, values, episode_rewards = [], [], []
        terminated = False
        while not terminated:
            logits, value, _ = model(**tensor_obs(obs))
            dist = Categorical(logits=logits)
            action = dist.sample()
            obs, reward, terminated, _, _ = env.step(int(action.item()))
            log_probs.append(dist.log_prob(action).squeeze())
            values.append(value.squeeze())
            episode_rewards.append(float(reward))
        returns = torch.tensor([episode_rewards[-1]] * len(values), dtype=torch.float32)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        advantage = returns - values_t.detach()
        loss = -(log_probs_t * advantage).mean() + 0.5 * (returns - values_t).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(episode_rewards[-1])
    return {"seed": seed, "episodes": episodes, "horizon": horizon, "mean_reward_last20": float(np.mean(rewards[-20:]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.episodes, args.horizon)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
