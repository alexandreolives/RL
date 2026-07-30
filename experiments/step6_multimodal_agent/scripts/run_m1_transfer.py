from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from agents.modular_agent import ModularMultimodalAgent
from agents.multimodal_env import MultimodalMemoryEnv, MultimodalParityEnv


VARIANTS = {
    "baseline": {},
    "engram": {"use_engram": True},
    "jepa": {"use_jepa": True},
    "loop": {"use_loop": True},
    "spectral": {"use_spectral": True},
    "full_m1": {"use_engram": True, "use_jepa": True, "use_loop": True, "use_spectral": True},
    "kda_loop_carry": {"use_loop": False, "hybrid_stages": ["kda", "loop"], "hybrid_iterations": 2, "hybrid_carry_kda_state": True},
    "anchored_b": {"use_loop": False, "use_anchored_macroblock": True, "macroblock_loop_repeats": 3},
    "anchored_c": {"use_loop": False, "use_anchored_macroblock": True, "macroblock_loop_repeats": 2, "macroblock_post_kda": True},
}


def batch(obs):
    return {"image": torch.from_numpy(obs["image"]).unsqueeze(0), "bytes_view": torch.from_numpy(obs["bytes"]).unsqueeze(0), "symbolic": torch.from_numpy(obs["symbolic"]).unsqueeze(0), "phase": torch.from_numpy(obs["phase"]).unsqueeze(0)}


def train(model, seed: int, episodes: int, env_kind: str):
    env = MultimodalMemoryEnv(seed=seed, reveal_each_step=True) if env_kind == "memory" else MultimodalParityEnv(seed=seed, sequence_length=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(episodes):
        obs, _ = env.reset()
        logs, values, reward = [], [], 0.0
        for _ in range((env.horizon if hasattr(env, "horizon") else env.sequence_length) - 1):
            logits, value, _, _ = model(**batch(obs))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            obs, reward, done, _, _ = env.step(int(action.item()))
            logs.append(dist.log_prob(action).squeeze()); values.append(value.squeeze())
            if done: break
        returns = torch.full((len(values),), float(reward)); values = torch.stack(values)
        loss = -(torch.stack(logs) * (returns - values.detach())).mean() + 0.5 * (returns - values).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()


def evaluate(model, seed: int, mask: str | None, episodes: int, env_kind: str) -> float:
    env = MultimodalMemoryEnv(seed=seed, reveal_each_step=True) if env_kind == "memory" else MultimodalParityEnv(seed=seed, sequence_length=16)
    scores = []
    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset(); reward = 0.0
            for _ in range((env.horizon if hasattr(env, "horizon") else env.sequence_length) - 1):
                item = batch(obs)
                if mask:
                    item[mask].zero_()
                logits, _, _, _ = model(**item)
                obs, reward, done, _, _ = env.step(int(logits.argmax(-1).item()))
                if done: break
            scores.append(reward)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--episodes", type=int, default=50); parser.add_argument("--env", choices=["memory", "parity"], default="memory"); parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS)); args = parser.parse_args()
    output = {}
    for name in args.variants:
        kwargs = VARIANTS[name]; torch.manual_seed(args.seed); model = ModularMultimodalAgent(**kwargs); train(model, args.seed, args.episodes, args.env)
        output[name] = {mask or "all": evaluate(model, args.seed + 1, mask, 30, args.env) for mask in (None, "image", "bytes_view", "symbolic")}
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
