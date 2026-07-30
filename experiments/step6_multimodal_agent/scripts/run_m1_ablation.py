from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from agents.modular_agent import ModularMultimodalAgent
from agents.multimodal_env import MultimodalMemoryEnv
from models.atoms.experiment import parameter_count


VARIANTS = {
    "baseline": {},
    "engram": {"use_engram": True},
    "jepa": {"use_jepa": True},
    "loop": {"use_loop": True},
    "spectral": {"use_spectral": True},
    "engram_jepa_loop": {"use_engram": True, "use_jepa": True, "use_loop": True},
    "full_m1": {"use_engram": True, "use_jepa": True, "use_loop": True, "use_spectral": True},
    "fourier_attention": {"use_loop": False, "hybrid_stages": ["fourier", "attention"]},
    "loop_only_hybrid": {"use_loop": False, "hybrid_stages": ["loop"]},
    "kda_loop": {"use_loop": False, "hybrid_stages": ["kda", "loop"]},
    "loop_kda": {"use_loop": False, "hybrid_stages": ["loop", "kda"]},
    "kda_loop_attention": {"use_loop": False, "hybrid_stages": ["kda", "loop", "attention"]},
    "loop_kda_attention": {"use_loop": False, "hybrid_stages": ["loop", "kda", "attention"]},
    "fourier_kda_loop": {"use_loop": False, "hybrid_stages": ["fourier", "kda", "loop"]},
    "kda_fourier_loop": {"use_loop": False, "hybrid_stages": ["kda", "fourier", "loop"]},
    "kda_loop_carry": {"use_loop": False, "hybrid_stages": ["kda", "loop"], "hybrid_iterations": 2},
}


@dataclass
class Result:
    variant: str
    seed: int
    parameters: int
    visible_reward: float
    hidden_reward: float
    forward_ms: float


def _batch(obs):
    return {
        "image": torch.from_numpy(obs["image"]).unsqueeze(0),
        "bytes_view": torch.from_numpy(obs["bytes"]).unsqueeze(0),
        "symbolic": torch.from_numpy(obs["symbolic"]).unsqueeze(0),
        "phase": torch.from_numpy(obs["phase"]).unsqueeze(0),
    }


def evaluate(model, *, seed: int, reveal: bool, episodes: int) -> float:
    env = MultimodalMemoryEnv(seed=seed, reveal_each_step=reveal)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rewards = []
    model.train()
    for _ in range(episodes):
        obs, _ = env.reset()
        log_probs, values, final_reward = [], [], 0.0
        for _ in range(env.horizon - 1):
            logits, value, _, _ = model(**_batch(obs))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            obs, reward, done, _, _ = env.step(int(action.item()))
            log_probs.append(dist.log_prob(action).squeeze())
            values.append(value.squeeze())
            final_reward = reward
            if done:
                break
        returns = torch.full((len(values),), float(final_reward))
        value_t = torch.stack(values)
        loss = -(torch.stack(log_probs) * (returns - value_t.detach())).mean() + 0.5 * (returns - value_t).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(final_reward)
    return float(np.mean(rewards[-20:]))


def run(seed: int, episodes: int, variant_names: list[str] | None = None, forward_warmup: int = 3) -> list[dict]:
    torch.manual_seed(seed)
    results = []
    selected = VARIANTS if variant_names is None else {name: VARIANTS[name] for name in variant_names}
    for name, kwargs in selected.items():
        model = ModularMultimodalAgent(**kwargs)
        sample = _batch(MultimodalMemoryEnv(seed=seed).reset(seed=seed)[0])
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(forward_warmup):
                model(**sample)
        forward_ms = (time.perf_counter() - start) * 1000 / max(1, forward_warmup)
        visible = evaluate(model, seed=seed, reveal=True, episodes=episodes)
        hidden = evaluate(model, seed=seed + 1000, reveal=False, episodes=episodes)
        results.append(asdict(Result(name, seed, parameter_count(model), visible, hidden, forward_ms)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS))
    parser.add_argument("--forward-warmup", type=int, default=3)
    args = parser.parse_args()
    results = run(args.seed, args.episodes, args.variants, args.forward_warmup)
    payload = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
