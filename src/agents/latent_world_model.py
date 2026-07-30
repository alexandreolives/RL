from __future__ import annotations

import torch
from torch import nn


class ActionConditionedLatentPredictor(nn.Module):
    """Small JEPA-style predictor for next latent state under an action."""

    def __init__(self, latent_dim: int = 64, num_actions: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or action.ndim != 1 or latent.size(0) != action.size(0):
            raise ValueError("latent must be (batch, dim) and action must be (batch,)")
        return self.predictor(torch.cat((latent, self.action_embedding(action.long())), dim=-1))


def latent_prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared latent prediction loss; normalization is intentionally external."""

    if predicted.shape != target.shape:
        raise ValueError("predicted and target latent tensors must have identical shapes")
    return (predicted - target.detach()).square().mean()


class JEPAWorldModel(nn.Module):
    """Minimal online/target JEPA world model with EMA target updates."""
    def __init__(self, input_dim: int = 64, latent_dim: int = 64, num_actions: int = 2, hidden_dim: int = 128, ema: float = 0.99):
        super().__init__()
        self.online_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        self.target_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        self.predictor = ActionConditionedLatentPredictor(latent_dim, num_actions, hidden_dim)
        self.ema = float(ema)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target.mul_(self.ema).add_(online, alpha=1.0 - self.ema)

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        online = self.online_encoder(state)
        with torch.no_grad():
            target = self.target_encoder(next_state)
        predicted = self.predictor(online, action)
        return predicted, target


def variance_covariance_regularizer(latent: torch.Tensor, *, target_std: float = 1.0) -> torch.Tensor:
    """VICReg-style variance/covariance penalty for small latent batches."""

    if latent.ndim != 2 or latent.size(0) < 2:
        raise ValueError("latent must have shape (batch, dim) with batch >= 2")
    centered = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = torch.relu(target_std - std).mean()
    covariance = centered.T @ centered / (latent.size(0) - 1)
    off_diag = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diag.square().mean()
    return variance_loss + covariance_loss
