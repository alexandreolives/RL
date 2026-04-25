from __future__ import annotations

from torch import nn

from .activations import build_activation


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, activation: str, dropout: float = 0.0) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        glu = self.activation_name in {"swiglu", "geglu"}
        in_hidden = hidden_dim * 2 if glu else hidden_dim
        out_hidden = hidden_dim

        self.fc1 = nn.Linear(d_model, in_hidden)
        self.act = build_activation(activation)
        self.fc2 = nn.Linear(out_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)

