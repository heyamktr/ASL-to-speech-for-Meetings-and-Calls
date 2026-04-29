"""Transformer classifier — experimental alternative to the LSTM (Week 3)."""

import torch
from torch import nn


class SignTransformer(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 128, nhead: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(63, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 30, 63)
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :])
