"""LSTM classifier — input: (batch, 30, 63), output: softmax over N words."""

import torch
from torch import nn


class SignLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=63,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 30, 63)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
