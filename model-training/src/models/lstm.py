"""LSTM classifier — input: (batch, seq_len, 82), output: logits over N words.

Feature layout per timestep (82-dim):
  [0:63]  right hand coords  (21 landmarks × xyz)
  [63]    right hand present (1.0 = detected, 0.0 = absent)
  [64:82] pose coords        (6 landmarks × xyz: shoulders, elbows, wrists)
"""

import torch
from torch import nn


class SignLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=82,    # 63 rh coords + 1 rh_present + 18 pose
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 82)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)
        return self.head(self.drop(out[:, -1, :]))
