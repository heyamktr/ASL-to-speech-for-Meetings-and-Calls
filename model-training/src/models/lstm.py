"""Bidirectional GRU classifier with temporal attention pooling.

Input : (batch, seq_len, INPUT_DIM=146)
Output: logits over N sign classes

Feature layout per timestep (146-dim):
  [0:63]   right hand coords  (21 landmarks × xyz)
  [63]     right hand present (1.0 detected, 0.0 absent)
  [64:127] left hand coords   (21 landmarks × xyz)
  [127]    left hand present  (1.0 detected, 0.0 absent)
  [128:146] pose coords       (6 landmarks × xyz: shoulders, elbows, wrists)

Why GRU over LSTM here:
  ~10 samples/class means overfitting is the main failure mode. GRU has ~25%
  fewer parameters than LSTM and trains faster, with no accuracy loss on
  sequences this short (~67 frames median).

Why attention pooling over last-hidden-state:
  Sequences are zero-padded; the last hidden state processes padding frames.
  Attention learns to weight real signing frames and ignore padding.
"""

import torch
from torch import nn


class SignGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 146,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_classes: int = 100,
    ) -> None:
        super().__init__()

        # Project raw landmarks into a richer embedding before the GRU.
        # LayerNorm here stabilises training at the start when coordinates
        # span very different ranges (wrist-centred coords vs presence bits).
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Single-head temporal attention: learns which frames matter most.
        self.attn = nn.Linear(hidden_size * 2, 1)

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       (B, T, input_size) — zero-padded to T frames
            lengths: (B,) int64 — actual (non-padding) frame count per sample.
                     Padding frames are masked out of the attention softmax so
                     they cannot contaminate the pooled representation.
        Returns:
            logits: (B, num_classes)
        """
        x = self.proj(x)                        # (B, T, H)
        out, _ = self.gru(x)                    # (B, T, H*2)

        scores = self.attn(out).squeeze(-1)     # (B, T)
        if lengths is not None:
            T = out.size(1)
            pad_mask = torch.arange(T, device=x.device)[None] >= lengths[:, None]
            scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)   # (B, T, 1)
        pooled = (out * weights).sum(dim=1)                    # (B, H*2)

        return self.head(self.drop(pooled))


# Keep old name as alias so any external code that imported SignLSTM still works.
SignLSTM = SignGRU
