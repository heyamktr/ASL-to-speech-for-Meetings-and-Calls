"""Transformer-encoder classifier — the Week 3 exploratory alternative to SignGRU.

This is a **drop-in replacement** for ``SignGRU``: same constructor signature
(``input_size, hidden_size, num_layers, dropout, num_classes``) and the same
``forward(x, lengths)`` contract, so ``src/train.py`` and ``src/evaluate.py`` can
train/evaluate it through the identical augmentation + TTA pipeline. That is what
makes the LSTM-vs-Transformer comparison a *fair* one — only the encoder changes.

Input : (batch, seq_len, input_size=292)   # 146 position + 146 velocity
Output: logits over N sign classes

Why this matches the GRU's design choices:
  * Same input projection (Linear → LayerNorm → ReLU) so the front-end is identical.
  * Padding frames are masked out of self-attention (``src_key_padding_mask``) AND
    out of the attention-pooling softmax — exactly like SignGRU masks padding.
  * Attention pooling (not a CLS token / last-state) so padded frames cannot leak
    into the pooled representation.
"""

import math

import torch
from torch import nn


class _PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def _pick_nhead(d_model: int) -> int:
    """Pick the largest head count in {8,4,2,1} that divides d_model."""
    for h in (8, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


class SignTransformer(nn.Module):
    def __init__(
        self,
        input_size: int = 292,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_classes: int = 100,
    ) -> None:
        super().__init__()

        # Same front-end as SignGRU: project raw landmarks, LayerNorm to put
        # coordinates / presence bits / velocity on a common scale.
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.pos_enc = _PositionalEncoding(hidden_size)

        nhead = _pick_nhead(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN — more stable on small datasets
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal attention pooling, mirroring SignGRU.
        self.attn = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       (B, T, input_size) — zero-padded to T frames
            lengths: (B,) int64 — real (non-padding) frame count per sample.
        Returns:
            logits: (B, num_classes)
        """
        B, T, _ = x.shape
        z = self.proj(x)            # (B, T, H)
        z = self.pos_enc(z)

        pad_mask = None
        if lengths is not None:
            # True where padded → ignored by attention and pooling.
            pad_mask = torch.arange(T, device=x.device)[None] >= lengths[:, None]

        z = self.encoder(z, src_key_padding_mask=pad_mask)  # (B, T, H)

        scores = self.attn(z).squeeze(-1)                   # (B, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (z * weights).sum(dim=1)                   # (B, H)

        return self.head(self.drop(pooled))
