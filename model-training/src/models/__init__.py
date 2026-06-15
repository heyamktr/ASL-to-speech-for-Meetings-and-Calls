"""Model architectures for the ASL sign classifier."""

from src.models.lstm import SignGRU, SignLSTM
from src.models.transformer import SignTransformer


def build_model(arch: str, **kwargs):
    """Factory used by train.py / evaluate.py so both architectures share the
    exact same training + evaluation pipeline (fair LSTM-vs-Transformer test).

    Args:
        arch: "gru" (production) or "transformer" (Week 3 experiment).
        **kwargs: input_size, hidden_size, num_layers, dropout, num_classes.
    """
    arch = arch.lower()
    if arch in ("gru", "lstm", "signgru"):
        return SignGRU(**kwargs)
    if arch in ("transformer", "signtransformer"):
        return SignTransformer(**kwargs)
    raise ValueError(f"unknown arch '{arch}' (expected 'gru' or 'transformer')")


__all__ = ["SignGRU", "SignLSTM", "SignTransformer", "build_model"]
