"""LSTM(GRU) vs Transformer comparison harness (Week 3 → finalized Week 5).

Two things to compare:

  1. **Architecture cost** — parameter count + CPU inference latency. This part
     needs no dataset and runs anywhere; it is what determines whether a model
     fits the <200 ms end-to-end latency budget.

  2. **Accuracy** — top-1/3/5 on the held-out splits. This requires the processed
     dataset and a trained checkpoint per architecture. Produce the checkpoints
     with the *identical* pipeline so the comparison is fair:

         python -m src.train --arch gru         --no-resume   # -> checkpoints/best.pt
         python -m src.train --arch transformer --no-resume   # -> checkpoints/best_transformer.pt
         python -m src.evaluate --ckpt checkpoints/best.pt
         python -m src.evaluate --ckpt checkpoints/best_transformer.pt

Run this harness:
    python -m src.compare_archs                 # cost comparison only
    python -m src.compare_archs --runs 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models import build_model

_REPO_ROOT = Path(__file__).resolve().parent.parent

SEQ_LEN = 100
INPUT_DIM = 292
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5
NUM_CLASSES = 100


def _bench(model: torch.nn.Module, n_warmup: int, n_runs: int) -> dict:
    model.eval()
    x = torch.zeros(1, SEQ_LEN, INPUT_DIM, dtype=torch.float32)
    lengths = torch.tensor([SEQ_LEN], dtype=torch.long)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, lengths=lengths)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x, lengths=lengths)
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "params": int(sum(p.numel() for p in model.parameters())),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU vs Transformer cost comparison")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    common = dict(
        input_size=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    )

    results = {}
    print("=" * 64)
    print(f"ARCHITECTURE COST  (batch=1, seq_len={SEQ_LEN}, input_dim={INPUT_DIM}, CPU)")
    print("=" * 64)
    print(f"{'Arch':<14}{'Params':>12}{'Mean ms':>12}{'Median ms':>12}{'P95 ms':>12}")
    print("-" * 64)
    for arch in ("gru", "transformer"):
        model = build_model(arch, **common)
        stats = _bench(model, args.warmup, args.runs)
        results[arch] = stats
        print(
            f"{arch:<14}{stats['params']:>12,}{stats['mean_ms']:>12.3f}"
            f"{stats['median_ms']:>12.3f}{stats['p95_ms']:>12.3f}"
        )
    print("=" * 64)

    out = _REPO_ROOT / "arch_comparison_results.json"
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "seq_len": SEQ_LEN,
                    "input_dim": INPUT_DIM,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": NUM_CLASSES,
                    "torch_version": torch.__version__,
                    "device": "cpu",
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved -> {out}")
    print(
        "\nAccuracy comparison requires trained checkpoints - see module docstring."
    )


if __name__ == "__main__":
    main()
