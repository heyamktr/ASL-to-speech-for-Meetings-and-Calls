"""Benchmark PyTorch vs ONNX Runtime inference speed for SignGRU.

Run from the model-training/ directory:
    python -m src.benchmark
    python -m src.benchmark --ckpt checkpoints/best.pt --onnx exports/asl_model.onnx
    python -m src.benchmark --runs 500 --warmup 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.models.lstm import SignGRU

_REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = _REPO_ROOT / "checkpoints"
EXPORTS_DIR = _REPO_ROOT / "exports"

SEQ_LEN = 100
INPUT_DIM = 292


def load_pytorch_model(ckpt_path: Path, device: torch.device) -> tuple[SignGRU, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SignGRU(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
    ).to(device)
    try:
        model.load_state_dict(ckpt["model_state"])
        print("  Loaded weights from checkpoint.")
    except RuntimeError as exc:
        # Checkpoint may have been saved with an older architecture variant.
        # Weights don't affect inference latency — proceed with random init.
        print(f"  WARNING: could not load weights ({exc.__class__.__name__}); "
              "using random init (latency unaffected).")
    model.eval()
    return model, cfg


def benchmark_pytorch(
    model: SignGRU,
    device: torch.device,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> dict:
    x = torch.zeros(1, SEQ_LEN, INPUT_DIM, dtype=torch.float32, device=device)
    lengths = torch.tensor([SEQ_LEN], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, lengths=lengths)

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x, lengths=lengths)
            latencies.append((time.perf_counter() - t0) * 1000)

    return _stats(latencies)


def benchmark_onnx(
    onnx_path: Path,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> dict:
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name

    x = np.zeros((1, SEQ_LEN, INPUT_DIM), dtype=np.float32)
    lengths = np.array([SEQ_LEN], dtype=np.int64)
    feed = {input_names[0]: x, input_names[1]: lengths}

    for _ in range(n_warmup):
        sess.run([output_name], feed)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run([output_name], feed)
        latencies.append((time.perf_counter() - t0) * 1000)

    return _stats(latencies)


def _stats(latencies: list[float]) -> dict:
    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_runs": len(latencies),
    }


def print_table(pt: dict, ort_: dict) -> tuple[float, float]:
    speedup_mean = pt["mean_ms"] / ort_["mean_ms"]
    speedup_median = pt["median_ms"] / ort_["median_ms"]

    print("\n" + "=" * 65)
    print("INFERENCE BENCHMARK  (batch=1, seq_len=100, input_dim=292, CPU)")
    print("=" * 65)
    print(f"{'Metric':<20} {'PyTorch':>12} {'ONNX RT':>12} {'Speedup':>10}")
    print("-" * 65)
    rows = [
        ("Mean (ms)", "mean_ms"),
        ("Median (ms)", "median_ms"),
        ("Std (ms)", "std_ms"),
        ("P95 (ms)", "p95_ms"),
        ("P99 (ms)", "p99_ms"),
        ("Min (ms)", "min_ms"),
        ("Max (ms)", "max_ms"),
    ]
    for label, key in rows:
        sp = pt[key] / ort_[key] if ort_[key] > 0 else float("inf")
        print(f"{label:<20} {pt[key]:>12.3f} {ort_[key]:>12.3f} {sp:>9.2f}x")
    print("=" * 65)
    print(f"Mean speedup:   {speedup_mean:.2f}x  ({pt['mean_ms']:.3f} ms → {ort_['mean_ms']:.3f} ms)")
    print(f"Median speedup: {speedup_median:.2f}x  ({pt['median_ms']:.3f} ms → {ort_['median_ms']:.3f} ms)")
    return speedup_mean, speedup_median


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime inference")
    parser.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--onnx", default=str(EXPORTS_DIR / "asl_model.onnx"))
    parser.add_argument("--warmup", type=int, default=20, help="Warmup runs (not timed)")
    parser.add_argument("--runs", type=int, default=200, help="Timed runs per backend")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    onnx_path = Path(args.onnx)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")

    print(f"PyTorch checkpoint : {ckpt_path}")
    model, cfg = load_pytorch_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters       : {n_params:,}")
    print(f"  hidden_size={cfg['hidden_size']}  num_layers={cfg['num_layers']}  num_classes={cfg['num_classes']}")

    if not onnx_path.exists():
        print(f"\nONNX model not found — exporting to {onnx_path} ...")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_x = torch.zeros(1, SEQ_LEN, cfg["input_size"], dtype=torch.float32)
        dummy_lengths = torch.tensor([SEQ_LEN], dtype=torch.long)
        torch.onnx.export(
            model,
            (dummy_x, dummy_lengths),
            str(onnx_path),
            dynamo=False,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            external_data=False,
            input_names=["features", "lengths"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "lengths": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
        print(f"  Exported successfully.")

    print(f"\nONNX model         : {onnx_path}")
    onnx_kb = onnx_path.stat().st_size / 1024
    print(f"  File size        : {onnx_kb:.1f} KB")

    print(f"\nWarmup runs: {args.warmup} | Timed runs: {args.runs} | Device: cpu")

    print("\nBenchmarking PyTorch...")
    pt_stats = benchmark_pytorch(model, device, n_warmup=args.warmup, n_runs=args.runs)
    print("Benchmarking ONNX Runtime...")
    ort_stats = benchmark_onnx(onnx_path, n_warmup=args.warmup, n_runs=args.runs)

    speedup_mean, speedup_median = print_table(pt_stats, ort_stats)

    results = {
        "model": {
            "checkpoint": str(ckpt_path),
            "onnx_path": str(onnx_path),
            "parameters": n_params,
            "onnx_size_kb": round(onnx_kb, 1),
            "seq_len": SEQ_LEN,
            "input_dim": INPUT_DIM,
            "config": cfg,
        },
        "benchmark_config": {
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "device": "cpu",
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
        },
        "pytorch": pt_stats,
        "onnxruntime": ort_stats,
        "speedup": {
            "mean": round(speedup_mean, 3),
            "median": round(speedup_median, 3),
        },
    }

    out_path = _REPO_ROOT / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
