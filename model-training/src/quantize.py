"""Apply dynamic INT8 quantization to the exported ONNX model.

ORT's dynamic quantization targets MatMul and Gemm operators.  The GRU
operator is not decomposed at this stage, so the quantization applies to the
linear layers (projection, attention head, classifier head) rather than the
recurrent weights.

Run from the model-training/ directory:
    python -m src.quantize
    python -m src.quantize --input exports/asl_model.onnx --output exports/asl_model_quant.onnx
    python -m src.quantize --runs 500
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process

_REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = _REPO_ROOT / "exports"

SEQ_LEN = 100
INPUT_DIM = 292


def _op_counts(model_path: Path) -> dict[str, int]:
    m = onnx.load(str(model_path))
    return dict(Counter(n.op_type for n in m.graph.node))


def _make_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _feed(sess: ort.InferenceSession, rng: np.random.Generator) -> dict:
    names = [inp.name for inp in sess.get_inputs()]
    return {
        names[0]: rng.standard_normal((1, SEQ_LEN, INPUT_DIM)).astype(np.float32),
        names[1]: np.array([SEQ_LEN], dtype=np.int64),
    }


def benchmark(path: Path, n_warmup: int, n_runs: int) -> dict:
    sess = _make_session(path)
    out_name = sess.get_outputs()[0].name
    rng = np.random.default_rng(0)
    feed = _feed(sess, rng)

    for _ in range(n_warmup):
        sess.run([out_name], feed)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run([out_name], feed)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_runs": n_runs,
    }


def verify_accuracy(orig_path: Path, quant_path: Path, n_trials: int = 20) -> dict:
    orig = _make_session(orig_path)
    quant = _make_session(quant_path)
    orig_out = orig.get_outputs()[0].name
    quant_out = quant.get_outputs()[0].name

    rng = np.random.default_rng(42)
    diffs, cos_sims = [], []
    for _ in range(n_trials):
        x = rng.standard_normal((1, SEQ_LEN, INPUT_DIM)).astype(np.float32)
        lengths = np.array([SEQ_LEN], dtype=np.int64)
        orig_names = [i.name for i in orig.get_inputs()]
        quant_names = [i.name for i in quant.get_inputs()]

        a = orig.run([orig_out], {orig_names[0]: x, orig_names[1]: lengths})[0][0]
        b = quant.run([quant_out], {quant_names[0]: x, quant_names[1]: lengths})[0][0]

        diffs.append(float(np.max(np.abs(a - b))))

        # cosine similarity of logit vectors — measures ranking preservation
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        cos_sims.append(cos_sim)

        # top-1 agreement
        # (stored separately so we can compute agreement rate)

    top1_orig = []
    top1_quant = []
    rng2 = np.random.default_rng(42)
    for _ in range(n_trials):
        x = rng2.standard_normal((1, SEQ_LEN, INPUT_DIM)).astype(np.float32)
        lengths = np.array([SEQ_LEN], dtype=np.int64)
        orig_names = [i.name for i in orig.get_inputs()]
        quant_names = [i.name for i in quant.get_inputs()]
        a = orig.run([orig_out], {orig_names[0]: x, orig_names[1]: lengths})[0][0]
        b = quant.run([quant_out], {quant_names[0]: x, quant_names[1]: lengths})[0][0]
        top1_orig.append(int(np.argmax(a)))
        top1_quant.append(int(np.argmax(b)))

    top1_agreement = sum(o == q for o, q in zip(top1_orig, top1_quant)) / n_trials

    return {
        "max_logit_diff": round(float(max(diffs)), 4),
        "mean_logit_diff": round(float(np.mean(diffs)), 4),
        "mean_cosine_sim": round(float(np.mean(cos_sims)), 6),
        "top1_agreement_rate": round(top1_agreement, 3),
    }


def _print_table(fp32: dict, int8: dict) -> tuple[float, float]:
    speedup_mean = fp32["mean_ms"] / int8["mean_ms"]
    speedup_median = fp32["median_ms"] / int8["median_ms"]
    rows = [
        ("Mean (ms)", "mean_ms"),
        ("Median (ms)", "median_ms"),
        ("Std (ms)", "std_ms"),
        ("P95 (ms)", "p95_ms"),
        ("P99 (ms)", "p99_ms"),
        ("Min (ms)", "min_ms"),
        ("Max (ms)", "max_ms"),
    ]
    print("\n" + "=" * 65)
    print("QUANTIZATION BENCHMARK  (batch=1, seq_len=100, CPU)")
    print("=" * 65)
    print(f"{'Metric':<20} {'FP32 ONNX':>12} {'INT8 Quant':>12} {'Speedup':>10}")
    print("-" * 65)
    for label, key in rows:
        sp = fp32[key] / int8[key] if int8[key] > 0 else float("inf")
        print(f"{label:<20} {fp32[key]:>12.3f} {int8[key]:>12.3f} {sp:>9.2f}x")
    print("=" * 65)
    print(f"Mean speedup:   {speedup_mean:.2f}x")
    print(f"Median speedup: {speedup_median:.2f}x")
    return speedup_mean, speedup_median


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for the ASL ONNX model")
    parser.add_argument("--input", default=str(EXPORTS_DIR / "asl_model.onnx"))
    parser.add_argument("--output", default=str(EXPORTS_DIR / "asl_model_quant.onnx"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    orig_path = Path(args.input)
    quant_path = Path(args.output)
    prep_path = quant_path.parent / (quant_path.stem + "_preprocessed.onnx")

    if not orig_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {orig_path}\nRun: python -m src.export_onnx"
        )

    print(f"Input  : {orig_path}")
    print(f"Output : {quant_path}")

    # ── 1. Inspect original model ──────────────────────────────────────────────
    orig_ops = _op_counts(orig_path)
    print(f"\nOriginal operators ({sum(orig_ops.values())} nodes):")
    for op, cnt in sorted(orig_ops.items()):
        print(f"  {op}: {cnt}")

    # ── 2. Preprocessing: shape inference + ORT model optimizations ───────────
    print("\nPreprocessing model (shape inference + ORT optimizations)...")
    quant_pre_process(
        input_model_path=str(orig_path),
        output_model_path=str(prep_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=True,
        verbose=0,
    )
    prep_ops = _op_counts(prep_path)
    new_ops = {op: cnt for op, cnt in prep_ops.items() if op not in orig_ops or cnt != orig_ops[op]}
    if new_ops:
        print("  Operator changes after preprocessing:")
        for op, cnt in sorted(new_ops.items()):
            prev = orig_ops.get(op, 0)
            print(f"    {op}: {prev} → {cnt}")
    else:
        print("  No operator changes (graph structure unchanged).")

    # ── 3. Dynamic INT8 quantization ──────────────────────────────────────────
    # ORT dynamic quant targets MatMul and Gemm by default. GRU is not
    # decomposed by this path — the recurrent weights stay in FP32.
    print("\nApplying dynamic INT8 quantization (MatMul + Gemm → QInt8 weights)...")
    quantize_dynamic(
        model_input=str(prep_path),
        model_output=str(quant_path),
        weight_type=QuantType.QInt8,
        per_channel=True,   # per-output-channel scale/zp → better accuracy
        reduce_range=False,  # keep full [-128, 127] range
    )

    # Clean up intermediate preprocessed file
    prep_path.unlink(missing_ok=True)

    quant_ops = _op_counts(quant_path)
    quant_only_ops = {op: cnt for op, cnt in quant_ops.items() if op not in orig_ops}
    print(f"  Quantization-specific ops added: {quant_only_ops or 'none'}")

    # ── 4. File sizes ──────────────────────────────────────────────────────────
    orig_kb = orig_path.stat().st_size / 1024
    quant_kb = quant_path.stat().st_size / 1024
    size_reduction_pct = (1 - quant_kb / orig_kb) * 100

    print(f"\nFile sizes:")
    print(f"  Original  : {orig_kb:,.1f} KB")
    print(f"  Quantized : {quant_kb:,.1f} KB  ({size_reduction_pct:+.1f}%)")

    # ── 5. Accuracy verification ───────────────────────────────────────────────
    print("\nVerifying output accuracy (20 random inputs)...")
    acc = verify_accuracy(orig_path, quant_path, n_trials=20)
    print(f"  Max logit diff        : {acc['max_logit_diff']}")
    print(f"  Mean logit diff       : {acc['mean_logit_diff']}")
    print(f"  Mean cosine similarity: {acc['mean_cosine_sim']:.6f}  (1.0 = identical ranking)")
    print(f"  Top-1 agreement rate  : {acc['top1_agreement_rate']:.1%}")

    # ── 6. Benchmark ──────────────────────────────────────────────────────────
    print(f"\nBenchmarking ({args.warmup} warmup + {args.runs} timed runs each)...")
    fp32_stats = benchmark(orig_path, args.warmup, args.runs)
    int8_stats = benchmark(quant_path, args.warmup, args.runs)
    speedup_mean, speedup_median = _print_table(fp32_stats, int8_stats)

    # ── 7. Save results ────────────────────────────────────────────────────────
    results = {
        "original_path": str(orig_path),
        "quantized_path": str(quant_path),
        "size": {
            "original_kb": round(orig_kb, 1),
            "quantized_kb": round(quant_kb, 1),
            "reduction_pct": round(size_reduction_pct, 1),
        },
        "operators": {
            "original": orig_ops,
            "quantized": quant_ops,
            "quantization_ops_added": quant_only_ops,
        },
        "accuracy": acc,
        "benchmark_config": {
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "device": "cpu",
            "onnxruntime_version": ort.__version__,
        },
        "fp32_onnx": fp32_stats,
        "int8_quantized": int8_stats,
        "speedup": {
            "mean": round(speedup_mean, 3),
            "median": round(speedup_median, 3),
        },
    }

    out_path = _REPO_ROOT / "quantization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"Quantized model : {quant_path}")


if __name__ == "__main__":
    main()
