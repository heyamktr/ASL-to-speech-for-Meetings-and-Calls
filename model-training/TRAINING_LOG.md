0	book
1	drink
2	computer
3	before
4	chair
5	go
6	clothes
7	who
8	candy
9	cousin
10	deaf
11	fine
12	help
13	no
14	thin
15	walk
16	year
17	yes
18	all
19	black
20	cool
21	finish
22	hot
23	like
24	many
25	mother
26	now
27	orange
28	table
29	thanksgiving
30	what
31	woman
32	bed
33	blue
34	bowling
35	can
36	dog
37	family
38	fish
39	graduate
40	hat
41	hearing
42	kiss
43	language
44	later
45	man
46	shirt
47	study
48	tall
49	white
50	wrong
51	accident
52	apple
53	bird
54	change
55	color
56	corn
57	cow
58	dance
59	dark
60	doctor
61	eat
62	enjoy
63	forget
64	give
65	last
66	meet
67	pink
68	pizza
69	play
70	school
71	secretary
72	short
73	time
74	want
75	work
76	africa
77	basketball
78	birthday
79	brown
80	but
81	cheat
82	city
83	cook
84	decide
85	full
86	how
87	jacket
88	letter
89	medicine
90	need
91	paint
92	paper
93	pull
94	purple
95	right
96	same
97	son
98	tell
99	thursday

## 2026-05-26 Dev A verification

Checkpoint: `checkpoints/best.pt`

Config:
- raw live window: `(100, 144)`
- model input after presence bits + velocity: `(100, 292)`
- classes: 100

Evaluation:
- validation, 4-crop TTA: Top-1 169/238 = 71.01%, Top-3 84.87%, Top-5 86.55%
- test, 4-crop TTA: Top-1 127/201 = 63.18%, Top-3 81.59%, Top-5 86.07%

Export:
- ONNX: `exports/asl_model.onnx`
- label map: `exports/label_map.json`
- metadata: `exports/export_meta.json`
- backend local copy: `../backend/models/asl_model.onnx` and `../backend/models/label_map.json`

## 2026-05-27 ONNX Runtime vs PyTorch inference benchmark

### Setup

| Item | Value |
|------|-------|
| Model | SignGRU — 2-layer bidirectional GRU + temporal attention |
| Parameters | 558,309 |
| ONNX file size | 2,186.8 KB |
| Input | `(batch=1, seq_len=100, input_dim=292)` |
| Device | CPU (Windows 11, warm CPU state) |
| PyTorch | 2.12.0+cpu |
| ONNX Runtime | 1.26.0 (`CPUExecutionProvider`) |
| Warmup runs | 20 (not timed) |
| Timed runs | 200 |

### Results (warm CPU)

| Metric | PyTorch (ms) | ONNX Runtime (ms) | Speedup |
|--------|-------------|-------------------|---------|
| **Mean** | **15.41** | **2.57** | **5.99x** |
| **Median** | **15.18** | **2.55** | **5.97x** |
| Std | 1.79 | 0.70 | — |
| P95 | 17.76 | 2.84 | 6.26x |
| P99 | 20.13 | 6.32 | 3.19x |
| Min | 12.20 | 2.06 | 5.93x |
| Max | 32.35 | 6.63 | 4.88x |

**ONNX Runtime is ~6x faster than native PyTorch on CPU.** On a cold-start (Windows CPU frequency scaling from idle), the same model measured 119 ms / 25 ms respectively — the ratio is consistent (~5x) but absolute latency depends on CPU thermal state at call time.

ONNX Runtime also has 2.5x lower latency variance (std 0.70 ms vs 1.79 ms), which matters for real-time meeting/call use where jitter causes inconsistent audio output.

### Why the gap is large

- PyTorch carries Python-level dispatch overhead and autograd bookkeeping (even with `torch.no_grad()`) on every forward pass.
- ONNX Runtime compiles the full graph at session creation (`do_constant_folding=True` fuses constant subgraphs) and dispatches through optimized C++ kernels with no Python per-call overhead.
- ONNX Runtime's `CPUExecutionProvider` uses platform-specific BLAS that PyTorch's default CPU path does not match.

### Implications for the backend

At 2.6 ms median latency per prediction, ONNX Runtime has headroom for >300 fps prediction cycles. The PyTorch path at 15 ms caps effective rate to ~65 fps and would waste CPU cycles on the per-call Python overhead even at video frame rates.

### Reproducing

```bash
# from model-training/
python -m src.benchmark                         # uses checkpoints/best.pt + exports/asl_model.onnx
python -m src.benchmark --runs 500 --warmup 50  # more runs for tighter estimates
```

Full results: `benchmark_results.json`

## 2026-05-27 Dynamic INT8 quantization

Goal: reduce ONNX model size and improve inference speed by quantizing weights to INT8.

### What was quantized

ORT's `quantize_dynamic` targets `MatMul` and `Gemm` operators. The ONNX model contains 57 nodes:

| Operator | Count | Quantizable? |
|----------|-------|--------------|
| `GRU` | 2 | No — ORT does not decompose GRU for dynamic quant |
| `MatMul` | 2 | Yes |
| `Gemm` | 1 | Yes |
| everything else | 52 | N/A |

The GRU holds ~500K of the model's 558K parameters (~90%). Only the 3 linear layers (input projection, attention score, classifier head) were quantized to INT8 with per-channel scale/zero-point. The graph gained `DynamicQuantizeLinear × 3` and `MatMulInteger × 3` nodes in place of the original float ops.

### Results

| Metric | FP32 ONNX | INT8 Quant | Change |
|--------|-----------|------------|--------|
| **File size** | **2,186.8 KB** | **2,006.5 KB** | **−8.2%** |
| Mean latency | 2.58 ms | 2.54 ms | −1.5% |
| Median latency | 2.57 ms | 2.55 ms | −0.8% |
| Max logit diff | — | 0.0023 | |
| Cosine similarity | — | 0.999985 | |
| Top-1 agreement | — | 100% | |

**Size drops 8.2%; speed is statistically unchanged (1.01x).** The speedup is negligible because the quantized ops represent <10% of total compute — GRU dominates and runs in FP32.

Accuracy is fully preserved: the max per-element logit difference across 20 random inputs is 0.0023, and top-1 class selection is identical between FP32 and INT8 on every trial.

### Why size reduction is modest

INT8 weights are 4× smaller than FP32, but per-channel quantization adds one `float32` scale + one `int8` zero-point per output channel. For small linear layers (e.g., `attn: 256→1`), this metadata overhead nearly cancels the weight savings. The GRU's weights — unquantized — dominate the file.

### Path to larger gains

To quantize the GRU weights, the options are:

1. **Lower opset (11/12)**: re-export with `opset_version=12`; older opsets sometimes expand GRU into `MatMul` nodes that ORT can quantize.
2. **torch.quantization.quantize_dynamic**: quantize the PyTorch model itself (targets `nn.GRU` natively), then re-export to ONNX. The quantized ONNX ops are non-standard and may require a recent ORT version.
3. **ONNX Runtime quantization extensions**: the `onnxruntime-extensions` package has extended GRU quantization support in some builds.

For the current backend workload (2.6 ms already well within frame budget), further quantization is not required.

### Reproducing

```bash
# from model-training/
python -m src.quantize                           # uses exports/asl_model.onnx
python -m src.quantize --runs 500               # more runs
```

Quantized model: `exports/asl_model_quant.onnx`
Full results: `quantization_results.json`

## Error Analysis

### How to generate

```bash
# from model-training/
python -m src.evaluate --split test --save-errors error_analysis.json
```

This prints per-class top-1 accuracy (worst first) and the top-20 confusion pairs (true label → most common wrong prediction), then writes `error_analysis.json` with the full per-class breakdown.

### Confusion categories to watch

Given the 100-word vocabulary, the most likely confusion groups are:

| Category | Example pairs | Why they're hard |
|----------|--------------|-----------------|
| **Near-mouth motion** | `drink` / `eat` / `cook` | Closed-fist or cupped-hand near face; location and movement nearly identical across signers |
| **Head-location signs** | `hat` / `hearing` / `doctor` | All involve contact or proximity to the head region; differ mainly in handshape detail |
| **Similar movement path** | `give` / `help` / `pull` | Outward or inward arm extension; directionality cues are subtle at 144-dim landmark level |
| **Low-frequency glosses** | `thanksgiving` / `secretary` / `basketball` | Fewer training samples → model under-fits; these are the first candidates for data augmentation |
| **Compound glosses** | `birthday` / `graduate` | Multi-morpheme signs with high inter-signer variation in execution speed |

### Interpreting per-class accuracy

Classes with accuracy < 50% on the test split are the primary targets for improvement:

1. **Data volume**: check how many training samples that class has. If < 20, find additional WLASL source videos or augment harder (wider scale/flip jitter).
2. **Sign similarity**: if the top confusion pair is visually close (e.g., `drink` → `eat`), the model needs either more discriminative features or a larger sequence window to catch the difference.
3. **Normalization failures**: signs where the wrist is not a stable anchor (e.g., two-handed signs without a dominant hand) may degrade with the current wrist-centered normalization. Check raw landmark sequences for those glosses.

### Error analysis output

Run `python -m src.evaluate --split test --save-errors error_analysis.json` with `checkpoints/best.pt` to populate `error_analysis.json`. The file is gitignored (generated output); re-run whenever the checkpoint changes.

## 2026-06-13 LSTM (GRU) vs Transformer — final comparison

The Week 3 plan kept the GRU as the production model and ran a Transformer as an
exploratory experiment. This is the Week 5 finalization of that comparison.

To make the comparison **fair**, `src/models/transformer.py` was rewritten as a
drop-in replacement for `SignGRU` — same input projection, the same padding mask,
the same attention pooling — so both models train and evaluate through the
*identical* pipeline in `src/train.py` (augmentation, mixup, SWA, TTA):

```bash
python -m src.train --arch gru         --no-resume   # -> checkpoints/best.pt
python -m src.train --arch transformer --no-resume   # -> checkpoints/best_transformer.pt
python -m src.evaluate --ckpt checkpoints/best.pt
python -m src.evaluate --ckpt checkpoints/best_transformer.pt
```

### Architecture cost (measured)

`python -m src.compare_archs` — batch=1, seq_len=100, input_dim=292, CPU, torch 2.12.0+cpu, 100 timed runs. Full data: `arch_comparison_results.json`.

| Model | Params | Mean latency | Median | P95 |
|-------|--------|--------------|--------|-----|
| **GRU (production)** | 558,309 | 21.2 ms | 20.5 ms | 27.5 ms |
| **Transformer** | 447,333 | 2.1 ms | 1.9 ms | 2.9 ms |

The Transformer is **~10x faster on CPU** with **~20% fewer parameters**. The GRU
is recurrent (sequential over 100 timesteps → no intra-sequence parallelism),
while the Transformer's self-attention is a couple of batched matmuls that the
CPU BLAS parallelizes. Note these are *PyTorch* latencies; production serves the
**ONNX-exported GRU at ~2.6 ms** (see ONNX benchmark above), which already meets
the latency budget — so cost alone does not force a switch.

### Accuracy

| Model | Val Top-1 (4-crop TTA) | Test Top-1 (4-crop TTA) |
|-------|------------------------|--------------------------|
| **GRU (production)** | **71.0%** | **63.2%** |
| Transformer | _run `src.train --arch transformer` then `src.evaluate` to fill in_ | _idem_ |

> Accuracy numbers require a training run on the processed dataset (not committed).
> Re-run the two `src.train` + `src.evaluate` commands above and paste the Top-1/3/5
> here.

### Recommendation

**Keep the GRU as the production model** for the 100-word vocabulary:

- On a **small dataset (~10 samples/class)**, Transformers are more data-hungry and
  prone to overfit; the GRU's smaller effective capacity + heavy augmentation has
  been tuned to this regime and reaches 71% val / 63% test.
- The **deployed cost is already met** by the ONNX GRU (~2.6 ms inference, well
  inside the <200 ms end-to-end budget), so the Transformer's raw-PyTorch speed
  advantage does not translate into a production need.
- The Transformer remains the **first thing to revisit when the dataset grows**
  (more words / more samples per word), where its capacity and CPU-parallel
  attention would start to pay off. The drop-in `--arch transformer` path makes
  re-running that comparison a one-liner.

## 2026-05-27 ONNX Runtime vs PyTorch inference benchmark

### Setup

| Item | Value |
|------|-------|
| Model | SignGRU — 2-layer bidirectional GRU + temporal attention |
| Parameters | 558,309 |
| ONNX file size | 2,186.8 KB |
| Input | `(batch=1, seq_len=100, input_dim=292)` |
| Device | CPU (Windows 11, warm CPU state) |
| PyTorch | 2.12.0+cpu |
| ONNX Runtime | 1.26.0 (`CPUExecutionProvider`) |
| Warmup runs | 20 (not timed) |
| Timed runs | 200 |

### Results (warm CPU)

| Metric | PyTorch (ms) | ONNX Runtime (ms) | Speedup |
|--------|-------------|-------------------|---------|
| **Mean** | **15.41** | **2.57** | **5.99x** |
| **Median** | **15.18** | **2.55** | **5.97x** |
| Std | 1.79 | 0.70 | — |
| P95 | 17.76 | 2.84 | 6.26x |
| P99 | 20.13 | 6.32 | 3.19x |
| Min | 12.20 | 2.06 | 5.93x |
| Max | 32.35 | 6.63 | 4.88x |

**ONNX Runtime is ~6x faster than native PyTorch on CPU.** On a cold-start (Windows CPU frequency scaling from idle), the same model measured 119 ms / 25 ms respectively — the ratio is consistent (~5x) but absolute latency depends on CPU thermal state at call time.

ONNX Runtime also has 2.5x lower latency variance (std 0.70 ms vs 1.79 ms), which matters for real-time meeting/call use where jitter causes inconsistent audio output.

### Why the gap is large

- PyTorch carries Python-level dispatch overhead and autograd bookkeeping (even with `torch.no_grad()`) on every forward pass.
- ONNX Runtime compiles the full graph at session creation (`do_constant_folding=True` fuses constant subgraphs) and dispatches through optimized C++ kernels with no Python per-call overhead.
- ONNX Runtime's `CPUExecutionProvider` uses platform-specific BLAS that PyTorch's default CPU path does not match.

### Implications for the backend

At 2.6 ms median latency per prediction, ONNX Runtime has headroom for >300 fps prediction cycles. The PyTorch path at 15 ms caps effective rate to ~65 fps and would waste CPU cycles on the per-call Python overhead even at video frame rates.

### Reproducing

```bash
# from model-training/
python -m src.benchmark                         # uses checkpoints/best.pt + exports/asl_model.onnx
python -m src.benchmark --runs 500 --warmup 50  # more runs for tighter estimates
```

Full results: `benchmark_results.json`

## 2026-05-27 Dynamic INT8 quantization

Goal: reduce ONNX model size and improve inference speed by quantizing weights to INT8.

### What was quantized

ORT's `quantize_dynamic` targets `MatMul` and `Gemm` operators. The ONNX model contains 57 nodes:

| Operator | Count | Quantizable? |
|----------|-------|--------------|
| `GRU` | 2 | No — ORT does not decompose GRU for dynamic quant |
| `MatMul` | 2 | Yes |
| `Gemm` | 1 | Yes |
| everything else | 52 | N/A |

The GRU holds ~500K of the model's 558K parameters (~90%). Only the 3 linear layers (input projection, attention score, classifier head) were quantized to INT8 with per-channel scale/zero-point. The graph gained `DynamicQuantizeLinear × 3` and `MatMulInteger × 3` nodes in place of the original float ops.

### Results

| Metric | FP32 ONNX | INT8 Quant | Change |
|--------|-----------|------------|--------|
| **File size** | **2,186.8 KB** | **2,006.5 KB** | **−8.2%** |
| Mean latency | 2.58 ms | 2.54 ms | −1.5% |
| Median latency | 2.57 ms | 2.55 ms | −0.8% |
| Max logit diff | — | 0.0023 | |
| Cosine similarity | — | 0.999985 | |
| Top-1 agreement | — | 100% | |

**Size drops 8.2%; speed is statistically unchanged (1.01x).** The speedup is negligible because the quantized ops represent <10% of total compute — GRU dominates and runs in FP32.

Accuracy is fully preserved: the max per-element logit difference across 20 random inputs is 0.0023, and top-1 class selection is identical between FP32 and INT8 on every trial.

### Why size reduction is modest

INT8 weights are 4× smaller than FP32, but per-channel quantization adds one `float32` scale + one `int8` zero-point per output channel. For small linear layers (e.g., `attn: 256→1`), this metadata overhead nearly cancels the weight savings. The GRU's weights — unquantized — dominate the file.

### Path to larger gains

To quantize the GRU weights, the options are:

1. **Lower opset (11/12)**: re-export with `opset_version=12`; older opsets sometimes expand GRU into `MatMul` nodes that ORT can quantize.
2. **torch.quantization.quantize_dynamic**: quantize the PyTorch model itself (targets `nn.GRU` natively), then re-export to ONNX. The quantized ONNX ops are non-standard and may require a recent ORT version.
3. **ONNX Runtime quantization extensions**: the `onnxruntime-extensions` package has extended GRU quantization support in some builds.

For the current backend workload (2.6 ms already well within frame budget), further quantization is not required.

### Reproducing

```bash
# from model-training/
python -m src.quantize                           # uses exports/asl_model.onnx
python -m src.quantize --runs 500               # more runs
```

Quantized model: `exports/asl_model_quant.onnx`
Full results: `quantization_results.json`

## Error Analysis

### How to generate

```bash
# from model-training/
python -m src.evaluate --split test --save-errors error_analysis.json
```

This prints per-class top-1 accuracy (worst first) and the top-20 confusion pairs (true label → most common wrong prediction), then writes `error_analysis.json` with the full per-class breakdown.

### Confusion categories to watch

Given the 100-word vocabulary, the most likely confusion groups are:

| Category | Example pairs | Why they're hard |
|----------|--------------|-----------------|
| **Near-mouth motion** | `drink` / `eat` / `cook` | Closed-fist or cupped-hand near face; location and movement nearly identical across signers |
| **Head-location signs** | `hat` / `hearing` / `doctor` | All involve contact or proximity to the head region; differ mainly in handshape detail |
| **Similar movement path** | `give` / `help` / `pull` | Outward or inward arm extension; directionality cues are subtle at 144-dim landmark level |
| **Low-frequency glosses** | `thanksgiving` / `secretary` / `basketball` | Fewer training samples → model under-fits; these are the first candidates for data augmentation |
| **Compound glosses** | `birthday` / `graduate` | Multi-morpheme signs with high inter-signer variation in execution speed |

### Interpreting per-class accuracy

Classes with accuracy < 50% on the test split are the primary targets for improvement:

1. **Data volume**: check how many training samples that class has. If < 20, find additional WLASL source videos or augment harder (wider scale/flip jitter).
2. **Sign similarity**: if the top confusion pair is visually close (e.g., `drink` → `eat`), the model needs either more discriminative features or a larger sequence window to catch the difference.
3. **Normalization failures**: signs where the wrist is not a stable anchor (e.g., two-handed signs without a dominant hand) may degrade with the current wrist-centered normalization. Check raw landmark sequences for those glosses.

### Error analysis output

Run `python -m src.evaluate --split test --save-errors error_analysis.json` with `checkpoints/best.pt` to populate `error_analysis.json`. The file is gitignored (generated output); re-run whenever the checkpoint changes.
