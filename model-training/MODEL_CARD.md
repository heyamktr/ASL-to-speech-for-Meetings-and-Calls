# Model Card — ASL Sign Classifier (`asl_model.onnx`)

A landmark-sequence classifier that maps a window of MediaPipe hand/pose landmarks
to one of 100 American Sign Language word glosses. It is the model served by the
backend inference server and consumed by the Chrome extension.

> This card follows the spirit of Mitchell et al., *Model Cards for Model
> Reporting* (2019). It is intended to be read before the model is deployed,
> published, or audited.

---

## Model details

| Field | Value |
|-------|-------|
| Name | ASL Sign Classifier (`SignGRU`) |
| Version | Week 5 (100-word vocabulary) |
| Architecture | Bidirectional GRU (2 layers) + temporal attention pooling |
| Parameters | 558,309 |
| Input | `(seq_len=100, 292)` — 146 position features + 146 velocity features per frame |
| Output | Softmax over 100 sign-gloss classes |
| Frameworks | PyTorch (training) → ONNX Runtime (serving) |
| Serving formats | `asl_model.onnx` (FP32), `asl_model_quant.onnx` (dynamic INT8) |
| Inference latency | ~2.6 ms median (ONNX Runtime, CPU, batch=1) — see `benchmark_results.json` |
| License / contact | Internal project; see repo owners |

### Input feature layout (per frame)

```
[0:63]    right-hand landmarks   (21 × xyz)
[63]      right-hand present     (0/1)
[64:127]  left-hand landmarks    (21 × xyz)
[127]     left-hand present      (0/1)
[128:146] pose landmarks         (6 × xyz: shoulders, elbows, wrists)
[146:292] velocity              (frame-to-frame diff of the above 146 dims)
```

Landmarks are normalized before inference: each hand is centered on its wrist and
scaled by the wrist→middle-knuckle distance; pose is centered on the shoulder
midpoint and scaled by shoulder width. This normalization is **mandatory** — it is
what lets the model generalize across users of different hand sizes and camera
distances. The identical normalization runs in `backend/app/inference.py`.

---

## Intended use

- **Primary use:** real-time recognition of single ASL word signs during video
  calls (Google Meet), to drive on-screen captions and synthesized speech.
- **Intended users:** deaf / hard-of-hearing signers communicating with hearing
  participants who do not sign.
- **In scope:** the **100 word glosses** listed in `label_map.json`, signed one at
  a time, facing the camera, with the signing hand(s) and upper body visible.

### Out of scope / not intended for

- **Not a full ASL translator.** It recognizes isolated word glosses, not
  grammar, fingerspelling, classifiers, facial grammar, or continuous sentences.
- **Not for safety-critical or legal interpretation.** Output is assistive and
  can be wrong; it must not be relied on where a misrecognition causes harm.
- **Not a biometric/identity system.** It is trained to be signer-independent and
  should not be used to identify individuals.
- Vocabulary outside the 100 trained glosses will be misclassified as the nearest
  known gloss or returned as `"uncertain"` (confidence below 0.4).

---

## Training data

- **Source:** [WLASL — Word-Level American Sign Language](https://dxli94.github.io/WLASL/)
  video dataset. Videos are **not** committed to this repo (size + dataset
  license); they are downloaded separately into `data/raw/`.
- **Vocabulary:** the 100 highest-frequency WLASL glosses (see `label_map.json`).
- **Preprocessing:** MediaPipe Holistic extracts hand + pose landmarks per frame
  (`src/preprocessing/extract_landmarks.py`); `src/preprocessing/build_dataset.py`
  filters to the top-100 glosses and produces train/val/test splits.
- **Scale:** roughly **~10 signing samples per class** — small. This dominates
  every modeling decision below.
- **Known data limitations:**
  - WLASL is sourced from public sign-language videos; signer demographics,
    lighting, camera quality, and dialect/regional variants are **not balanced**.
  - Some glosses have far fewer clean samples than others, so per-class accuracy
    is uneven (see failure cases).
  - The dataset reflects the signers who happen to appear in WLASL; it is not a
    representative sample of all ASL users.

---

## Training procedure

Because the dataset is tiny, the pipeline is built around overfitting control
(full details in `README.md` / `TRAINING_LOG.md` / `src/train.py`):

- Heavy augmentation: random temporal crop, time-warp (±20%), Gaussian jitter,
  scale jitter, horizontal flip (left/right-handed signers), 25% frame dropout,
  mixup.
- `CrossEntropyLoss` with inverse-frequency class weights + label smoothing 0.1.
- Adam (`weight_decay=1e-3`), `ReduceLROnPlateau`, Stochastic Weight Averaging,
  early stopping on validation loss.
- 4-crop test-time augmentation at evaluation.

---

## Evaluation

Checkpoint `checkpoints/best.pt`, 4-crop TTA (see `TRAINING_LOG.md`):

| Split | Top-1 | Top-3 | Top-5 |
|-------|-------|-------|-------|
| Validation (238 samples) | **71.0%** | 84.9% | 86.6% |
| Test (201 samples) | **63.2%** | 81.6% | 86.1% |

Top-3/Top-5 are much higher than Top-1, which is consistent with the failure mode
below: the model usually places the correct gloss among its top candidates but
confuses visually similar signs for the #1 slot.

### Accuracy by word category (qualitative)

Per-class accuracy is uneven. Regenerate the exact per-class table with:

```bash
python -m src.evaluate --split test --save-errors error_analysis.json
```

The recurring hard categories (from confusion analysis):

| Category | Example glosses | Why hard |
|----------|-----------------|----------|
| Near-mouth motion | `drink` / `eat` / `cook` | Cupped hand near the face; location + motion nearly identical |
| Head-location signs | `hat` / `hearing` / `doctor` | All contact/near the head; differ mainly in handshape detail |
| Similar movement path | `give` / `help` / `pull` | Outward/inward arm extension; subtle directionality at landmark level |
| Low-frequency glosses | `thanksgiving` / `secretary` / `basketball` | Few samples → underfit |
| Compound glosses | `birthday` / `graduate` | Multi-morpheme; high inter-signer speed variation |

---

## Known failure cases & limitations

- **Visually similar signs** (the categories above) are the dominant error source.
- **Lighting / occlusion:** poor lighting or a hand leaving the frame degrades
  MediaPipe landmarks and therefore predictions.
- **Hand-size / distance extremes:** mitigated by wrist normalization but not
  eliminated at the extremes.
- **Fast signers:** very fast signing can fall outside the 100-frame window or
  blur landmark trajectories.
- **Two-handed signs without a stable dominant hand** can degrade because
  normalization anchors on the wrist.
- **Out-of-vocabulary signs:** anything outside the 100 glosses is forced into a
  known class or returned as `"uncertain"`.
- **Distribution shift:** real webcam users (lighting, backgrounds, camera angle)
  differ from WLASL source videos; live accuracy may be below the test numbers.

### Mitigations in the serving path

- **Confidence gating:** predictions below 0.4 confidence return `"uncertain"`
  instead of a wrong word.
- **Temporal smoothing:** the backend requires the same prediction across
  consecutive windows before emitting it, suppressing single-frame noise.

---

## Quantization

A dynamic INT8 variant (`asl_model_quant.onnx`) is provided. It quantizes the
linear layers only (the GRU stays FP32), giving an **8.2% size reduction** with
**no accuracy change** (top-1 agreement 100%, max logit diff 0.0023). Speed is
unchanged because the GRU dominates compute. See `quantization_results.json` and
`TRAINING_LOG.md`. The FP32 model is the default in production; the quantized
model is available where binary size matters.

---

## Ethical considerations

- The model is **assistive, not authoritative.** Misrecognitions are expected;
  downstream UI must make uncertainty visible and never present output as a
  verified transcription.
- **Representation:** WLASL signer demographics are not balanced, so accuracy may
  vary across signers, skin tones, and regional sign variants. This has not been
  formally audited and should be before any broad release.
- **Privacy:** the model consumes only landmark coordinates, never raw video.
  MediaPipe runs in the browser and only 144 numbers/frame are sent to the server
  (see the project privacy notice).

---

## How to reproduce

```bash
# from model-training/
python -m src.preprocessing.extract_landmarks
python -m src.preprocessing.build_dataset
python -m src.train                                   # train the GRU
python -m src.evaluate --split test                   # accuracy + confusion
python -m src.evaluate --split test --save-errors error_analysis.json
python -m src.export_onnx                              # -> exports/asl_model.onnx
python -m src.quantize                                 # -> exports/asl_model_quant.onnx
python -m src.benchmark                                # PyTorch vs ONNX latency
python -m src.compare_archs                            # GRU vs Transformer cost
```
