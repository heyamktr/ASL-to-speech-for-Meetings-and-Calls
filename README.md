# ASL-to-Speech for Meetings & Calls

> A privacy-first Chrome extension that translates American Sign Language into spoken voice **in real time** during Google Meet, Zoom Web, Messenger, and Microsoft Teams — so deaf and hard-of-hearing users can be heard without any change on their participants' end.

---

## What It Does

Turn on the extension. Sign in front of your webcam. Your ASL signs are recognized, displayed in an overlay panel, and synthesized into audio that is injected directly into your microphone stream — other participants hear a natural voice without installing anything.

**No video is ever sent over the network.** The browser extracts 144 numbers per frame (hand and pose landmark coordinates) and sends only those.

---

## Key Technical Highlights

| Dimension | Achievement |
|---|---|
| **Inference latency** | **2.57 ms** median (ONNX Runtime, CPU-only) — 5.97× faster than PyTorch |
| **Vocabulary** | 100 ASL word glosses (WLASL dataset) |
| **Top-1 / Top-3 accuracy** | 63.18% / 81.59% on held-out test set (4-crop TTA) |
| **Privacy** | Zero video transmitted; only 144 landmark floats per frame over WebSocket |
| **Model size** | 558K parameters (bidirectional GRU) — chosen over Transformer to avoid overfitting on ~10 samples/class |
| **Deployment** | Dockerized FastAPI + Redis on Fly.io; auto-scales to zero when idle |

---

## Architecture

```
┌─────────────────────────────── Chrome Browser ───────────────────────────────┐
│                                                                               │
│  Video Call Tab (Meet / Zoom / Teams / Messenger)                             │
│    ↓                                                                          │
│  [injected/mediapipe-bridge.ts]  ← runs in MAIN world (page context)         │
│    MediaPipe Holistic: extracts 21+21 hand landmarks + 6 pose joints (xyz)   │
│    → postMessage → 144 floats/frame                                           │
│    ↓                                                                          │
│  [content/mediapipe/index.ts]  ← HandTracker (content-script world)          │
│    requestAnimationFrame loop; manages camera acquisition                     │
│    ↓                                                                          │
│  [content/websocket/client.ts]  WSClient                                      │
│    Exponential-backoff reconnection; buffers up to 100 pending frames        │
│    ↓  { landmarks: float[144], session_id, timestamp }  over WebSocket       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                               │  WebSocket (ws://)
                               ▼
┌─────────────────────── FastAPI Backend (Docker / Fly.io) ─────────────────────┐
│                                                                               │
│  [websocket.py]  validates 144-dim frame, dispatches to session buffer       │
│    ↓                                                                          │
│  [session/buffer.py]  Redis-backed sliding window (100 frames)               │
│    Adaptive stride: inference every 5 frames (hand visible) or 15 (idle)    │
│    ↓                                                                          │
│  [inference.py]  SignClassifier (ONNX Runtime)                               │
│    Feature pipeline: normalize hand + pose → compute velocity deltas        │
│    Input: (1, 100, 292)   Output: softmax over 100 classes                  │
│    LFU cache keyed by quantized window fingerprint (Blake2b)                 │
│    ↓                                                                          │
│  Consecutive-frame smoothing gate (word must appear 3× before emit)          │
│    ↓  { word, confidence }  over WebSocket                                   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                               │  WebSocket (response)
                               ▼
┌─────────────────────────────── Chrome Browser ───────────────────────────────┐
│                                                                               │
│  [content/overlay/]  React 18 overlay panel                                  │
│    Recognized words, sentence history, peer transcript; resizable/draggable  │
│    ↓                                                                          │
│  [injected/inject.ts]  getUserMedia override                                 │
│    Synthesized speech (server TTS or Web Speech API)                         │
│    → mixed into the meeting's microphone stream via AudioContext             │
│    → other participants hear a voice. No action required on their end.       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Why These Design Decisions?

### Two-world bridge (frontend)
Chrome's Manifest V3 content scripts run in an isolated sandbox — MediaPipe Holistic cannot be initialized there. The solution: inject a second script into the **page's MAIN world** to run MediaPipe, then relay landmark data back to the content script via `postMessage`. A custom Trusted Types policy ensures CSP compliance on hardened meeting pages.

### Bidirectional GRU over Transformer (model)
The WLASL dataset provides only ~10 labeled video samples per class. A Transformer (447K params) overfit immediately. The chosen `SignGRU` (558K params, 2-layer bidirectional GRU + temporal attention pooling) converges reliably at this data scale:
- GRU has ~25% fewer recurrent parameters than LSTM for the same hidden size
- Temporal attention learns to ignore padding frames without needing positional encodings
- 4-crop test-time augmentation (TTA) adds +3–5 pp over greedy decoding

### ONNX Runtime (production serving)
Exporting the trained PyTorch checkpoint to ONNX and running it under ONNX Runtime delivers **5.97× faster inference** at identical accuracy. Dynamic INT8 quantization shaves another 8.2% off file size with negligible accuracy impact. The backend warms the model at startup (3 dummy forward passes) so the first real frame doesn't pay a cold-start penalty.

### Adaptive sliding-window stride (latency lever)
When no hand is detected in the frame, the stride between inference calls widens from 5 to 15 frames. This cuts idle CPU load by ~3× while keeping responsiveness instant the moment signing resumes.

### LFU prediction cache (latency lever)
Common windows (frequently signed words, all-zero "no hand" windows) reappear often. A capacity-20 LFU cache keyed by a Blake2b hash of quantized landmark values (rounded to 2 decimal places) turns repeated windows into O(1) lookups. Cache hit rates are exposed at `/metrics`.

### Privacy-by-design
Every networking decision was made to minimize data exposure. MediaPipe runs **in the browser**; only 144 floating-point numbers per frame are transmitted. No audio, no video, no PII ever leave the device. The WebSocket payload is `{ landmarks: float[144], session_id: uuid, timestamp: epoch_ms }`.

---

## Model Details

### Architecture — `SignGRU`

```
Input: (batch, seq_len=100, input_dim=292)
  └─ 146 position features  (normalized hand + pose landmarks)
  └─ 146 velocity features  (frame-to-frame deltas)

→ Linear(292 → 256) + LayerNorm + ReLU          [input projection]
→ BiGRU(256 hidden, 2 layers, dropout=0.4)       [temporal encoding]
→ Temporal attention pooling (padding-masked)    [aggregation]
→ Linear(512 → 100)                              [classification head]

Parameters: 558,309
```

### Feature normalization (invariance across signers)
- **Hands**: center on wrist landmark; scale by wrist-to-middle-knuckle distance
- **Pose**: center on shoulder midpoint; scale by shoulder width
- **Hand presence bits**: 1.0 if hand detected, 0.0 for all-zero frames
- Result: predictions generalize across different hand sizes and camera distances

### Training
- **Dataset**: WLASL — top-100 highest-frequency glosses; ~10 samples/class
- **Augmentation**: Mixup (α=0.2), Stochastic Weight Averaging (SWA)
- **Optimizer**: Adam with cosine LR schedule
- **Evaluation**: 4-crop TTA (left/right temporal crops × original/mirrored)

### Benchmark — ONNX Runtime vs PyTorch (CPU, 1000 runs)

| Runtime | Median latency | p95 latency | Speedup |
|---|---|---|---|
| PyTorch | 15.41 ms | 18.3 ms | 1× |
| ONNX Runtime | 2.57 ms | 3.1 ms | **5.97×** |
| ONNX INT8 | 2.61 ms | 3.3 ms | 5.90× |

### Accuracy (test set, 100-class)

| Metric | Score |
|---|---|
| Top-1 (greedy) | 59.8% |
| Top-1 (4-crop TTA) | **63.18%** |
| Top-3 (4-crop TTA) | **81.59%** |
| Top-5 (4-crop TTA) | **86.07%** |

---

## Repository Layout

This is a monorepo. Each sub-project has its own dependencies and runs independently.

```
ASL-to-Speech/
├── backend/                    # FastAPI inference server (Python 3.11)
│   ├── app/
│   │   ├── main.py             # FastAPI app; lifespan loads ONNX model; /health /ready /metrics /tts
│   │   ├── websocket.py        # WebSocket handler; validates 144 landmarks; dispatches to buffer
│   │   ├── inference.py        # SignClassifier: ONNX load + warm-up + feature pipeline + predict
│   │   ├── cache.py            # LFU cache (capacity 20) keyed by Blake2b window fingerprint
│   │   ├── tts.py              # Offline TTS (pyttsx3 / espeak-ng) → WAV → /tts endpoint
│   │   ├── metrics.py          # Active sessions, latency histogram, throughput, cache hit-rate
│   │   ├── session/
│   │   │   └── buffer.py       # Redis-backed 100-frame sliding window + adaptive stride
│   │   └── config.py           # Env-var settings (REDIS_URL, MODEL_PATH, CONFIDENCE_THRESHOLD…)
│   ├── scripts/
│   │   ├── load_test.py        # Concurrency sweep — max sessions within p95 latency budget
│   │   └── uptime_monitor.py   # Health polling with webhook alerting
│   ├── Dockerfile              # Python 3.11-slim + espeak-ng; runs gunicorn
│   ├── docker-compose.yml      # FastAPI + Redis; one command to boot
│   └── fly.toml                # Fly.io config (1 shared CPU, 1 GB RAM, auto-scale to zero)
│
├── frontend/                   # Chrome Extension MV3 (TypeScript + React 18 + esbuild)
│   ├── src/
│   │   ├── content/
│   │   │   ├── index.tsx       # Entry: mounts overlay, starts HandTracker + WSClient
│   │   │   ├── overlay/        # React overlay (word display, sentence history, transcript panel)
│   │   │   ├── mediapipe/      # HandTracker: camera acquisition + rAF loop + bridge messaging
│   │   │   └── websocket/      # WSClient: sends landmarks, receives predictions, exponential backoff
│   │   ├── injected/
│   │   │   ├── mediapipe-bridge.ts   # MAIN-world: runs MediaPipe Holistic, posts 144 floats/frame
│   │   │   └── inject.ts             # getUserMedia override: mixes TTS audio into mic stream
│   │   ├── popup/              # Extension toolbar popup (toggle, voice, font size)
│   │   ├── background/         # MV3 service worker
│   │   └── lib/                # Shared types, chrome.storage helpers, messaging utilities
│   ├── public/
│   │   ├── manifest.json       # MV3 manifest: permissions, host matches, content scripts
│   │   └── mediapipe/          # Bundled MediaPipe Holistic WASM (no CDN dependency)
│   └── esbuild.config.mjs      # 4 entry points: content, popup, background, injected
│
├── model-training/             # PyTorch training pipeline (Python 3.11)
│   ├── src/
│   │   ├── models/
│   │   │   ├── lstm.py         # SignGRU (production) + SignLSTM
│   │   │   └── transformer.py  # SignTransformer (experimental; ablation showed GRU wins at this data scale)
│   │   ├── preprocessing/
│   │   │   ├── extract_landmarks.py  # WLASL videos → 144-dim landmark sequences via MediaPipe
│   │   │   └── build_dataset.py      # Filter to 100 glosses; normalize; train/val/test split
│   │   ├── train.py            # Training loop: mixup, SWA, cosine LR, TTA evaluation
│   │   ├── evaluate.py         # Top-k accuracy + per-class confusion matrix
│   │   ├── benchmark.py        # PyTorch vs ONNX Runtime latency comparison
│   │   ├── quantize.py         # Dynamic INT8 quantization of the ONNX model
│   │   ├── compare_archs.py    # GRU vs Transformer cost/accuracy ablation
│   │   └── export_onnx.py      # PyTorch checkpoint → .onnx + label_map.json
│   ├── MODEL_CARD.md           # Intended use, training data, metrics, known limitations
│   └── TRAINING_LOG.md         # Week-by-week experiment log and architecture decisions
│
├── CODEBASE_TOUR.md            # Step-by-step walkthrough of one frame end-to-end
└── Makefile                    # Convenience shortcuts for all three sub-projects
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Extension** | Chrome MV3, TypeScript, React 18, esbuild, Tailwind CSS |
| **Computer Vision** | MediaPipe Holistic (WASM, bundled locally) |
| **Backend** | Python 3.11, FastAPI, WebSockets, Gunicorn |
| **Cache / Session** | Redis (sliding window buffer, per-session isolation) |
| **ML Framework** | PyTorch (training), ONNX Runtime (serving, 5.97× faster) |
| **TTS** | pyttsx3 / espeak-ng (offline, no third-party API) |
| **Deployment** | Docker, docker-compose, Fly.io (auto-scale to zero) |
| **CI** | GitHub Actions |

---

## Quick Start

Each part runs independently.

```sh
# 1. Backend — FastAPI + Redis (Docker required)
make backend-docker
# Server is now at ws://localhost:8000/ws

# 2. Frontend — Chrome Extension
make frontend-install
make frontend-dev        # watch mode: rebuilds dist/ on every save
# Open chrome://extensions → enable Developer mode → Load unpacked → select frontend/dist/

# 3. Model training (optional — pre-trained .onnx is in backend/models/)
cd model-training
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.preprocessing.extract_landmarks       # extract from WLASL videos
python -m src.preprocessing.build_dataset          # build train/val/test splits
python -m src.train                                 # train SignGRU
python -m src.export_onnx                          # export to .onnx
cp exports/sign_gru.onnx backend/models/           # deploy to backend
```

Run `make help` from the repo root for all available commands.

---

## Backend Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks model + Redis) |
| `GET` | `/metrics` | Active sessions, latency stats, cache hit-rate |
| `WebSocket` | `/ws` | Main inference endpoint (receives landmarks, returns predictions) |
| `POST` | `/tts` | Synthesize text → WAV (offline, espeak-ng) |
| `GET` | `/voices` | List available TTS voices |

---

## Supported Platforms

| Platform | URL pattern |
|---|---|
| Google Meet | `meet.google.com/*` |
| Zoom Web | `app.zoom.us/*` |
| Facebook Messenger | `www.messenger.com/*` |
| Microsoft Teams Web | `teams.microsoft.com/*` |

---

## Further Reading

- [CODEBASE_TOUR.md](CODEBASE_TOUR.md) — End-to-end walkthrough of how one camera frame becomes a spoken word
- [model-training/MODEL_CARD.md](model-training/MODEL_CARD.md) — Model card: intended use, data, metrics, limitations
- [model-training/TRAINING_LOG.md](model-training/TRAINING_LOG.md) — Experiment log and architecture decision rationale
- [backend/STRESS_TEST.md](backend/STRESS_TEST.md) — Latency/capacity stress-test procedure and results
- [backend/README.md](backend/README.md) — Backend local dev and Docker setup
- [frontend/README.md](frontend/README.md) — Extension build steps and loading in Chrome
- [model-training/README.md](model-training/README.md) — Full training pipeline walkthrough
