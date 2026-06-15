# ASL-to-Speech

A Chrome extension that translates ASL (American Sign Language) signs into voice and text in real time during video calls — Google Meet, Zoom Web, Messenger, and Microsoft Teams Web.

The extension runs **MediaPipe Holistic** in the browser to extract **144 numbers per frame** (21 right-hand + 21 left-hand landmarks × x/y/z, plus 6 upper-body pose joints × x/y/z), and sends only those numbers — never video — over a WebSocket to a FastAPI server. The server buffers frames into a sliding window, runs a **bidirectional GRU classifier exported to ONNX** (`SignGRU`, 100-word WLASL vocabulary), and returns the predicted word. Recognized words appear in an overlay panel and can be synthesized to audio via the server's TTS endpoint and mixed into the meeting's microphone stream so other participants hear a natural voice.

## Repository layout

This is a monorepo with three independent parts. Each has its own dependencies and runs independently of the others.

```
ASL-to-Speech/
├── .gitignore                          # OS files, editor cruft (.DS_Store, .vscode/, etc.)
├── .editorconfig                       # consistent indentation across editors
├── README.md                           # this file
├── CODEBASE_TOUR.md                    # end-to-end walkthrough of how a frame flows through the system
├── Makefile                            # convenience shortcuts: backend-dev, frontend-build, train, ...
│
├── backend/                            # FastAPI inference server (Python 3.11)
│   ├── .gitignore                      # __pycache__, .venv, .env, models/*.onnx
│   ├── README.md                       # how to run locally + with Docker
│   ├── STRESS_TEST.md                  # latency/capacity stress-test procedure + results template
│   ├── requirements.txt                # fastapi, uvicorn, gunicorn, websockets, redis, onnxruntime, numpy, pyttsx3
│   ├── requirements-dev.txt            # pytest, ruff, black
│   ├── Dockerfile                      # production container image (includes espeak-ng for TTS)
│   ├── docker-compose.yml              # FastAPI + Redis, one command to boot the stack
│   ├── .env.example                    # template (REDIS_URL, MODEL_PATH, CONFIDENCE_THRESHOLD, ...)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI entry — app instance, lifespan, health/ready/metrics/tts routes
│   │   ├── websocket.py                # WebSocket handler — validates 144 landmarks, runs inference, returns words
│   │   ├── inference.py                # ONNX Runtime model load + feature build + predict (SignClassifier)
│   │   ├── cache.py                    # LFU prediction cache keyed by a quantized window fingerprint
│   │   ├── tts.py                      # offline text-to-speech (pyttsx3 / espeak-ng) → WAV
│   │   ├── metrics.py                  # active sessions, latency, throughput, cache hit-rate
│   │   ├── session/
│   │   │   └── buffer.py               # per-session sliding-window buffer (Redis-backed, adaptive stride)
│   │   └── config.py                   # env vars / settings loading
│   ├── scripts/
│   │   ├── load_test.py                # concurrency sweep — max sessions within a p95 latency budget
│   │   └── uptime_monitor.py           # health polling with webhook alerting
│   ├── models/                         # deployed .onnx weights + label_map.json (gitignored, large)
│   └── tests/                          # pytest: websocket, buffer, cache, tts
│
├── frontend/                           # Chrome Extension MV3 (React + TypeScript + esbuild)
│   ├── .gitignore                      # node_modules, dist
│   ├── README.md                       # build steps + how to load the unpacked extension in Chrome
│   ├── package.json
│   ├── tsconfig.json                   # TypeScript compiler config
│   ├── esbuild.config.mjs              # bundler — multiple entry points (content, popup, bg, injected)
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── public/
│   │   ├── manifest.json               # MV3 manifest — permissions, content scripts, host matches
│   │   ├── mediapipe/                  # bundled MediaPipe Holistic wasm/assets (served locally, no CDN)
│   │   └── icons/                      # extension icons (16/48/128 px)
│   ├── src/
│   │   ├── background/
│   │   │   └── service-worker.ts       # MV3 background service worker
│   │   ├── content/                    # scripts injected into Meet/Zoom/Teams pages
│   │   │   ├── index.tsx               # entry — mounts overlay, starts tracker + WS client
│   │   │   ├── overlay/                # React overlay UI
│   │   │   │   ├── App.tsx
│   │   │   │   ├── OverlayApp.tsx       # top-level overlay component
│   │   │   │   └── mount.ts             # mount/unmount + word/sentence helpers
│   │   │   ├── mediapipe/              # HandTracker — drives camera + the injected Holistic bridge
│   │   │   │   └── index.ts
│   │   │   └── websocket/              # WS client — sends 144 landmarks, receives predictions
│   │   │       └── client.ts
│   │   ├── popup/                      # extension popup (toolbar icon click)
│   │   │   ├── index.html
│   │   │   ├── index.tsx
│   │   │   └── Popup.tsx               # popup UI — toggle, theme, font size, voice
│   │   ├── injected/                   # MAIN-world scripts (page context, NOT extension context)
│   │   │   ├── mediapipe-bridge.ts      # runs MediaPipe Holistic, posts back 144 landmark numbers/frame
│   │   │   └── inject.ts                # overrides getUserMedia to inject TTS audio into the mic stream
│   │   ├── lib/
│   │   │   ├── storage.ts              # chrome.storage helpers + runtime state
│   │   │   ├── messaging.ts            # content ↔ background message passing
│   │   │   └── types.ts                # shared TypeScript types (LandmarkMessage, PredictionMessage, ...)
│   │   └── styles/
│   │       └── globals.css             # Tailwind directives
│   └── dist/                           # build output (gitignored) — load this directory in Chrome
│
├── model-training/                     # PyTorch training pipeline (Python 3.11)
│   ├── .gitignore                      # data/, checkpoints/, .venv, __pycache__
│   ├── README.md                       # download WLASL, preprocess, train, export to ONNX
│   ├── MODEL_CARD.md                   # model card — data, metrics, intended use, limitations
│   ├── TRAINING_LOG.md                 # experiment log + architecture decision (GRU vs Transformer)
│   ├── requirements.txt                # torch, mediapipe, opencv-python, numpy, pandas, onnx, onnxruntime
│   ├── data/                           # WLASL videos + extracted landmarks (gitignored — huge)
│   │   ├── raw/                        # downloaded WLASL videos
│   │   └── processed/                  # extracted landmark .npy files + train/val/test .npz
│   ├── notebooks/                      # exploratory analysis
│   ├── src/
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── extract_landmarks.py    # video frames → 144-dim hand+pose landmarks via MediaPipe
│   │   │   └── build_dataset.py        # normalize, filter to 100 glosses, train/val/test split
│   │   ├── models/                     # Python classes defining architectures (NOT weight files)
│   │   │   ├── lstm.py                 # SignGRU (production) + SignLSTM
│   │   │   └── transformer.py          # SignTransformer — experimental alternative
│   │   ├── train.py                    # training loop
│   │   ├── evaluate.py                 # accuracy / confusion matrix on the test split
│   │   ├── benchmark.py                # PyTorch vs ONNX latency benchmark
│   │   ├── quantize.py                 # dynamic INT8 quantization of the ONNX model
│   │   ├── compare_archs.py            # GRU vs Transformer cost/accuracy comparison
│   │   └── export_onnx.py              # PyTorch checkpoint → .onnx file for the backend
│   ├── checkpoints/                    # large .pt training checkpoints (gitignored)
│   └── exports/                        # final .onnx files — copy to backend/models/ for serving
│
└── .github/
    └── workflows/                      # GitHub Actions CI
```

> **Note on `models/` appearing twice.** `backend/models/` stores the deployed `.onnx` weight files served at runtime. `model-training/src/models/` stores the Python classes that define the model architectures (`SignGRU`, `SignLSTM`, `SignTransformer`). One is data, the other is code.

## Quick start

Each part runs independently. See the README inside each folder for detailed instructions.

```sh
# Backend (FastAPI + Redis via Docker) — serves ws://localhost:8000/ws
make backend-docker

# Frontend (Chrome extension)
make frontend-install
make frontend-dev          # watch mode — rebuilds dist/ on save
# then load frontend/dist/ as an unpacked extension in chrome://extensions

# Model training
cd model-training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train        # train the GRU
python -m src.export_onnx  # export checkpoint → .onnx, then copy into backend/models/
```

Run `make help` from the root to see all available shortcut commands.
For a step-by-step walkthrough of how one frame flows through the whole system, see [CODEBASE_TOUR.md](CODEBASE_TOUR.md).

## Architecture

```
Browser (Chrome Extension)
         ↓
MediaPipe Holistic — runs in browser, never on server
         ↓
144 landmark numbers per frame (2 hands + 6 pose joints) — the only thing sent over the network
         ↓
WebSocket → FastAPI Server (Dockerized, cloud deployed)
         ↓
Redis sliding window (100 frames) + adaptive stride + prediction cache
         ↓
Bidirectional GRU / ONNX Inference — model warm at startup
         ↓
Confidence threshold + consecutive-frame smoothing
         ↓
Predicted Word → WebSocket back to browser
         ↓
Overlay UI (sentence builder + transcript panel)
         ↓
TTS Endpoint → AudioContext → getUserMedia override → Meet mic stream
         ↓
Other participants hear synthetic voice — no action required on their end
```
