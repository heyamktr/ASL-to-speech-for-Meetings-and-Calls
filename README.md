# ASL-to-Speech

A Chrome extension that translates ASL (American Sign Language) hand gestures into voice and text in real time during video calls — Google Meet, Zoom Web, Messenger, and Microsoft Teams Web.

The extension uses MediaPipe Hands in the browser to extract 21 hand landmarks per frame, sends those 63 numbers to a FastAPI server, runs an LSTM classifier (exported to ONNX) on the server, and returns predicted words. Recognized words are shown in an overlay panel and (Phase 2) synthesized into audio that's mixed into the meeting's microphone stream so other participants hear a natural voice.

## Repository layout

This is a monorepo with three independent parts. Each has its own dependencies and runs independently of the others.

```
ASL-to-Speech/
├── .gitignore                          # OS files, editor cruft (.DS_Store, .vscode/, etc.)
├── .editorconfig                       # consistent indentation across editors
├── README.md                           # this file
├── Makefile                            # convenience shortcuts: backend-dev, frontend-build, train, ...
│
├── backend/                            # FastAPI inference server (Python 3.11)
│   ├── .gitignore                      # __pycache__, .venv, .env, models/*.onnx
│   ├── README.md                       # how to run locally + with Docker
│   ├── requirements.txt                # fastapi, uvicorn, gunicorn, websockets, redis, onnxruntime, numpy
│   ├── requirements-dev.txt            # pytest, ruff, black
│   ├── pyproject.toml                  # ruff/black config
│   ├── Dockerfile                      # production container image
│   ├── docker-compose.yml              # FastAPI + Redis, one command to boot the stack
│   ├── .env.example                    # template (REDIS_URL, MODEL_PATH, ...)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI entry — app instance, route registration
│   │   ├── websocket.py                # WebSocket handler — receives landmarks, returns words
│   │   ├── inference.py                # ONNX Runtime model load + predict
│   │   ├── session.py                  # per-session sliding window buffer (Redis-backed)
│   │   └── config.py                   # env vars / settings loading
│   ├── models/                         # deployed .onnx weight files (gitignored, large)
│   └── tests/
│
├── frontend/                           # Chrome Extension MV3 (React + TypeScript + esbuild)
│   ├── .gitignore                      # node_modules, dist
│   ├── README.md                       # build steps + how to load the unpacked extension in Chrome
│   ├── package.json
│   ├── tsconfig.json                   # TypeScript compiler config
│   ├── esbuild.config.mjs              # bundler — multiple entry points (content, popup, bg, injected)
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .eslintrc.json
│   ├── .prettierrc
│   ├── public/
│   │   ├── manifest.json               # MV3 manifest — permissions, content scripts, host matches
│   │   └── icons/                      # extension icons (16/48/128 px) — placeholders for now
│   ├── src/
│   │   ├── background/
│   │   │   └── service-worker.ts       # MV3 background service worker
│   │   ├── content/                    # scripts injected into Meet/Zoom/Teams pages
│   │   │   ├── index.tsx               # entry — mounts overlay, starts MediaPipe loop
│   │   │   ├── overlay/                # React sidebar UI
│   │   │   │   └── App.tsx             # top-level overlay component
│   │   │   ├── mediapipe/              # MediaPipe Hands wrapper — emits 63 landmark numbers per frame
│   │   │   │   └── index.ts
│   │   │   └── websocket/              # WS client — sends landmarks, receives words
│   │   │       └── client.ts
│   │   ├── popup/                      # extension popup (toolbar icon click)
│   │   │   ├── index.html
│   │   │   ├── index.tsx               # popup entry
│   │   │   └── Popup.tsx               # popup UI — toggle, theme, font size, voice
│   │   ├── injected/                   # MAIN-world script (page context, NOT extension context)
│   │   │   └── inject.ts               # overrides getUserMedia to inject TTS audio into mic stream
│   │   ├── lib/
│   │   │   ├── storage.ts              # chrome.storage.sync helpers
│   │   │   ├── messaging.ts            # content ↔ background message passing
│   │   │   └── types.ts                # shared TypeScript types
│   │   └── styles/
│   │       └── globals.css             # Tailwind directives
│   └── dist/                           # build output (gitignored) — load this directory in Chrome
│
├── model-training/                     # PyTorch training pipeline (Python 3.11)
│   ├── .gitignore                      # data/, checkpoints/, .venv, __pycache__
│   ├── README.md                       # download WLASL, preprocess, train, export to ONNX
│   ├── requirements.txt                # torch, tensorflow, mediapipe, opencv-python, numpy, pandas, jupyter
│   ├── pyproject.toml                  # ruff/black config
│   ├── data/                           # WLASL videos + extracted landmarks (gitignored — huge)
│   │   ├── raw/                        # downloaded WLASL videos
│   │   └── processed/                  # extracted landmark .npy files
│   ├── notebooks/
│   │   └── 01_data_exploration.ipynb   # exploratory analysis on the dataset
│   ├── src/
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── extract_landmarks.py    # video frames → 63-dim landmarks via MediaPipe
│   │   │   └── build_dataset.py        # filter words, train/val/test split
│   │   ├── models/                     # Python classes defining architectures (NOT weight files)
│   │   │   ├── lstm.py                 # the LSTM classifier
│   │   │   └── transformer.py          # experimental — Week 3 alternative
│   │   ├── train.py                    # training loop
│   │   ├── evaluate.py                 # accuracy / confusion matrix on the test split
│   │   └── export_onnx.py              # PyTorch checkpoint → .onnx file for the backend
│   ├── checkpoints/                    # large .pt training checkpoints (gitignored)
│   └── exports/                        # final .onnx files — copy to backend/models/ for serving
│
└── .github/
    └── workflows/                      # GitHub Actions CI (placeholder — add when first PRs land)
```

> **Note on `models/` appearing twice.** `backend/models/` stores the deployed `.onnx` weight files served at runtime. `model-training/src/models/` stores the Python classes that define the model architectures (`SignLSTM`, `SignTransformer`). One is data, the other is code.

## Quick start

Each part runs independently. See the README inside each folder for detailed instructions.

```sh
# Backend (FastAPI + Redis via Docker)
make backend-docker

# Frontend (Chrome extension)
make frontend-install
make frontend-dev          # watch mode — rebuilds dist/ on save
# then load frontend/dist/ as an unpacked extension in chrome://extensions

# Model training
cd model-training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

Run `make help` from the root to see all available shortcut commands.

## Architecture

```
Browser (Chrome Extension)
         ↓
MediaPipe Hands — runs in browser, never on server
         ↓
63 landmark numbers per frame — the only thing sent over the network
         ↓
WebSocket → FastAPI Server (Dockerized, cloud deployed)
         ↓
LSTM / ONNX Inference — GPU accelerated, model warm at startup
         ↓
Sliding Window + Smoothing + Majority Vote
         ↓
Predicted Word → WebSocket back to browser
         ↓
Overlay UI (sentence builder + peer transcript panel)
         ↓
TTS Endpoint → AudioContext → getUserMedia override → Meet mic stream
         ↓
Other participants hear synthetic voice — no action required on their end
```
