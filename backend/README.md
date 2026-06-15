# Backend — FastAPI inference server

Receives 63-dim hand landmarks per frame over a WebSocket, buffers them per session, runs the ONNX-exported LSTM classifier, and streams predicted words back to the browser extension.

## Stack

- Python 3.11
- FastAPI + Uvicorn (Gunicorn in production)
- Redis (per-session sliding window buffers)
- ONNX Runtime (model inference)

## Run locally (without Docker)

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env       # adjust if needed
uvicorn app.main:app --reload
```

You'll need a Redis instance running locally. Easiest way:

```sh
docker run --rm -p 6379:6379 redis:7-alpine
```

## Run with Docker (FastAPI + Redis together)

```sh
docker-compose up --build
```

Server is then available at `http://localhost:8000`.

- Health check: `GET /health`
- Readiness probe (503 until model warm): `GET /ready`
- Monitoring (sessions, latency, throughput, cache hit-rate): `GET /metrics`
- WebSocket inference endpoint: `ws://localhost:8000/ws`
- Text-to-speech: `POST /tts` `{ "text": "...", "rate": 170, "voice": "..." }` → `audio/wav`
- Available server voices: `GET /voices`
- Shared WebSocket contract: `../docs/websocket-contract.md`

## Week 5 — latency, capacity, ops

Three latency levers (all env-tunable, see `.env.example`):

1. **Adaptive sliding-window stride** — inference runs every `INFERENCE_STRIDE`
   frames while a hand is visible, backing off to `IDLE_INFERENCE_STRIDE` when no
   hand is detected (`app/session/buffer.py`).
2. **Prediction cache** — an LFU cache (`PREDICTION_CACHE_SIZE`, default 20) keyed
   by a quantized fingerprint of the window; repeated windows bypass the ONNX run
   (`app/cache.py`). Hit-rate is reported at `GET /metrics`.
3. **TTS endpoint** — offline synthesis via `pyttsx3` (espeak-ng in Docker),
   returns WAV for the frontend to inject (`app/tts.py`).

```sh
# Capacity sweep: ramp concurrency, find max sessions within a 200 ms p95 budget
python scripts/load_test.py --sweep --steps 1,5,10,25,50,100 --p95-budget 200

# Uptime monitor with webhook alerting (Slack/Discord-compatible)
python scripts/uptime_monitor.py --url http://localhost:8000/health \
    --interval 30 --failures 3 --webhook "$ALERT_WEBHOOK_URL"
```

See `STRESS_TEST.md` for the full stress-test procedure and results template.

## Deploying the trained model

Drop the `.onnx` file produced by `model-training/` into `backend/models/` and point `MODEL_PATH` at it. The directory is gitignored so weights are not committed.

## Tests

```sh
pytest
```
