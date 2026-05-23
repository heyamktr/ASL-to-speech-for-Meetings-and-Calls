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
- WebSocket inference endpoint: `ws://localhost:8000/ws`
- Shared WebSocket contract: `../docs/websocket-contract.md`

## Deploying the trained model

Drop the `.onnx` file produced by `model-training/` into `backend/models/` and point `MODEL_PATH` at it. The directory is gitignored so weights are not committed.

## Tests

```sh
pytest
```
