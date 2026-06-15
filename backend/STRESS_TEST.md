# Backend stress test & latency report (Week 5)

Goal: keep end-to-end response latency **< 200 ms** and find the **maximum number
of concurrent sessions** the server sustains before p95 latency degrades past that
budget.

## Latency levers in place (Week 5)

| Lever | Where | Effect |
|-------|-------|--------|
| Adaptive sliding-window stride | `app/session/buffer.py`, `INFERENCE_STRIDE` / `IDLE_INFERENCE_STRIDE` | Inference runs every 5 frames while signing, every 15 when no hand is detected — idle sessions stop burning CPU, freeing it for active ones. |
| Prediction cache (LFU, top-N) | `app/cache.py`, `PREDICTION_CACHE_*` | Repeated/near-identical windows (incl. the all-zero "no hand" window) skip feature-build + ONNX entirely. Hit rate visible at `GET /metrics`. |
| ONNX Runtime model | `app/inference.py` | ~2.6 ms median inference vs ~15 ms for native PyTorch (see `model-training/benchmark_results.json`). |
| Model warm at startup | `app/main.py` lifespan + `/ready` | No cold-start penalty on first request. |

## How to run the sweep

Start the full stack (FastAPI + Redis):

```sh
cd backend
docker-compose up --build       # server at ws://localhost:8000/ws
```

From another shell, ramp concurrency and find where p95 crosses the 200 ms budget:

```sh
python scripts/load_test.py --sweep \
    --url ws://localhost:8000/ws \
    --steps 1,5,10,25,50,100,150,200 \
    --frames 200 \
    --p95-budget 200
```

The sweep prints p50/p95/p99 and an errors column per level and reports the
**max concurrent sessions within the p95 budget**. A single fixed-load run:

```sh
python scripts/load_test.py --sessions 50 --frames 200
```

While the sweep runs, watch live server-side numbers:

```sh
curl http://localhost:8000/metrics
# -> active_sessions, avg_latency_ms_60s, inference_throughput_per_s,
#    prediction_cache.hit_rate
```

## Results

> Fill in from a run against the deployed (Redis-backed) server. Record the host
> so numbers are comparable across runs.

**Environment:** _e.g. Render Standard / 1 vCPU, 2 GB, Redis 7_
**Date:** _YYYY-MM-DD_   **Command:** _exact `--steps` / `--frames` used_

| Concurrent sessions | p50 (ms) | p95 (ms) | p99 (ms) | errors | within 200 ms? |
|---------------------|----------|----------|----------|--------|----------------|
| 1 | | | | | |
| 5 | | | | | |
| 10 | | | | | |
| 25 | | | | | |
| 50 | | | | | |
| 100 | | | | | |

**Max concurrent sessions within 200 ms p95:** _N_

### Notes / interpretation

- The dummy frames in `load_test.py` are random, so the prediction cache hit rate
  during the sweep is low by design (worst case). With real signers, repeated
  windows raise the hit rate and lower observed latency.
- If p95 degrades early, the first knobs are: gunicorn worker count (`-w`),
  `MAX_CONNECTIONS`, and `INFERENCE_STRIDE` (larger stride = fewer inferences).
- `MAX_CONNECTIONS` (default 100) hard-caps connections; beyond it the server
  closes new sockets with code 1008 ("Server at capacity") rather than degrading.
