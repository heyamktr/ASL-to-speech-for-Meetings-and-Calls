import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.inference import SignClassifier
from app.metrics import metrics
from app.session.buffer import redis_client
from app.websocket import router as websocket_router

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = SignClassifier(
        settings.model_path,
        settings.label_map_path,
        settings.model_seq_len,
        settings.confidence_threshold,
    )
    app.state.ready = True
    yield
    app.state.ready = False
    await redis_client.aclose()


app = FastAPI(title="ASL-to-Speech Inference Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe — returns 503 until the model is loaded and warm."""
    if not getattr(app.state, "ready", False):
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready"}


@app.get("/metrics")
async def get_metrics():
    """Active sessions, avg latency (last 60 s), and inference throughput."""
    stats = await metrics.snapshot()
    session_keys = await redis_client.keys("session:*:frames")
    stats["active_sessions"] = len(session_keys)
    return stats


app.include_router(websocket_router)
