import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.cache import PredictionCache
from app.config import settings
from app.inference import SignClassifier
from app.metrics import metrics
from app.session.buffer import redis_client
from app.tts import TTSRequest, TTSUnavailable, list_voices, synthesize
from app.websocket import router as websocket_router

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache = (
        PredictionCache(
            capacity=settings.prediction_cache_size,
            decimals=settings.prediction_cache_decimals,
        )
        if settings.prediction_cache_enabled
        else None
    )
    app.state.cache = cache
    app.state.classifier = SignClassifier(
        settings.model_path,
        settings.label_map_path,
        settings.model_seq_len,
        settings.confidence_threshold,
        cache=cache,
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
    cache = getattr(app.state, "cache", None)
    stats["prediction_cache"] = cache.stats() if cache else {"enabled": False}
    return stats


@app.post("/tts")
async def tts(req: TTSRequest):
    """Synthesize a sentence to speech and return WAV audio.

    Returns 503 if TTS is disabled or no speech engine is available, 413 if the
    text is too long. The browser can play/inject the returned audio/wav blob.
    """
    if not settings.tts_enabled:
        raise HTTPException(status_code=503, detail="TTS is disabled")
    if len(req.text) > settings.tts_max_chars:
        raise HTTPException(
            status_code=413,
            detail=f"text exceeds {settings.tts_max_chars} characters",
        )

    rate = req.rate if req.rate is not None else settings.tts_rate
    try:
        audio = await run_in_threadpool(synthesize, req.text, rate, req.voice)
    except TTSUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"TTS unavailable: {exc}") from exc

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )


@app.get("/voices")
async def voices():
    """List speech-synthesis voices available on the server (for the popup UI)."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=503, detail="TTS is disabled")
    try:
        return {"voices": await run_in_threadpool(list_voices)}
    except TTSUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"TTS unavailable: {exc}") from exc


app.include_router(websocket_router)
