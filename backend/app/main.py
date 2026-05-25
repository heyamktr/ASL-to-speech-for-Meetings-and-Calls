import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.session.buffer import redis_client
from app.websocket import router as websocket_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await redis_client.aclose()


app = FastAPI(title="ASL-to-Speech Inference Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(websocket_router)
