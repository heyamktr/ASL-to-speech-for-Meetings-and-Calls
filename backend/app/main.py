import logging

from fastapi import FastAPI

from app.websocket import router as websocket_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ASL-to-Speech Inference Server")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(websocket_router)
