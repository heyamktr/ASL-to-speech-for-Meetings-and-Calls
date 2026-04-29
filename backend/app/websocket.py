"""WebSocket handler — receives landmark frames and returns predicted words."""

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/predict")
async def predict(websocket: WebSocket) -> None:
    await websocket.accept()
    # TODO: per-frame loop — receive 63 landmark numbers, push to session buffer,
    #       run inference on each new sliding window, send predicted word back.
    await websocket.close()
