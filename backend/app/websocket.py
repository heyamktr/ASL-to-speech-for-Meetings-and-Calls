"""WebSocket handlers for landmark inference messages."""

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)

DUMMY_PREDICTION = "hello"
DUMMY_CONFIDENCE = 0.99
LANDMARK_COUNT = 258  # 33 pose landmarks (x,y,z,visibility) + 21 left hand (x,y,z) + 21 right hand (x,y,z)
# = (33 × 4) + (21 × 3) + (21 × 3) = 132 + 63 + 63 = 258


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_message(payload: Any) -> tuple[list[float], str, int | float]:
    if not isinstance(payload, dict):
        raise ValueError("message must be a JSON object")

    landmarks = payload.get("landmarks")
    session_id = payload.get("session_id")
    timestamp = payload.get("timestamp")

    if not isinstance(landmarks, list) or len(landmarks) != LANDMARK_COUNT:
        raise ValueError("landmarks must be a list of exactly 258 numbers")
    if not all(_is_number(value) for value in landmarks):
        raise ValueError("landmarks must contain only numbers")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    if not _is_number(timestamp):
        raise ValueError("timestamp must be a number")

    return [float(value) for value in landmarks], session_id, timestamp


async def _predict_loop(websocket: WebSocket) -> None:
    await websocket.accept()

    while True:
        start = perf_counter()

        try:
            payload = await websocket.receive_json()
            landmarks, session_id, timestamp = _validate_message(payload)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
            break
        except ValueError as exc:
            logger.warning("Invalid WebSocket message: %s", exc)
            await websocket.send_json({"error": str(exc)})
            continue

        response = {
            "prediction": DUMMY_PREDICTION,
            "confidence": DUMMY_CONFIDENCE,
            "timestamp": timestamp,
        }
        await websocket.send_json(response)

        latency_ms = (perf_counter() - start) * 1000
        logger.info(
            "prediction_sent session_id=%s frame_values=%d latency_ms=%.2f",
            session_id,
            len(landmarks),
            latency_ms,
        )


@router.websocket("/ws")
async def predict(websocket: WebSocket) -> None:
    await _predict_loop(websocket)


@router.websocket("/ws/predict")
async def predict_legacy(websocket: WebSocket) -> None:
    await _predict_loop(websocket)
