"""WebSocket handlers for landmark inference messages."""

import logging
from time import perf_counter
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.session.buffer import add_frame, should_emit

router = APIRouter()
logger = logging.getLogger(__name__)

DUMMY_PREDICTION = "hello"
DUMMY_CONFIDENCE = 0.99
LANDMARK_COUNT = 144  # 21 right hand (x,y,z) + 21 left hand (x,y,z) + 6 pose joints (x,y,z)
# = 63 + 63 + 18 = 144


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_message(payload: Any) -> tuple[list[float], str, int | float]:
    if not isinstance(payload, dict):
        raise ValueError("message must be a JSON object")

    landmarks = payload.get("landmarks")
    session_id = payload.get("session_id")
    timestamp = payload.get("timestamp")

    if not isinstance(landmarks, list) or len(landmarks) != LANDMARK_COUNT:
        raise ValueError("landmarks must be a list of exactly 144 numbers")
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
        t0 = perf_counter()

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

        t1 = perf_counter()  # receive done

        response: dict[str, Any] = {"timestamp": timestamp}

        frames = await add_frame(session_id, landmarks)
        if frames is not None:
            # Full 30-frame window ready — run inference (TODO: replace with SignClassifier)
            prediction = DUMMY_PREDICTION
            confidence = DUMMY_CONFIDENCE
            if await should_emit(session_id, prediction):
                response["prediction"] = prediction
                response["confidence"] = confidence

        t2 = perf_counter()  # inference done

        await websocket.send_json(response)

        t3 = perf_counter()  # send done

        logger.info(
            "latency session_id=%s recv_ms=%.2f infer_ms=%.2f send_ms=%.2f total_ms=%.2f",
            session_id,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            (t3 - t0) * 1000,
        )


@router.websocket("/ws")
async def predict(websocket: WebSocket) -> None:
    await _predict_loop(websocket)


@router.websocket("/ws/predict")
async def predict_legacy(websocket: WebSocket) -> None:
    await _predict_loop(websocket)
