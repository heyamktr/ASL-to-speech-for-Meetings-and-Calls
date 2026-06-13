"""WebSocket handlers for landmark inference messages."""

import logging
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.metrics import metrics
from app.session.buffer import add_frame, should_emit

router = APIRouter()
logger = logging.getLogger(__name__)

LANDMARK_COUNT = 144  # 63 right hand + 63 left hand + 18 pose


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
    if not await metrics.try_connect(settings.max_connections):
        await websocket.close(code=1008, reason="Server at capacity")
        return

    try:
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

            t1 = perf_counter()

            response: dict[str, Any] = {"timestamp": timestamp}
            ran_inference = False

            frames = await add_frame(session_id, landmarks)
            if frames is not None:
                classifier = websocket.app.state.classifier
                prediction, confidence = classifier.predict(
                    np.asarray(frames, dtype=np.float32)
                )
                ran_inference = True

                if await should_emit(session_id, prediction):
                    response["prediction"] = prediction
                    response["confidence"] = confidence

            t2 = perf_counter()

            await websocket.send_json(response)

            t3 = perf_counter()
            total_ms = (t3 - t0) * 1000
            await metrics.record(total_ms, ran_inference)

            logger.info(
                "latency session_id=%s recv_ms=%.2f infer_ms=%.2f send_ms=%.2f total_ms=%.2f",
                session_id,
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
                total_ms,
            )
    finally:
        await metrics.disconnect()


@router.websocket("/ws")
async def predict(websocket: WebSocket) -> None:
    await _predict_loop(websocket)


@router.websocket("/ws/predict")
async def predict_legacy(websocket: WebSocket) -> None:
    await _predict_loop(websocket)
