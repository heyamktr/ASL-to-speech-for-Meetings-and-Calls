import json
from redis import asyncio as aioredis

from app.config import settings

FRAME_WINDOW = 100
SMOOTHING_K = 3 # same prediction must be made in K consecutive frames to be considered valid

redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)

BUFFER_KEY = lambda sid: f"session:{sid}:frames"
SMOOTHING_KEY = lambda sid: f"session:{sid}:smooth"

async def add_frame(session_id: str, landmarks: list[float]) -> list[list[float]] | None:
    """
    Append one frame to the session buffer.
    Returns the full 30-frame window when ready, otherwise None
    """

    key = BUFFER_KEY(session_id)

    # Append serialized frame
    await redis_client.rpush(key, json.dumps(landmarks))

    # Keep only the last 30 frames (sliding window)
    await redis_client.ltrim(key, -FRAME_WINDOW, -1)

    # Set TTL so abandoned sessions clean themselves up
    await redis_client.expire(key, 300) # 5 minutes

    # Only run inference once we have a full window
    length = await redis_client.llen(key)
    if length < FRAME_WINDOW:
        return None

    raw_frames = await redis_client.lrange(key, 0, -1)
    return [json.loads(frame) for frame in raw_frames]

async def should_emit(session_id: str, prediction: str) -> bool:
    """
    Smoothing gate: the same prediction must appear SMOOTHING_K times
    consecutively before we send it to the frontend.
    Return True only when the threshold is crossed
    """

    key = SMOOTHING_KEY(session_id)

    # Get current smoothing state
    raw = await redis_client.get(key)
    state = json.loads(raw) if raw else {"last": None, "count": 0}

    if prediction == state["last"]:
        state["count"] += 1
    else:
        state["last"] = prediction
        state["count"] = 1

    await redis_client.set(key, json.dumps(state), ex=300)

    return state["count"] >= SMOOTHING_K
