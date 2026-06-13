import json
from redis import asyncio as aioredis

from app.config import settings

FRAME_WINDOW = 100
SMOOTHING_K = 3
SESSION_TTL = 300

redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)

BUFFER_KEY = lambda sid: f"session:{sid}:frames"
SMOOTHING_KEY = lambda sid: f"session:{sid}:smooth"
COUNTER_KEY = lambda sid: f"session:{sid}:counter"


async def add_frame(session_id: str, landmarks: list[float]) -> list[list[float]] | None:
    """
    Append one frame to the session buffer.
    Returns the full window when the buffer is full AND the stride counter fires.
    Inference runs every INFERENCE_STRIDE frames rather than on every frame.
    """
    pipe = redis_client.pipeline()
    pipe.rpush(BUFFER_KEY(session_id), json.dumps(landmarks))
    pipe.ltrim(BUFFER_KEY(session_id), -FRAME_WINDOW, -1)
    pipe.incr(COUNTER_KEY(session_id))
    pipe.expire(BUFFER_KEY(session_id), SESSION_TTL)
    pipe.expire(COUNTER_KEY(session_id), SESSION_TTL)
    pipe.llen(BUFFER_KEY(session_id))
    results = await pipe.execute()

    length: int = results[5]
    counter: int = results[2]

    if length < FRAME_WINDOW or counter % settings.inference_stride != 0:
        return None

    raw_frames = await redis_client.lrange(BUFFER_KEY(session_id), 0, -1)
    return [json.loads(frame) for frame in raw_frames]


async def should_emit(session_id: str, prediction: str) -> bool:
    """
    Smoothing gate: the same prediction must appear SMOOTHING_K times
    consecutively before we send it to the frontend.
    """
    key = SMOOTHING_KEY(session_id)

    raw = await redis_client.get(key)
    state = json.loads(raw) if raw else {"last": None, "count": 0, "emitted": False}

    if prediction == state["last"]:
        state["count"] += 1
    else:
        state["last"] = prediction
        state["count"] = 1
        state["emitted"] = False

    emit = False
    if state["count"] >= SMOOTHING_K and not state.get("emitted", False):
        state["emitted"] = True
        emit = True

    await redis_client.set(key, json.dumps(state), ex=SESSION_TTL)

    return emit
