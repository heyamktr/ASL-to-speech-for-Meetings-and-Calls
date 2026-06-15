import json

from redis import asyncio as aioredis

from app.config import settings

FRAME_WINDOW = 100
SMOOTHING_K = 3
SESSION_TTL = 300

redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)

BUFFER_KEY = lambda sid: f"session:{sid}:frames"
SMOOTHING_KEY = lambda sid: f"session:{sid}:smooth"
SINCE_KEY = lambda sid: f"session:{sid}:since"  # frames since last inference


async def add_frame(
    session_id: str, landmarks: list[float], has_hand: bool = True
) -> list[list[float]] | None:
    """Append one frame to the session buffer.

    Returns the full window when the buffer is full AND enough frames have
    elapsed since the last inference. The stride is *adaptive*: while a hand is
    visible we run every ``inference_stride`` frames; when no hand is detected we
    back off to ``idle_inference_stride`` so idle sessions don't waste CPU and
    overall latency under load stays low (Week 5 latency lever #1).
    """
    pipe = redis_client.pipeline()
    pipe.rpush(BUFFER_KEY(session_id), json.dumps(landmarks))
    pipe.ltrim(BUFFER_KEY(session_id), -FRAME_WINDOW, -1)
    pipe.incr(SINCE_KEY(session_id))
    pipe.expire(BUFFER_KEY(session_id), SESSION_TTL)
    pipe.expire(SINCE_KEY(session_id), SESSION_TTL)
    pipe.llen(BUFFER_KEY(session_id))
    results = await pipe.execute()

    since: int = results[2]
    length: int = results[5]

    stride = settings.inference_stride if has_hand else settings.idle_inference_stride
    stride = max(1, stride)

    if length < FRAME_WINDOW or since < stride:
        return None

    # Reset the stride counter and return the window for inference.
    await redis_client.set(SINCE_KEY(session_id), 0, ex=SESSION_TTL)
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
