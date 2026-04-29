"""Per-session sliding window buffer backed by Redis.

Each connected user gets an isolated 30-frame deque so concurrent sessions
don't share application memory.
"""


class SessionBuffer:
    def __init__(self, session_id: str, redis_url: str) -> None:
        self.session_id = session_id
        self.redis_url = redis_url
        # TODO: connect to Redis

    async def push_frame(self, landmarks: list[float]) -> None:
        """Append a single frame's 63 landmark numbers to this session's buffer."""
        # TODO: implement (LPUSH + LTRIM to keep window size = 30)
        raise NotImplementedError

    async def get_window(self) -> list[list[float]]:
        """Return the most recent 30 frames as a list of length-63 lists."""
        # TODO: implement
        raise NotImplementedError
