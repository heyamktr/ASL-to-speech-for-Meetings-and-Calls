"""Adaptive sliding-window stride tests (uses the fake_redis autouse fixture)."""

import pytest

from app.config import settings
from app.session import buffer

FRAME = [0.1] * 144


async def _fill(session_id: str, n: int, has_hand: bool = True):
    results = []
    for _ in range(n):
        results.append(await buffer.add_frame(session_id, FRAME, has_hand=has_hand))
    return results


@pytest.mark.asyncio
async def test_no_inference_before_window_full() -> None:
    results = await _fill("s1", buffer.FRAME_WINDOW - 1)
    assert all(r is None for r in results)


@pytest.mark.asyncio
async def test_active_stride_runs_every_n_frames() -> None:
    # Fill the window first.
    await _fill("s2", buffer.FRAME_WINDOW)
    # Now each `inference_stride` additional frames should yield one window.
    fired = await _fill("s2", settings.inference_stride * 2)
    windows = [r for r in fired if r is not None]
    assert len(windows) == 2
    assert all(len(w) == buffer.FRAME_WINDOW for w in windows)


@pytest.mark.asyncio
async def test_idle_stride_is_larger_than_active() -> None:
    # Idle sessions should fire LESS often than active ones over the same frames.
    await _fill("active", buffer.FRAME_WINDOW)
    await _fill("idle", buffer.FRAME_WINDOW)

    n = settings.idle_inference_stride
    active_fires = sum(r is not None for r in await _fill("active", n, has_hand=True))
    idle_fires = sum(r is not None for r in await _fill("idle", n, has_hand=False))

    assert active_fires > idle_fires
    assert idle_fires == 1  # exactly one fire after idle_inference_stride frames
