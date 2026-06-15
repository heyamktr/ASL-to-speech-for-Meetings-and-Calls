"""Shared test fixtures.

Replaces the module-level Redis client with an in-memory fake so the WebSocket /
session-buffer tests run without a real Redis server.
"""

import fakeredis.aioredis
import pytest


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Swap the real async Redis client for an in-memory fake everywhere it's used."""
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("app.session.buffer.redis_client", fake)
    # main.py imported redis_client by reference for /metrics — patch it too.
    monkeypatch.setattr("app.main.redis_client", fake, raising=False)
    return fake
