"""In-process metrics collector for the /metrics monitoring endpoint."""

import asyncio
from collections import deque
from time import time


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._latency_window: deque[tuple[float, float]] = deque()  # (wall_time, latency_ms)
        self._inference_window: deque[float] = deque()              # wall_time of each inference
        self._inference_count_total: int = 0
        self._active_connections: int = 0

    def _prune(self, now: float) -> None:
        cutoff = now - 60
        while self._latency_window and self._latency_window[0][0] < cutoff:
            self._latency_window.popleft()
        while self._inference_window and self._inference_window[0] < cutoff:
            self._inference_window.popleft()

    async def record(self, latency_ms: float, ran_inference: bool) -> None:
        async with self._lock:
            now = time()
            self._latency_window.append((now, latency_ms))
            if ran_inference:
                self._inference_window.append(now)
                self._inference_count_total += 1
            self._prune(now)

    async def try_connect(self, max_connections: int) -> bool:
        """Atomically check capacity and register the connection."""
        async with self._lock:
            if self._active_connections >= max_connections:
                return False
            self._active_connections += 1
            return True

    async def disconnect(self) -> None:
        async with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    async def snapshot(self) -> dict:
        async with self._lock:
            now = time()
            self._prune(now)
            latencies = [l for _, l in self._latency_window]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            throughput = len(self._inference_window) / 60.0
            return {
                "active_connections": self._active_connections,
                "avg_latency_ms_60s": round(avg_latency, 2),
                "inference_throughput_per_s": round(throughput, 3),
                "inference_count_total": self._inference_count_total,
                "latency_samples_60s": len(latencies),
            }


metrics = MetricsCollector()
