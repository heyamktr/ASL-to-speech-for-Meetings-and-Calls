"""Frequency-aware prediction cache (Week 5 latency lever #2).

The roadmap calls for "caching the most frequently predicted words so their
response bypasses full inference." We implement that as a small **LFU (least
frequently used) cache** keyed by a *quantized fingerprint* of the landmark
window: identical or near-identical windows (rounded to a few decimals) reuse the
previous result and skip both feature-building and the ONNX run.

When the cache is full, the entry with the lowest hit count is evicted, so the
hottest windows (the words a given user signs most often, and the all-zero
"no hand" window) stay resident — which is exactly the "20 most frequently
predicted" behaviour the roadmap asks for.

This cache is process-local and single-threaded-safe for the asyncio event loop
(no awaits inside the critical section).
"""

from __future__ import annotations

import hashlib

import numpy as np


class PredictionCache:
    def __init__(self, capacity: int = 20, decimals: int = 2) -> None:
        self.capacity = max(0, capacity)
        self.decimals = decimals
        # key -> (prediction, confidence, hits)
        self._store: dict[str, tuple[str, float, int]] = {}
        self.hits = 0
        self.misses = 0

    def fingerprint(self, window: np.ndarray) -> str:
        """Stable hash of the window rounded to ``decimals`` places."""
        rounded = np.round(np.asarray(window, dtype=np.float32), self.decimals)
        return hashlib.blake2b(rounded.tobytes(), digest_size=16).hexdigest()

    def get(self, key: str) -> tuple[str, float] | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        prediction, confidence, count = entry
        self._store[key] = (prediction, confidence, count + 1)
        self.hits += 1
        return prediction, confidence

    def put(self, key: str, prediction: str, confidence: float) -> None:
        if self.capacity == 0:
            return
        if key not in self._store and len(self._store) >= self.capacity:
            # Evict the least frequently used entry.
            victim = min(self._store, key=lambda k: self._store[k][2])
            del self._store[victim]
        # New entries start with hit count 1 so a single reuse is enough to
        # outrank a never-reused neighbour.
        self._store[key] = (prediction, confidence, 1)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
