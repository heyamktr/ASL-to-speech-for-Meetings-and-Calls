"""Concurrent WebSocket latency benchmark.

Usage:
    python scripts/load_test.py
    python scripts/load_test.py --url ws://your-server/ws --sessions 20 --frames 300
"""

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from typing import NamedTuple

import websockets


class SessionResult(NamedTuple):
    session_id: str
    latencies_ms: list[float]
    errors: int


def _dummy_landmarks() -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(144)]


async def run_session(url: str, num_frames: int) -> SessionResult:
    session_id = str(uuid.uuid4())
    latencies: list[float] = []
    errors = 0

    try:
        async with websockets.connect(url) as ws:
            for _ in range(num_frames):
                payload = {
                    "landmarks": _dummy_landmarks(),
                    "session_id": session_id,
                    "timestamp": time.time() * 1000,
                }
                t0 = time.perf_counter()
                await ws.send(json.dumps(payload))
                await ws.recv()
                latencies.append((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        errors += 1
        print(f"  session {session_id[:8]}... error: {exc}")

    return SessionResult(session_id, latencies, errors)


def _percentile(sorted_data: list[float], p: float) -> float:
    idx = max(0, int(p * len(sorted_data)) - 1)
    return sorted_data[idx]


def report(results: list[SessionResult], elapsed: float) -> None:
    all_latencies = sorted(l for r in results for l in r.latencies_ms)
    total_errors = sum(r.errors for r in results)
    successful = sum(1 for r in results if not r.errors)

    print(f"\n{'='*52}")
    print(f"  Sessions:              {len(results)} ({successful} succeeded)")
    print(f"  Total frames sent:     {len(all_latencies)}")
    print(f"  Errors:                {total_errors}")
    print(f"  Wall time:             {elapsed:.2f}s")

    if not all_latencies:
        print("  No latency data collected.")
        print(f"{'='*52}")
        return

    print(f"  Min latency:           {min(all_latencies):.1f} ms")
    print(f"  Mean latency:          {statistics.mean(all_latencies):.1f} ms")
    print(f"  Median (p50):          {_percentile(all_latencies, 0.50):.1f} ms")
    print(f"  p95 latency:           {_percentile(all_latencies, 0.95):.1f} ms")
    print(f"  p99 latency:           {_percentile(all_latencies, 0.99):.1f} ms")
    print(f"  Max latency:           {max(all_latencies):.1f} ms")
    print(f"  Throughput:            {len(all_latencies) / elapsed:.1f} frames/s")
    print(f"{'='*52}\n")


async def main(url: str, num_sessions: int, num_frames: int) -> None:
    print(f"Load test: {num_sessions} concurrent sessions × {num_frames} frames")
    print(f"Target:    {url}\n")

    start = time.perf_counter()
    results = await asyncio.gather(*[run_session(url, num_frames) for _ in range(num_sessions)])
    elapsed = time.perf_counter() - start

    report(list(results), elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--sessions", type=int, default=10, help="concurrent sessions")
    parser.add_argument("--frames", type=int, default=200, help="frames per session")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.sessions, args.frames))
