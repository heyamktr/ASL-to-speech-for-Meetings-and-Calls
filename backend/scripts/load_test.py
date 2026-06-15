"""Concurrent WebSocket latency benchmark + capacity sweep.

Usage:
    python scripts/load_test.py
    python scripts/load_test.py --url ws://your-server/ws --sessions 20 --frames 300

    # Capacity sweep: ramp concurrency until p95 latency degrades past a threshold
    python scripts/load_test.py --sweep --steps 1,5,10,25,50,100 --p95-budget 200
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


async def run_level(
    url: str, num_sessions: int, num_frames: int
) -> tuple[list[SessionResult], float]:
    start = time.perf_counter()
    results = await asyncio.gather(
        *[run_session(url, num_frames) for _ in range(num_sessions)]
    )
    elapsed = time.perf_counter() - start
    return list(results), elapsed


async def main(url: str, num_sessions: int, num_frames: int) -> None:
    print(f"Load test: {num_sessions} concurrent sessions × {num_frames} frames")
    print(f"Target:    {url}\n")
    results, elapsed = await run_level(url, num_sessions, num_frames)
    report(results, elapsed)


async def sweep(url: str, steps: list[int], num_frames: int, p95_budget: float) -> None:
    """Ramp concurrency and find the max level whose p95 stays within budget."""
    print(f"Capacity sweep on {url}")
    print(f"Frames/session: {num_frames}   p95 budget: {p95_budget:.0f} ms\n")
    header = f"{'Sessions':>9}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}"
    print(header + f"{'errors':>8}{'within budget':>15}")
    print("-" * 62)

    max_ok = 0
    for n in steps:
        results, _ = await run_level(url, n, num_frames)
        latencies = sorted(ms for r in results for ms in r.latencies_ms)
        errors = sum(r.errors for r in results)
        if not latencies:
            print(f"{n:>9}{'-':>10}{'-':>10}{'-':>10}{errors:>8}{'NO DATA':>15}")
            break
        p50 = _percentile(latencies, 0.50)
        p95 = _percentile(latencies, 0.95)
        p99 = _percentile(latencies, 0.99)
        ok = p95 <= p95_budget and errors == 0
        if ok:
            max_ok = n
        print(f"{n:>9}{p50:>10.1f}{p95:>10.1f}{p99:>10.1f}{errors:>8}{('yes' if ok else 'NO'):>15}")

    print("-" * 62)
    print(f"\nMax concurrent sessions within {p95_budget:.0f} ms p95 budget: {max_ok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--sessions", type=int, default=10, help="concurrent sessions")
    parser.add_argument("--frames", type=int, default=200, help="frames per session")
    parser.add_argument("--sweep", action="store_true", help="run a capacity sweep")
    parser.add_argument("--steps", default="1,5,10,25,50,100",
                        help="comma-separated concurrency levels for --sweep")
    parser.add_argument("--p95-budget", type=float, default=200.0,
                        help="p95 latency budget in ms for --sweep")
    args = parser.parse_args()

    if args.sweep:
        levels = [int(s) for s in args.steps.split(",") if s.strip()]
        asyncio.run(sweep(args.url, levels, args.frames, args.p95_budget))
    else:
        asyncio.run(main(args.url, args.sessions, args.frames))
