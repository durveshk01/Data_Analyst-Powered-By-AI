"""In-process metrics — zero-cost observability.

Keeps bounded, rolling counters and latency histograms in memory and exposes
them over `GET /api/v1/health/metrics`. This deliberately avoids a paid APM
vendor; for durable history, scrape the endpoint or ship the JSON logs.

Everything is bounded: counters are plain ints, timers keep only the last
`_WINDOW` samples per key, so memory use is constant.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Iterator

_WINDOW = 256


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timers: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))
        self._events: deque[dict[str, Any]] = deque(maxlen=_WINDOW)
        self._started_at = time.time()

    # ------------------------------------------------------------- counters --
    def incr(self, name: str, value: int = 1, **labels: Any) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] += value

    # --------------------------------------------------------------- timers --
    def observe(self, name: str, seconds: float, **labels: Any) -> None:
        key = _key(name, labels)
        with self._lock:
            self._timers[key].append(seconds)

    @contextmanager
    def timer(self, name: str, **labels: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start, **labels)

    # --------------------------------------------------------------- events --
    def record_event(self, kind: str, **fields: Any) -> None:
        """Append a bounded audit-style event (user actions, failures)."""
        with self._lock:
            self._events.append({"ts": time.time(), "kind": kind, **fields})

    # -------------------------------------------------------------- snapshot --
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            timers = {
                key: _summarise(list(samples))
                for key, samples in self._timers.items()
                if samples
            }
            events = list(self._events)[-40:]
        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "counters": counters,
            "latency_ms": timers,
            "recent_events": events,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timers.clear()
            self._events.clear()


def _key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    parts = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
    return f"{name}{{{parts}}}"


def _summarise(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0] * 1000
        idx = min(n - 1, max(0, int(round((p / 100) * (n - 1)))))
        return ordered[idx] * 1000

    return {
        "count": n,
        "avg": round(sum(ordered) / n * 1000, 2),
        "p50": round(pct(50), 2),
        "p95": round(pct(95), 2),
        "max": round(ordered[-1] * 1000, 2),
    }


metrics = MetricsRegistry()
