"""In-process rate limiting — no Redis, no paid service.

A sliding-window counter keyed by (bucket, identity). Memory is bounded by
periodic eviction of idle keys. This is per-process, which is exactly right for
the single-container free-tier deployment InsightOS targets; if you later scale
horizontally, swap the `_Store` for a shared backend without touching callers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from app.config import settings
from app.errors import RateLimitedError

_EVICT_EVERY_SECONDS = 300
_MAX_KEYS = 20_000


@dataclass(frozen=True, slots=True)
class LimitPolicy:
    name: str
    requests: int
    window_seconds: int


def default_policies() -> dict[str, LimitPolicy]:
    return {
        "global": LimitPolicy(
            "global", settings.rate_limit_requests, settings.rate_limit_window_seconds
        ),
        "upload": LimitPolicy(
            "upload",
            settings.rate_limit_upload_requests,
            settings.rate_limit_window_seconds,
        ),
        "ai": LimitPolicy(
            "ai", settings.rate_limit_ai_requests, settings.rate_limit_window_seconds
        ),
        "auth": LimitPolicy("auth", 10, 300),
        "report": LimitPolicy("report", 20, 600),
    }


class RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._last_evict = time.monotonic()
        self._policies = default_policies()

    def reload_policies(self) -> None:
        with self._lock:
            self._policies = default_policies()

    def check(self, bucket: str, identity: str) -> None:
        """Raise :class:`RateLimitedError` if `identity` exceeded `bucket`."""
        if not settings.rate_limit_enabled:
            return
        policy = self._policies.get(bucket) or self._policies["global"]
        if policy.requests <= 0:
            return

        now = time.monotonic()
        key = (bucket, identity)
        cutoff = now - policy.window_seconds

        with self._lock:
            self._maybe_evict(now)
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= policy.requests:
                retry_after = max(1, int(hits[0] + policy.window_seconds - now) + 1)
                raise RateLimitedError(
                    f"Rate limit reached for '{bucket}': "
                    f"{policy.requests} requests per {policy.window_seconds}s. "
                    f"Retry in {retry_after}s.",
                    retry_after=retry_after,
                )
            hits.append(now)

    def remaining(self, bucket: str, identity: str) -> int:
        policy = self._policies.get(bucket) or self._policies["global"]
        with self._lock:
            hits = self._hits.get((bucket, identity))
            used = len(hits) if hits else 0
        return max(0, policy.requests - used)

    def _maybe_evict(self, now: float) -> None:
        if now - self._last_evict < _EVICT_EVERY_SECONDS and len(self._hits) < _MAX_KEYS:
            return
        self._last_evict = now
        longest_window = max((p.window_seconds for p in self._policies.values()), default=60)
        cutoff = now - longest_window
        stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
        for k in stale:
            self._hits.pop(k, None)
        if len(self._hits) > _MAX_KEYS:  # pragma: no cover - pathological load
            self._hits.clear()

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()
