"""Simple in-memory sliding-window rate limiter (per client IP)."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from app.config import settings


class RateLimiter:
    def __init__(self, per_minute: int | None = None):
        self.per_minute = per_minute or settings.RATE_LIMIT_PER_MINUTE
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = now or time.time()
        window = self._hits[key]
        cutoff = now - 60.0
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


rate_limiter = RateLimiter()

# Paths that never count against the limit
EXEMPT_PREFIXES = ("/api/v1/health", "/plots", "/api/docs", "/openapi.json", "/")
LIMITED_PREFIXES = ("/api/v1/solve", "/api/v1/tutor", "/api/v1/practice")


def is_limited(path: str) -> bool:
    return any(path.startswith(p) for p in LIMITED_PREFIXES)
