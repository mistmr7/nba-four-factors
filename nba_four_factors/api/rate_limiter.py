"""Fixed-rate HTTP rate limiter with jitter.

Locked decision (Section 2.4): 30 requests/minute, fixed, with jitter.
`nba_api` and stats.nba.com have no published limit; 30/min is deliberately
conservative. Jitter avoids synchronized bursts when multiple retries all
come due at the same monotonic instant.

Implementation: sliding window. We keep the timestamps of the last N
requests; `acquire()` purges entries older than the window, and if the
window is full, sleeps until the oldest entry exits, plus jitter.

Time and randomness are injectable so unit tests can be deterministic
without real sleep.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable

from nba_four_factors.config import (
    RATE_LIMIT_JITTER_SECONDS,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
)


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Args:
        max_requests: Ceiling within the window. Defaults to the configured
            RATE_LIMIT_REQUESTS.
        window_seconds: Window size in seconds. Defaults to the configured
            RATE_LIMIT_WINDOW_SECONDS.
        jitter_seconds: Maximum uniform random delay added on every
            acquire. Applied whether or not we had to block. 0.0 disables.
        time_func: Monotonic clock source. Injectable for tests.
        sleep_func: Blocking sleep function. Injectable for tests.
        random_func: RNG returning [0.0, 1.0). Injectable for tests.
    """

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_PER_MINUTE,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        jitter_seconds: float = RATE_LIMIT_JITTER_SECONDS,
        time_func: Callable[[], float] = time.monotonic,
        sleep_func: Callable[[float], None] = time.sleep,
        random_func: Callable[[], float] = random.random,
    ) -> None:
        if max_requests < 1:
            raise ValueError(f"max_requests must be >= 1, got {max_requests}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if jitter_seconds < 0:
            raise ValueError(f"jitter_seconds must be >= 0, got {jitter_seconds}")

        self._max_requests = max_requests
        self._window = window_seconds
        self._jitter = jitter_seconds
        self._time = time_func
        self._sleep = sleep_func
        self._random = random_func
        self._timestamps: deque[float] = deque(maxlen=max_requests)
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available, then record the acquisition."""
        with self._lock:
            now = self._time()
            self._purge(now)

            if len(self._timestamps) >= self._max_requests:
                oldest = self._timestamps[0]
                wait = (oldest + self._window) - now
                if wait > 0:
                    self._sleep(wait)
                    now = self._time()
                    self._purge(now)

            if self._jitter > 0:
                self._sleep(self._random() * self._jitter)
                now = self._time()

            self._timestamps.append(now)

    def _purge(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    @property
    def recorded(self) -> int:
        """Number of acquisitions currently inside the window (after purge at last call)."""
        return len(self._timestamps)
