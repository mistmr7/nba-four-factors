"""HTTP client for stats.nba.com.

Wraps a `requests.Session` with:
    - rate limiting (RateLimiter from api.rate_limiter)
    - exponential backoff retry on 429, 5xx, and timeouts (locked policy)
    - nba_api-compatible headers (reused from nba_api.stats.library.http)

Non-retryable 4xx responses fail fast to surface real bugs rather than
burning attempts against a request that will never succeed (locked, 2.4).

Usage:
    client = Client()
    payload = client.fetch(
        Endpoint.SCHEDULE,
        {"Season": "2023-24", "SeasonType": "Regular Season", ...},
    )

`payload` is the parsed JSON response body.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import requests
from nba_api.stats.library.http import NBAStatsHTTP

from nba_four_factors.api.rate_limiter import RateLimiter
from nba_four_factors.config import (
    RETRY_BACKOFF_BASE_SECONDS,
    RETRY_BACKOFF_FACTOR,
    RETRY_BACKOFF_MAX_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_STATUS_CODES,
    Endpoint,
)

BASE_URL = "https://stats.nba.com/stats/{endpoint}"
DEFAULT_TIMEOUT_SECONDS: float = 30.0


class FetchError(Exception):
    """Base class for client-layer failures."""


class NonRetryableStatusError(FetchError):
    """A non-retryable HTTP status was returned (4xx other than 429)."""

    def __init__(self, status_code: int, url: str, body: str) -> None:
        super().__init__(f"{status_code} from {url}: {body[:200]}")
        self.status_code = status_code
        self.url = url
        self.body = body


class RetriesExhaustedError(FetchError):
    """All retry attempts failed for a retryable condition."""

    def __init__(self, attempts: int, last_reason: str) -> None:
        super().__init__(f"exhausted {attempts} attempts; last: {last_reason}")
        self.attempts = attempts
        self.last_reason = last_reason


def _default_headers() -> dict[str, str]:
    """Copy of nba_api's request headers. Copying defangs shared mutable state."""
    return dict(NBAStatsHTTP().headers or {})


class Client:
    """Thin HTTP client with rate limiting and retry for stats.nba.com."""

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = RETRY_MAX_ATTEMPTS,
        backoff_base_seconds: float = RETRY_BACKOFF_BASE_SECONDS,
        backoff_factor: float = RETRY_BACKOFF_FACTOR,
        backoff_max_seconds: float = RETRY_BACKOFF_MAX_SECONDS,
        retry_status_codes: frozenset[int] = RETRY_STATUS_CODES,
        sleep_func: Callable[[float], None] = time.sleep,
        random_func: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        self._rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self._session = session if session is not None else requests.Session()
        self._session.headers.update(_default_headers())
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base_seconds
        self._backoff_factor = backoff_factor
        self._backoff_max = backoff_max_seconds
        self._retry_status = retry_status_codes
        self._sleep = sleep_func
        self._random = random_func

    def fetch(self, endpoint: Endpoint, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch `endpoint` with `params` and return parsed JSON body."""
        url = BASE_URL.format(endpoint=endpoint.value)
        sorted_params = dict(sorted(params.items()))
        last_reason = "no attempts made"

        for attempt in range(1, self._max_attempts + 1):
            self._rate_limiter.acquire()
            try:
                response = self._session.get(url, params=sorted_params, timeout=self._timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_reason = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_attempts:
                    self._sleep(self._compute_backoff(attempt))
                    continue
                raise RetriesExhaustedError(attempt, last_reason) from exc

            if response.status_code == 200:
                return response.json()

            if response.status_code in self._retry_status:
                last_reason = f"status {response.status_code}"
                if attempt < self._max_attempts:
                    self._sleep(self._compute_backoff(attempt))
                    continue
                raise RetriesExhaustedError(attempt, last_reason)

            raise NonRetryableStatusError(response.status_code, response.url, response.text)

        raise RetriesExhaustedError(self._max_attempts, last_reason)

    def _compute_backoff(self, attempt: int) -> float:
        """Exponential backoff with decorrelated jitter, capped at backoff_max."""
        base = self._backoff_base * (self._backoff_factor ** (attempt - 1))
        capped = min(base, self._backoff_max)
        return capped * (0.5 + 0.5 * self._random())
