"""Unit tests for nba_four_factors.api.client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from nba_four_factors.api.client import (
    BASE_URL,
    Client,
    FetchError,
    NonRetryableStatusError,
    RetriesExhaustedError,
)
from nba_four_factors.api.rate_limiter import RateLimiter
from nba_four_factors.config import Endpoint


def _fake_rate_limiter():
    rl = MagicMock(spec=RateLimiter)
    rl.acquire = MagicMock()
    return rl


def _session() -> MagicMock:
    s = MagicMock(spec=requests.Session)
    s.headers = {}
    return s


def _response(status_code: int, body: dict | None = None, text: str = "") -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.json = MagicMock(return_value=body if body is not None else {})
    r.text = text
    r.url = "https://stats.nba.com/stats/test"
    return r


def _client(session, rate_limiter=None, max_attempts: int = 5) -> Client:
    return Client(
        rate_limiter=rate_limiter or _fake_rate_limiter(),
        session=session,
        max_attempts=max_attempts,
        sleep_func=lambda _s: None,
        random_func=lambda: 0.0,
    )


class TestConstruction:
    def test_rejects_zero_max_attempts(self):
        with pytest.raises(ValueError, match="max_attempts"):
            Client(max_attempts=0, session=_session())

    def test_applies_default_headers_to_session(self):
        session = requests.Session()
        Client(session=session, rate_limiter=_fake_rate_limiter())
        assert "Host" in session.headers
        assert session.headers["Host"] == "stats.nba.com"


class TestSuccessfulFetch:
    def test_200_returns_parsed_body(self):
        session = _session()
        session.get.return_value = _response(200, body={"resultSets": []})
        client = _client(session)
        result = client.fetch(Endpoint.SCHEDULE, {"Season": "2023-24"})
        assert result == {"resultSets": []}

    def test_uses_correct_url_for_endpoint(self):
        session = _session()
        session.get.return_value = _response(200)
        client = _client(session)
        client.fetch(Endpoint.BOXSCORE_TRADITIONAL, {"GameID": "0022300001"})
        url_arg = session.get.call_args.args[0]
        assert url_arg == BASE_URL.format(endpoint="boxscoretraditionalv3")

    def test_sorts_params_alphabetically(self):
        session = _session()
        session.get.return_value = _response(200)
        client = _client(session)
        client.fetch(Endpoint.SCHEDULE, {"Z": "last", "A": "first", "M": "middle"})
        params = session.get.call_args.kwargs["params"]
        assert list(params.keys()) == ["A", "M", "Z"]

    def test_passes_timeout(self):
        session = _session()
        session.get.return_value = _response(200)
        client = Client(
            session=session,
            rate_limiter=_fake_rate_limiter(),
            timeout_seconds=7.5,
            sleep_func=lambda _s: None,
            random_func=lambda: 0.0,
        )
        client.fetch(Endpoint.SCHEDULE, {})
        assert session.get.call_args.kwargs["timeout"] == 7.5


class TestRateLimiterIntegration:
    def test_acquire_called_once_per_success(self):
        session = _session()
        session.get.return_value = _response(200)
        rl = _fake_rate_limiter()
        client = _client(session, rate_limiter=rl)
        client.fetch(Endpoint.SCHEDULE, {})
        assert rl.acquire.call_count == 1

    def test_acquire_called_once_per_attempt(self):
        session = _session()
        session.get.side_effect = [_response(500), _response(500), _response(200)]
        rl = _fake_rate_limiter()
        client = _client(session, rate_limiter=rl)
        client.fetch(Endpoint.SCHEDULE, {})
        assert rl.acquire.call_count == 3


class TestRetryableStatuses:
    def test_429_retries_then_succeeds(self):
        session = _session()
        session.get.side_effect = [_response(429), _response(200, body={"ok": 1})]
        client = _client(session)
        assert client.fetch(Endpoint.SCHEDULE, {}) == {"ok": 1}
        assert session.get.call_count == 2

    def test_500_retries_then_succeeds(self):
        session = _session()
        session.get.side_effect = [_response(500), _response(200, body={"ok": 1})]
        client = _client(session)
        assert client.fetch(Endpoint.SCHEDULE, {}) == {"ok": 1}

    def test_503_retries_then_succeeds(self):
        session = _session()
        session.get.side_effect = [_response(503), _response(200, body={"ok": 1})]
        client = _client(session)
        assert client.fetch(Endpoint.SCHEDULE, {}) == {"ok": 1}

    def test_all_attempts_retryable_raises_exhausted(self):
        session = _session()
        session.get.side_effect = [_response(503)] * 5
        client = _client(session, max_attempts=5)
        with pytest.raises(RetriesExhaustedError) as exc_info:
            client.fetch(Endpoint.SCHEDULE, {})
        assert exc_info.value.attempts == 5
        assert "503" in exc_info.value.last_reason


class TestNonRetryable:
    def test_404_raises_immediately_no_retry(self):
        session = _session()
        session.get.return_value = _response(404, text="Not Found")
        client = _client(session)
        with pytest.raises(NonRetryableStatusError) as exc_info:
            client.fetch(Endpoint.SCHEDULE, {})
        assert exc_info.value.status_code == 404
        assert session.get.call_count == 1

    def test_403_raises_immediately(self):
        session = _session()
        session.get.return_value = _response(403, text="Forbidden")
        client = _client(session)
        with pytest.raises(NonRetryableStatusError):
            client.fetch(Endpoint.SCHEDULE, {})
        assert session.get.call_count == 1

    def test_422_raises_immediately(self):
        session = _session()
        session.get.return_value = _response(422, text="Unprocessable")
        client = _client(session)
        with pytest.raises(NonRetryableStatusError):
            client.fetch(Endpoint.SCHEDULE, {})
        assert session.get.call_count == 1


class TestExceptionRetries:
    def test_timeout_retries_then_succeeds(self):
        session = _session()
        session.get.side_effect = [requests.Timeout("slow"), _response(200, body={"ok": 1})]
        client = _client(session)
        assert client.fetch(Endpoint.SCHEDULE, {}) == {"ok": 1}

    def test_connection_error_retries_then_succeeds(self):
        session = _session()
        session.get.side_effect = [
            requests.ConnectionError("refused"),
            _response(200, body={"ok": 1}),
        ]
        client = _client(session)
        assert client.fetch(Endpoint.SCHEDULE, {}) == {"ok": 1}

    def test_all_timeouts_raises_exhausted(self):
        session = _session()
        session.get.side_effect = [requests.Timeout("slow")] * 5
        client = _client(session, max_attempts=5)
        with pytest.raises(RetriesExhaustedError) as exc_info:
            client.fetch(Endpoint.SCHEDULE, {})
        assert exc_info.value.attempts == 5
        assert "Timeout" in exc_info.value.last_reason

    def test_non_requests_exception_propagates(self):
        session = _session()
        session.get.side_effect = ValueError("unexpected")
        client = _client(session)
        with pytest.raises(ValueError):
            client.fetch(Endpoint.SCHEDULE, {})


class TestBackoff:
    def test_backoff_increases_exponentially(self):
        sleeps: list[float] = []
        session = _session()
        session.get.side_effect = [_response(500)] * 4 + [_response(200, body={"ok": 1})]
        client = Client(
            rate_limiter=_fake_rate_limiter(),
            session=session,
            max_attempts=5,
            backoff_base_seconds=1.0,
            backoff_factor=2.0,
            backoff_max_seconds=60.0,
            sleep_func=sleeps.append,
            random_func=lambda: 1.0,
        )
        client.fetch(Endpoint.SCHEDULE, {})
        assert sleeps == pytest.approx([1.0, 2.0, 4.0, 8.0])

    def test_backoff_capped_at_max(self):
        sleeps: list[float] = []
        session = _session()
        session.get.side_effect = [_response(500)] * 4 + [_response(200, body={"ok": 1})]
        client = Client(
            rate_limiter=_fake_rate_limiter(),
            session=session,
            max_attempts=5,
            backoff_base_seconds=10.0,
            backoff_factor=10.0,
            backoff_max_seconds=30.0,
            sleep_func=sleeps.append,
            random_func=lambda: 1.0,
        )
        client.fetch(Endpoint.SCHEDULE, {})
        assert all(s <= 30.0 for s in sleeps)
        assert sleeps[-1] == pytest.approx(30.0)

    def test_backoff_jitter_halves_at_random_zero(self):
        sleeps: list[float] = []
        session = _session()
        session.get.side_effect = [_response(500), _response(200, body={"ok": 1})]
        client = Client(
            rate_limiter=_fake_rate_limiter(),
            session=session,
            max_attempts=5,
            backoff_base_seconds=2.0,
            backoff_factor=2.0,
            backoff_max_seconds=60.0,
            sleep_func=sleeps.append,
            random_func=lambda: 0.0,
        )
        client.fetch(Endpoint.SCHEDULE, {})
        assert sleeps == pytest.approx([1.0])


class TestExceptionHierarchy:
    def test_non_retryable_is_fetch_error(self):
        assert issubclass(NonRetryableStatusError, FetchError)

    def test_retries_exhausted_is_fetch_error(self):
        assert issubclass(RetriesExhaustedError, FetchError)
