"""Unit tests for nba_four_factors.api.rate_limiter.

All tests inject a fake clock + sleep so they run in microseconds with
deterministic timing. No real wall-clock sleeps.
"""

from __future__ import annotations

import pytest

from nba_four_factors.api.rate_limiter import RateLimiter


class FakeClock:
    """Monotonic fake. `sleep(s)` advances the clock by s."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _fixed_random(value: float):
    return lambda: value


class TestConstruction:
    def test_rejects_zero_max_requests(self):
        with pytest.raises(ValueError, match="max_requests"):
            RateLimiter(max_requests=0)

    def test_rejects_negative_window(self):
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimiter(window_seconds=-1)

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimiter(window_seconds=0)

    def test_rejects_negative_jitter(self):
        with pytest.raises(ValueError, match="jitter_seconds"):
            RateLimiter(jitter_seconds=-0.1)


class TestAcquireBelowCap:
    def test_first_acquire_no_sleep_with_zero_jitter(self):
        clock = FakeClock()
        rl = RateLimiter(
            max_requests=5,
            window_seconds=60,
            jitter_seconds=0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.0),
        )
        rl.acquire()
        assert clock.sleeps == []
        assert rl.recorded == 1

    def test_many_acquires_under_cap_no_blocking(self):
        clock = FakeClock()
        rl = RateLimiter(
            max_requests=30,
            window_seconds=60,
            jitter_seconds=0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.0),
        )
        for _ in range(10):
            rl.acquire()
        assert clock.sleeps == []
        assert rl.recorded == 10


class TestAcquireAtCap:
    def test_blocks_until_oldest_exits_window(self):
        clock = FakeClock(start=1000.0)
        rl = RateLimiter(
            max_requests=3,
            window_seconds=60,
            jitter_seconds=0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.0),
        )
        rl.acquire()
        rl.acquire()
        rl.acquire()
        assert clock.sleeps == []
        rl.acquire()
        assert len(clock.sleeps) == 1
        assert clock.sleeps[0] == pytest.approx(60.0)
        assert rl.recorded == 1

    def test_blocks_by_time_remaining_not_full_window(self):
        clock = FakeClock(start=1000.0)
        rl = RateLimiter(
            max_requests=2,
            window_seconds=60,
            jitter_seconds=0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.0),
        )
        rl.acquire()
        clock.now = 1010.0
        rl.acquire()
        clock.now = 1020.0
        rl.acquire()
        assert clock.sleeps[0] == pytest.approx(40.0)


class TestJitter:
    def test_applies_scaled_uniform_jitter(self):
        clock = FakeClock()
        rl = RateLimiter(
            max_requests=5,
            window_seconds=60,
            jitter_seconds=1.0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.5),
        )
        rl.acquire()
        assert clock.sleeps == [pytest.approx(0.5)]

    def test_zero_jitter_means_zero_sleep(self):
        clock = FakeClock()
        rl = RateLimiter(
            max_requests=5,
            window_seconds=60,
            jitter_seconds=0.0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.9),
        )
        rl.acquire()
        assert clock.sleeps == []

    def test_jitter_stacks_with_blocking_wait(self):
        clock = FakeClock(start=1000.0)
        rl = RateLimiter(
            max_requests=2,
            window_seconds=60,
            jitter_seconds=1.0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.25),
        )
        rl.acquire()
        rl.acquire()
        rl.acquire()
        assert clock.sleeps == [
            pytest.approx(0.25),
            pytest.approx(0.25),
            pytest.approx(59.75),
            pytest.approx(0.25),
        ]


class TestPurgeBehavior:
    def test_old_entries_leave_window(self):
        clock = FakeClock(start=1000.0)
        rl = RateLimiter(
            max_requests=3,
            window_seconds=60,
            jitter_seconds=0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.0),
        )
        rl.acquire()
        rl.acquire()
        rl.acquire()
        clock.now = 1100.0
        rl.acquire()
        assert clock.sleeps == []
        assert rl.recorded == 1

    def test_partial_purge(self):
        clock = FakeClock(start=1000.0)
        rl = RateLimiter(
            max_requests=3,
            window_seconds=60,
            jitter_seconds=0,
            time_func=clock.time,
            sleep_func=clock.sleep,
            random_func=_fixed_random(0.0),
        )
        rl.acquire()
        clock.now = 1030.0
        rl.acquire()
        rl.acquire()
        clock.now = 1070.0
        rl.acquire()
        assert clock.sleeps == []
        assert rl.recorded == 3


class TestDefaults:
    def test_uses_config_defaults(self):
        rl = RateLimiter(
            time_func=lambda: 0.0,
            sleep_func=lambda _s: None,
            random_func=_fixed_random(0.0),
        )
        assert rl._max_requests == 30
        assert rl._window == 60.0
