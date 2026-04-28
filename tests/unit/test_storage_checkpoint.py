"""Unit tests for storage.checkpoint."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nba_four_factors.config import Endpoint, SeasonType
from nba_four_factors.storage import checkpoint as cp

# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------


class TestCheckpointPath:
    def test_regular_season(self) -> None:
        path = cp.checkpoint_path("2023_24", SeasonType.REGULAR)
        assert path == cp.CHECKPOINTS_DIR / "2023_24_regular_season.json"

    def test_playoffs(self) -> None:
        path = cp.checkpoint_path("2019_20", SeasonType.PLAYOFFS)
        assert path == cp.CHECKPOINTS_DIR / "2019_20_playoffs.json"

    def test_play_in(self) -> None:
        path = cp.checkpoint_path("2024_25", SeasonType.PLAY_IN)
        assert path == cp.CHECKPOINTS_DIR / "2024_25_play_in.json"


# ---------------------------------------------------------------------------
# I/O — atomic write, round-trip, missing-file load
# ---------------------------------------------------------------------------


class TestIO:
    def test_round_trip(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cp, "CHECKPOINTS_DIR", tmp_path)
        data = cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)
        cp.mark_game_complete(
            data,
            Endpoint.BOXSCORE_TRADITIONAL,
            "0022300001",
            now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )
        cp.save_checkpoint("2023_24", SeasonType.REGULAR, data)
        loaded = cp.load_checkpoint("2023_24", SeasonType.REGULAR)
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cp, "CHECKPOINTS_DIR", tmp_path)
        assert cp.load_checkpoint("2023_24", SeasonType.REGULAR) is None

    def test_save_no_tmp_leftover(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cp, "CHECKPOINTS_DIR", tmp_path)
        data = cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)
        cp.save_checkpoint("2023_24", SeasonType.REGULAR, data)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_save_creates_parent_dir(self, tmp_path, monkeypatch) -> None:
        nested = tmp_path / "data" / "checkpoints"
        monkeypatch.setattr(cp, "CHECKPOINTS_DIR", nested)
        data = cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)
        cp.save_checkpoint("2023_24", SeasonType.REGULAR, data)
        assert (nested / "2023_24_regular_season.json").exists()


# ---------------------------------------------------------------------------
# initialize_checkpoint shape
# ---------------------------------------------------------------------------


class TestInitializeCheckpoint:
    def test_top_level_fields(self) -> None:
        data = cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)
        assert data["season"] == "2023_24"
        assert data["season_type"] == "regular_season"

    def test_schedule_block(self) -> None:
        data = cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)
        assert data["schedule"]["endpoint"] == Endpoint.SCHEDULE.value
        assert data["schedule"]["status"] == "not_started"
        assert data["schedule"]["started_at"] is None
        assert data["schedule"]["completed_at"] is None

    def test_box_score_keys(self) -> None:
        data = cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)
        assert set(data["box_scores"]) == {
            Endpoint.BOXSCORE_TRADITIONAL.value,
            Endpoint.BOXSCORE_ADVANCED.value,
            Endpoint.BOXSCORE_SUMMARY.value,
        }

    def test_box_score_blocks_empty(self) -> None:
        data = cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)
        for ep_value, block in data["box_scores"].items():
            assert block["completed"] == [], ep_value
            assert block["failed"] == [], ep_value
            assert block["started_at"] is None, ep_value
            assert block["last_updated"] is None, ep_value


# ---------------------------------------------------------------------------
# Schedule transitions
# ---------------------------------------------------------------------------


class TestScheduleTransitions:
    @staticmethod
    def _fresh() -> dict:
        return cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)

    def test_not_started_to_in_progress(self) -> None:
        c = self._fresh()
        fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        cp.mark_schedule_started(c, now=lambda: fixed)
        assert c["schedule"]["status"] == "in_progress"
        assert c["schedule"]["started_at"] == "2026-01-01T12:00:00Z"
        assert c["schedule"]["completed_at"] is None

    def test_in_progress_to_complete(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        fixed = datetime(2026, 1, 1, 12, 5, 0, tzinfo=UTC)
        cp.mark_schedule_complete(c, now=lambda: fixed)
        assert c["schedule"]["status"] == "complete"
        assert c["schedule"]["completed_at"] == "2026-01-01T12:05:00Z"

    def test_in_progress_to_failed(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        cp.mark_schedule_failed(c)
        assert c["schedule"]["status"] == "failed"
        assert c["schedule"]["completed_at"] is not None

    def test_failed_to_in_progress_retry(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        cp.mark_schedule_failed(c)
        cp.mark_schedule_started(c)
        assert c["schedule"]["status"] == "in_progress"
        assert c["schedule"]["completed_at"] is None  # reset on retry

    def test_invalid_complete_to_in_progress(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        cp.mark_schedule_complete(c)
        with pytest.raises(ValueError, match="Invalid schedule transition"):
            cp.mark_schedule_started(c)

    def test_invalid_not_started_to_complete(self) -> None:
        c = self._fresh()
        with pytest.raises(ValueError, match="Invalid schedule transition"):
            cp.mark_schedule_complete(c)

    def test_invalid_not_started_to_failed(self) -> None:
        c = self._fresh()
        with pytest.raises(ValueError, match="Invalid schedule transition"):
            cp.mark_schedule_failed(c)

    def test_invalid_complete_to_failed(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        cp.mark_schedule_complete(c)
        with pytest.raises(ValueError, match="Invalid schedule transition"):
            cp.mark_schedule_failed(c)


# ---------------------------------------------------------------------------
# Game mutators
# ---------------------------------------------------------------------------


class TestMarkGameComplete:
    @staticmethod
    def _fresh() -> dict:
        return cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)

    def test_adds_to_completed(self) -> None:
        c = self._fresh()
        fixed = datetime(2026, 1, 1, tzinfo=UTC)
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001", now=lambda: fixed)
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["completed"] == ["0022300001"]
        assert block["last_updated"] == "2026-01-01T00:00:00Z"
        assert block["started_at"] == "2026-01-01T00:00:00Z"

    def test_idempotent_does_not_duplicate(self) -> None:
        c = self._fresh()
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["completed"] == ["0022300001"]

    def test_idempotent_does_not_update_last_updated(self) -> None:
        c = self._fresh()
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 1, 2, tzinfo=UTC)
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001", now=lambda: t1)
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001", now=lambda: t2)
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["last_updated"] == "2026-01-01T00:00:00Z"

    def test_removes_from_failed_on_retry_success(self) -> None:
        c = self._fresh()
        cp.mark_game_failed(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["completed"] == ["0022300001"]
        assert block["failed"] == []

    def test_started_at_only_set_once(self) -> None:
        c = self._fresh()
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 1, 2, tzinfo=UTC)
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001", now=lambda: t1)
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300002", now=lambda: t2)
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["started_at"] == "2026-01-01T00:00:00Z"
        assert block["last_updated"] == "2026-01-02T00:00:00Z"


class TestMarkGameFailed:
    @staticmethod
    def _fresh() -> dict:
        return cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)

    def test_adds_to_failed(self) -> None:
        c = self._fresh()
        cp.mark_game_failed(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["failed"] == ["0022300001"]

    def test_idempotent(self) -> None:
        c = self._fresh()
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 1, 2, tzinfo=UTC)
        cp.mark_game_failed(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001", now=lambda: t1)
        cp.mark_game_failed(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001", now=lambda: t2)
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["failed"] == ["0022300001"]
        assert block["last_updated"] == "2026-01-01T00:00:00Z"

    def test_does_not_downgrade_completed(self) -> None:
        c = self._fresh()
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        cp.mark_game_failed(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        block = c["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]
        assert block["completed"] == ["0022300001"]
        assert block["failed"] == []


# ---------------------------------------------------------------------------
# Endpoint validation — game mutators and queries reject schedule endpoint
# ---------------------------------------------------------------------------


class TestEndpointValidation:
    @staticmethod
    def _fresh() -> dict:
        return cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)

    def test_mark_complete_rejects_schedule(self) -> None:
        with pytest.raises(ValueError, match="not a box-score endpoint"):
            cp.mark_game_complete(self._fresh(), Endpoint.SCHEDULE, "0022300001")

    def test_mark_failed_rejects_schedule(self) -> None:
        with pytest.raises(ValueError, match="not a box-score endpoint"):
            cp.mark_game_failed(self._fresh(), Endpoint.SCHEDULE, "0022300001")

    def test_pending_games_rejects_schedule(self) -> None:
        with pytest.raises(ValueError, match="not a box-score endpoint"):
            cp.pending_games(self._fresh(), Endpoint.SCHEDULE, ["0022300001"])

    def test_is_endpoint_complete_rejects_schedule(self) -> None:
        with pytest.raises(ValueError, match="not a box-score endpoint"):
            cp.is_endpoint_complete(self._fresh(), Endpoint.SCHEDULE, 1)


# ---------------------------------------------------------------------------
# pending_games
# ---------------------------------------------------------------------------


class TestPendingGames:
    @staticmethod
    def _fresh() -> dict:
        return cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)

    def test_all_pending_when_none_done(self) -> None:
        c = self._fresh()
        all_ids = ["0022300001", "0022300002", "0022300003"]
        assert cp.pending_games(c, Endpoint.BOXSCORE_TRADITIONAL, all_ids) == all_ids

    def test_excludes_completed(self) -> None:
        c = self._fresh()
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        all_ids = ["0022300001", "0022300002"]
        assert cp.pending_games(c, Endpoint.BOXSCORE_TRADITIONAL, all_ids) == ["0022300002"]

    def test_includes_failed_for_retry(self) -> None:
        c = self._fresh()
        cp.mark_game_failed(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        all_ids = ["0022300001", "0022300002"]
        assert cp.pending_games(c, Endpoint.BOXSCORE_TRADITIONAL, all_ids) == [
            "0022300001",
            "0022300002",
        ]

    def test_preserves_input_order(self) -> None:
        c = self._fresh()
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300002")
        all_ids = ["0022300003", "0022300001", "0022300002", "0022300004"]
        assert cp.pending_games(c, Endpoint.BOXSCORE_TRADITIONAL, all_ids) == [
            "0022300003",
            "0022300001",
            "0022300004",
        ]

    def test_isolated_per_endpoint(self) -> None:
        c = self._fresh()
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        all_ids = ["0022300001", "0022300002"]
        # Other endpoints unaffected — both still pending.
        assert cp.pending_games(c, Endpoint.BOXSCORE_ADVANCED, all_ids) == all_ids


# ---------------------------------------------------------------------------
# is_endpoint_complete (lenient: completed + failed == expected_count)
# ---------------------------------------------------------------------------


class TestIsEndpointComplete:
    @staticmethod
    def _fresh() -> dict:
        return cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)

    def test_false_when_nothing_done(self) -> None:
        assert not cp.is_endpoint_complete(self._fresh(), Endpoint.BOXSCORE_TRADITIONAL, 5)

    def test_true_when_all_completed(self) -> None:
        c = self._fresh()
        for i in range(1, 4):
            cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, f"002230000{i}")
        assert cp.is_endpoint_complete(c, Endpoint.BOXSCORE_TRADITIONAL, 3)

    def test_true_when_completed_plus_failed_meets_count(self) -> None:
        c = self._fresh()
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300002")
        cp.mark_game_failed(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300003")
        assert cp.is_endpoint_complete(c, Endpoint.BOXSCORE_TRADITIONAL, 3)

    def test_false_when_under_count(self) -> None:
        c = self._fresh()
        cp.mark_game_complete(c, Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        assert not cp.is_endpoint_complete(c, Endpoint.BOXSCORE_TRADITIONAL, 3)


# ---------------------------------------------------------------------------
# is_schedule_complete
# ---------------------------------------------------------------------------


class TestIsScheduleComplete:
    @staticmethod
    def _fresh() -> dict:
        return cp.initialize_checkpoint("2023_24", SeasonType.REGULAR)

    def test_false_initially(self) -> None:
        assert not cp.is_schedule_complete(self._fresh())

    def test_false_in_progress(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        assert not cp.is_schedule_complete(c)

    def test_true_when_complete(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        cp.mark_schedule_complete(c)
        assert cp.is_schedule_complete(c)

    def test_false_when_failed(self) -> None:
        c = self._fresh()
        cp.mark_schedule_started(c)
        cp.mark_schedule_failed(c)
        assert not cp.is_schedule_complete(c)
