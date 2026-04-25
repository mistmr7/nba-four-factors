"""Unit tests for nba_four_factors.storage.raw."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nba_four_factors.config import Endpoint, SeasonType
from nba_four_factors.storage import raw as raw_storage
from nba_four_factors.storage.raw import (
    _season_type_slug,
    exists_raw,
    load_raw,
    raw_game_path,
    raw_season_path,
    save_raw,
)


class TestSeasonTypeSlug:
    def test_regular(self):
        assert _season_type_slug(SeasonType.REGULAR) == "regular_season"

    def test_playoffs(self):
        assert _season_type_slug(SeasonType.PLAYOFFS) == "playoffs"

    def test_play_in_camel_case(self):
        assert _season_type_slug(SeasonType.PLAY_IN) == "play_in"


class TestRawSeasonPath:
    def test_schedule_regular(self, monkeypatch, tmp_path):
        monkeypatch.setattr(raw_storage, "RAW_DIR", tmp_path)
        path = raw_season_path(Endpoint.SCHEDULE, "2023_24", SeasonType.REGULAR)
        assert path == tmp_path / "leaguegamelog" / "2023_24" / "regular_season.json"

    def test_schedule_playoffs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(raw_storage, "RAW_DIR", tmp_path)
        path = raw_season_path(Endpoint.SCHEDULE, "2019_20", SeasonType.PLAYOFFS)
        assert path == tmp_path / "leaguegamelog" / "2019_20" / "playoffs.json"

    def test_schedule_play_in(self, monkeypatch, tmp_path):
        monkeypatch.setattr(raw_storage, "RAW_DIR", tmp_path)
        path = raw_season_path(Endpoint.SCHEDULE, "2024_25", SeasonType.PLAY_IN)
        assert path.name == "play_in.json"


class TestRawGamePath:
    def test_boxscore_traditional(self, monkeypatch, tmp_path):
        monkeypatch.setattr(raw_storage, "RAW_DIR", tmp_path)
        path = raw_game_path(Endpoint.BOXSCORE_TRADITIONAL, "0022300001")
        assert path == tmp_path / "boxscoretraditionalv2" / "0022300001.json"

    def test_boxscore_advanced(self, monkeypatch, tmp_path):
        monkeypatch.setattr(raw_storage, "RAW_DIR", tmp_path)
        path = raw_game_path(Endpoint.BOXSCORE_ADVANCED, "0022300001")
        assert path == tmp_path / "boxscoreadvancedv2" / "0022300001.json"

    def test_boxscore_summary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(raw_storage, "RAW_DIR", tmp_path)
        path = raw_game_path(Endpoint.BOXSCORE_SUMMARY, "0042300401")
        assert path == tmp_path / "boxscoresummaryv2" / "0042300401.json"

    def test_rejects_empty_game_id(self):
        with pytest.raises(ValueError, match="game_id"):
            raw_game_path(Endpoint.BOXSCORE_TRADITIONAL, "")

    def test_rejects_non_digit_game_id(self):
        with pytest.raises(ValueError, match="game_id"):
            raw_game_path(Endpoint.BOXSCORE_TRADITIONAL, "ABC123")

    def test_rejects_game_id_with_spaces(self):
        with pytest.raises(ValueError, match="game_id"):
            raw_game_path(Endpoint.BOXSCORE_TRADITIONAL, "002230 001")


class TestSaveLoadRoundtrip:
    def test_simple_dict_roundtrip(self, tmp_path):
        path = tmp_path / "x.json"
        payload = {"resultSets": [{"name": "GameLog", "rowSet": [[1, 2, 3]]}]}
        save_raw(path, payload)
        assert load_raw(path) == payload

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "x.json"
        save_raw(path, {"k": "v"})
        assert path.is_file()

    def test_save_overwrites(self, tmp_path):
        path = tmp_path / "x.json"
        save_raw(path, {"v": 1})
        save_raw(path, {"v": 2})
        assert load_raw(path) == {"v": 2}

    def test_save_is_atomic_no_tmp_leftover(self, tmp_path):
        path = tmp_path / "x.json"
        save_raw(path, {"k": "v"})
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_save_uses_compact_json(self, tmp_path):
        path = tmp_path / "x.json"
        save_raw(path, {"a": 1, "b": 2})
        text = path.read_text(encoding="utf-8")
        assert " " not in text

    def test_save_preserves_unicode(self, tmp_path):
        path = tmp_path / "x.json"
        payload = {"team": "Raptors", "city": "Montréal", "note": "é"}
        save_raw(path, payload)
        assert load_raw(path) == payload

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw(tmp_path / "nope.json")

    def test_load_malformed_raises(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text("not json{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_raw(path)


class TestExistsRaw:
    def test_existing(self, tmp_path):
        path = tmp_path / "x.json"
        save_raw(path, {})
        assert exists_raw(path) is True

    def test_missing(self, tmp_path):
        assert exists_raw(tmp_path / "nope.json") is False

    def test_directory_is_not_file(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        assert exists_raw(d) is False
