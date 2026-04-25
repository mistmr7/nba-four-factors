"""Unit tests for nba_four_factors.storage.manifest."""

from __future__ import annotations

import json

import pytest

from nba_four_factors.config import SeasonType
from nba_four_factors.storage import manifest as manifest_storage
from nba_four_factors.storage.manifest import (
    exists_manifest,
    load_manifest,
    manifest_path,
    save_manifest,
)


class TestManifestPath:
    def test_regular(self, monkeypatch, tmp_path):
        monkeypatch.setattr(manifest_storage, "MANIFESTS_DIR", tmp_path)
        assert manifest_path("2024_25", SeasonType.REGULAR) == (
            tmp_path / "2024_25_regular_season.json"
        )

    def test_playoffs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(manifest_storage, "MANIFESTS_DIR", tmp_path)
        assert manifest_path("2019_20", SeasonType.PLAYOFFS) == tmp_path / "2019_20_playoffs.json"

    def test_play_in_camel_case(self, monkeypatch, tmp_path):
        monkeypatch.setattr(manifest_storage, "MANIFESTS_DIR", tmp_path)
        assert manifest_path("2024_25", SeasonType.PLAY_IN) == tmp_path / "2024_25_play_in.json"


class TestSaveLoadRoundtrip:
    def test_basic_roundtrip(self, tmp_path):
        path = tmp_path / "m.json"
        payload = {"season": "2024-25", "season_type": "Regular Season", "game_ids": ["0022400001"]}
        save_manifest(path, payload)
        assert load_manifest(path) == payload

    def test_save_creates_parents(self, tmp_path):
        path = tmp_path / "a" / "b" / "m.json"
        save_manifest(path, {"k": "v"})
        assert path.is_file()

    def test_save_is_pretty(self, tmp_path):
        path = tmp_path / "m.json"
        save_manifest(path, {"b": 2, "a": 1})
        text = path.read_text(encoding="utf-8")
        assert "\n" in text

    def test_save_sorts_keys(self, tmp_path):
        path = tmp_path / "m.json"
        save_manifest(path, {"z": 1, "a": 2, "m": 3})
        text = path.read_text(encoding="utf-8")
        a_pos = text.index('"a"')
        m_pos = text.index('"m"')
        z_pos = text.index('"z"')
        assert a_pos < m_pos < z_pos

    def test_save_ends_with_newline(self, tmp_path):
        path = tmp_path / "m.json"
        save_manifest(path, {"k": "v"})
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_save_is_atomic_no_tmp_leftover(self, tmp_path):
        path = tmp_path / "m.json"
        save_manifest(path, {"k": "v"})
        assert list(tmp_path.glob("*.tmp")) == []

    def test_save_overwrites(self, tmp_path):
        path = tmp_path / "m.json"
        save_manifest(path, {"v": 1})
        save_manifest(path, {"v": 2})
        assert load_manifest(path) == {"v": 2}

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nope.json")

    def test_load_malformed_raises(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("not json{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_manifest(path)

    def test_load_rejects_non_object_root(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_manifest(path)


class TestExistsManifest:
    def test_existing(self, tmp_path):
        path = tmp_path / "m.json"
        save_manifest(path, {})
        assert exists_manifest(path) is True

    def test_missing(self, tmp_path):
        assert exists_manifest(tmp_path / "nope.json") is False
