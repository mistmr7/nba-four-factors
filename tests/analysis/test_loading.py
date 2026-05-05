"""Tests for nba_four_factors.analysis._loading.

Coverage:

* ``_expand_season_range`` validates inputs and produces correct
  inclusive ranges
* ``read_processed`` concatenates per-season files, filters by
  ``season_type``, preserves the 39-column schema
* Missing seasons are silently skipped with an INFO log
* When no files match the request, an empty DataFrame with the canonical
  schema is returned (not None)
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from nba_four_factors.analysis import _loading
from nba_four_factors.analysis._loading import (
    _PROCESSED_SCHEMA,
    _expand_season_range,
    read_processed,
)
from nba_four_factors.config import SeasonType


def test_expand_inclusive_endpoints():
    seasons = _expand_season_range(("2020_21", "2022_23"))
    assert seasons == ["2020_21", "2021_22", "2022_23"]


def test_expand_single_season():
    assert _expand_season_range(("2024_25", "2024_25")) == ["2024_25"]


def test_expand_unknown_start_raises():
    with pytest.raises(ValueError, match="Unknown start season"):
        _expand_season_range(("1899_00", "2020_21"))


def test_expand_unknown_end_raises():
    with pytest.raises(ValueError, match="Unknown end season"):
        _expand_season_range(("2020_21", "9999_00"))


def test_expand_inverted_range_raises():
    with pytest.raises(ValueError, match="Inverted season range"):
        _expand_season_range(("2024_25", "2020_21"))


def test_read_processed_concatenates_seasons(monkeypatch, make_processed_tree):
    root = make_processed_tree(
        [
            ("2020_21", SeasonType.REGULAR),
            ("2021_22", SeasonType.REGULAR),
            ("2022_23", SeasonType.REGULAR),
        ],
        n_games=2,
    )
    monkeypatch.setattr(_loading, "PROCESSED_DIR", root)

    df = read_processed(("2020_21", "2022_23"), SeasonType.REGULAR)

    assert len(df) == 12
    assert set(df["season"].unique()) == {"2020_21", "2021_22", "2022_23"}


def test_read_processed_filters_by_season_type(monkeypatch, make_processed_tree):
    root = make_processed_tree(
        [
            ("2024_25", SeasonType.REGULAR),
            ("2024_25", SeasonType.PLAYOFFS),
        ],
        n_games=3,
    )
    monkeypatch.setattr(_loading, "PROCESSED_DIR", root)

    rs = read_processed(("2024_25", "2024_25"), SeasonType.REGULAR)
    po = read_processed(("2024_25", "2024_25"), SeasonType.PLAYOFFS)

    assert len(rs) == 6
    assert len(po) == 6
    assert (rs["season_type"] == "regular_season").all()
    assert (po["season_type"] == "playoffs").all()


def test_read_processed_schema_matches(monkeypatch, make_processed_tree):
    root = make_processed_tree([("2024_25", SeasonType.REGULAR)])
    monkeypatch.setattr(_loading, "PROCESSED_DIR", root)

    df = read_processed(("2024_25", "2024_25"), SeasonType.REGULAR)

    assert tuple(df.columns) == _PROCESSED_SCHEMA


def test_read_processed_uses_ignore_index(monkeypatch, make_processed_tree):
    """Concatenated frame must have a fresh RangeIndex, not stacked indices."""
    root = make_processed_tree(
        [
            ("2020_21", SeasonType.REGULAR),
            ("2021_22", SeasonType.REGULAR),
        ]
    )
    monkeypatch.setattr(_loading, "PROCESSED_DIR", root)

    df = read_processed(("2020_21", "2021_22"), SeasonType.REGULAR)

    assert isinstance(df.index, pd.RangeIndex)
    assert df.index.tolist() == list(range(len(df)))


def test_read_processed_skips_missing_seasons(
    monkeypatch,
    make_processed_tree,
    caplog,
):
    """Seasons in the requested range without files are skipped with INFO."""
    root = make_processed_tree(
        [
            ("2022_23", SeasonType.REGULAR),
            ("2024_25", SeasonType.REGULAR),
        ]
    )
    monkeypatch.setattr(_loading, "PROCESSED_DIR", root)

    with caplog.at_level(logging.INFO, logger=_loading.__name__):
        df = read_processed(("2020_21", "2024_25"), SeasonType.REGULAR)

    assert set(df["season"].unique()) == {"2022_23", "2024_25"}
    info_text = " ".join(rec.message for rec in caplog.records)
    assert "2020_21" in info_text
    assert "2021_22" in info_text
    assert "2023_24" in info_text


def test_read_processed_no_files_returns_empty_with_schema(
    monkeypatch,
    tmp_path,
    caplog,
):
    """An empty result is a properly-shaped empty DataFrame, not None."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.setattr(_loading, "PROCESSED_DIR", empty_root)

    with caplog.at_level(logging.INFO, logger=_loading.__name__):
        df = read_processed(("2020_21", "2022_23"), SeasonType.REGULAR)

    assert df.empty
    assert tuple(df.columns) == _PROCESSED_SCHEMA
    assert any("no files matched" in rec.message for rec in caplog.records)


def test_read_processed_returns_dataframe_not_none(monkeypatch, tmp_path):
    monkeypatch.setattr(_loading, "PROCESSED_DIR", tmp_path / "missing")

    df = read_processed(("2020_21", "2020_21"), SeasonType.REGULAR)

    assert isinstance(df, pd.DataFrame)


def test_read_processed_drops_canceled_games_by_default(
    monkeypatch,
    make_processed_tree,
    caplog,
):
    """Rows with pts == 0 AND opp_pts == 0 are excluded by default."""
    root = make_processed_tree([("2024_25", SeasonType.REGULAR)], n_games=2)
    monkeypatch.setattr(_loading, "PROCESSED_DIR", root)

    target = root / "2024_25" / "regular_season.parquet"
    df = pd.read_parquet(target)
    df.loc[0:1, "pts"] = 0
    df.loc[0:1, "opp_pts"] = 0
    df.to_parquet(target)

    with caplog.at_level(logging.INFO, logger=_loading.__name__):
        result = read_processed(("2024_25", "2024_25"), SeasonType.REGULAR)

    assert len(result) == 2
    assert ((result["pts"] == 0) & (result["opp_pts"] == 0)).sum() == 0
    assert any("canceled-game" in rec.message for rec in caplog.records)


def test_read_processed_keeps_canceled_when_requested(
    monkeypatch,
    make_processed_tree,
):
    """include_canceled=True passes 0-0 rows through unchanged."""
    root = make_processed_tree([("2024_25", SeasonType.REGULAR)], n_games=2)
    monkeypatch.setattr(_loading, "PROCESSED_DIR", root)

    target = root / "2024_25" / "regular_season.parquet"
    df = pd.read_parquet(target)
    df.loc[0:1, "pts"] = 0
    df.loc[0:1, "opp_pts"] = 0
    df.to_parquet(target)

    result = read_processed(
        ("2024_25", "2024_25"),
        SeasonType.REGULAR,
        include_canceled=True,
    )

    assert len(result) == 4
    assert ((result["pts"] == 0) & (result["opp_pts"] == 0)).sum() == 2
