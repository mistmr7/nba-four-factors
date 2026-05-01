"""Shared fixtures for analysis-layer tests.

Builds a synthetic processed Parquet tree under tmp_path so tests don't
touch real data and don't pollute the working tree. The synthetic data
matches the 39-column processed schema with deterministic placeholder
values; tests assert structure and routing, not factor numerics
(those are covered by tests/processed/test_factors.py and test_pipeline.py).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from nba_four_factors.config import SeasonType
from nba_four_factors.storage.raw import season_type_slug


def _synthetic_team_game(
    season: str,
    season_type: SeasonType,
    n_games: int = 2,
) -> pd.DataFrame:
    """Build a tiny DataFrame matching the 39-column processed schema.

    Two rows per game (one per team). Numbers are placeholders, not
    physically meaningful; schema fidelity is what matters here.
    """
    rows = []
    for g in range(n_games):
        game_id = f"00{season[:4]}{g:05d}"
        for is_home, team_abbr, opp_abbr in [(True, "AAA", "BBB"), (False, "BBB", "AAA")]:
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": pd.Timestamp(f"{season[:4]}-11-{g + 1:02d}"),
                    "season": season,
                    "season_type": season_type_slug(season_type),
                    "team_id": 1 if is_home else 2,
                    "team_abbr": team_abbr,
                    "opp_team_id": 2 if is_home else 1,
                    "opp_abbr": opp_abbr,
                    "is_home": is_home,
                    "is_neutral": False,
                    "fgm": 40,
                    "fga": 85,
                    "fg3m": 10,
                    "fg3a": 30,
                    "ftm": 15,
                    "fta": 20,
                    "oreb": 10,
                    "dreb": 33,
                    "tov": 14,
                    "pts": 110 if is_home else 100,
                    "opp_fgm": 38,
                    "opp_fga": 88,
                    "opp_fg3m": 8,
                    "opp_fg3a": 28,
                    "opp_ftm": 14,
                    "opp_fta": 18,
                    "opp_oreb": 9,
                    "opp_dreb": 32,
                    "opp_tov": 16,
                    "opp_pts": 100 if is_home else 110,
                    "off_efg_pct": 0.50,
                    "off_tov_pct": 0.13,
                    "off_orb_pct": 0.25,
                    "off_ft_rate": 0.23,
                    "def_efg_pct": 0.47,
                    "def_tov_pct": 0.14,
                    "def_orb_pct": 0.22,
                    "def_ft_rate": 0.20,
                    "margin": 10 if is_home else -10,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def make_processed_tree(tmp_path) -> Callable[..., Path]:
    """Factory fixture: write synthetic Parquets at requested (season, type) pairs.

    Returns the root path of the synthetic tree. Tests use this with
    monkeypatch to redirect ``PROCESSED_DIR`` inside ``_loading``.

    Example
    -------
        root = make_processed_tree([
            ("2020_21", SeasonType.REGULAR),
            ("2021_22", SeasonType.PLAYOFFS),
        ])
        monkeypatch.setattr(_loading, "PROCESSED_DIR", root)
    """

    def _make(
        pairs: list[tuple[str, SeasonType]],
        n_games: int = 2,
    ) -> Path:
        root = tmp_path / "processed"
        for season, st in pairs:
            df = _synthetic_team_game(season, st, n_games=n_games)
            target = root / season / f"{season_type_slug(st)}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(target)
        return root

    return _make
