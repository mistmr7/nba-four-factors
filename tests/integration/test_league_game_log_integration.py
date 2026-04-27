"""End-to-end integration test for fetch_league_game_log against a recorded cassette.

This is the first test that exercises the full stack:
client (with rate limiter + retry) -> endpoint wrapper -> recorded HTTP response.

Cassette: tests/fixtures/cassettes/leaguegamelog_2023_24_regular.yaml
Recorded against: stats.nba.com/stats/leaguegamelog, 2023-24 regular season.
"""

from __future__ import annotations

from nba_four_factors.api.client import Client
from nba_four_factors.api.endpoints.league_game_log import fetch_league_game_log
from nba_four_factors.config import SeasonType

CASSETTE_NAME = "leaguegamelog_2023_24_regular"

EXPECTED_HEADERS_2023_24 = {
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
    "VIDEO_AVAILABLE",
}

EXPECTED_ROW_COUNT_2023_24 = 2460


class TestLeagueGameLogIntegration:
    def test_payload_top_level_shape(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_league_game_log(Client(), "2023_24", SeasonType.REGULAR)

        assert "resultSets" in payload
        assert isinstance(payload["resultSets"], list)
        assert len(payload["resultSets"]) >= 1

    def test_first_resultset_is_league_game_log(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_league_game_log(Client(), "2023_24", SeasonType.REGULAR)

        rs = payload["resultSets"][0]
        assert rs["name"] == "LeagueGameLog"

    def test_headers_match_expected_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_league_game_log(Client(), "2023_24", SeasonType.REGULAR)

        rs = payload["resultSets"][0]
        assert set(rs["headers"]) == EXPECTED_HEADERS_2023_24

    def test_row_count_matches_full_regular_season(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_league_game_log(Client(), "2023_24", SeasonType.REGULAR)

        rs = payload["resultSets"][0]
        assert len(rs["rowSet"]) == EXPECTED_ROW_COUNT_2023_24

    def test_row_width_matches_header_width(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_league_game_log(Client(), "2023_24", SeasonType.REGULAR)

        rs = payload["resultSets"][0]
        header_count = len(rs["headers"])
        assert all(len(row) == header_count for row in rs["rowSet"])
