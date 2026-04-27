"""End-to-end integration test for fetch_box_score_traditional against a recorded cassette.

Exercises the full stack against a per-game endpoint:
client (with rate limiter + retry) -> endpoint wrapper -> recorded HTTP response.

Cassette: tests/fixtures/cassettes/boxscoretraditionalv2_0022300590.yaml
Recorded against: stats.nba.com/stats/boxscoretraditionalv2,
2023-24 regular season game 0022300590 (IND @ POR, 2024-01-19).

Schema notes:
- This endpoint returns three resultSets: PlayerStats, TeamStats,
  TeamStarterBenchStats. All three are asserted on schema even though
  only TeamStats is currently consumed by four-factor computation.
- V2 uses 'TO' for turnovers (not 'TOV', which is what LeagueGameLog
  returns). Schema normalization at the processed layer will need to
  reconcile this.
"""

from __future__ import annotations

from nba_four_factors.api.client import Client
from nba_four_factors.api.endpoints.box_score_traditional import fetch_box_score_traditional

CASSETTE_NAME = "boxscoretraditionalv2_0022300590"
GAME_ID = "0022300590"

EXPECTED_PLAYER_STATS_HEADERS = {
    "GAME_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_CITY",
    "PLAYER_ID",
    "PLAYER_NAME",
    "NICKNAME",
    "START_POSITION",
    "COMMENT",
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
    "TO",
    "PF",
    "PTS",
    "PLUS_MINUS",
}

EXPECTED_TEAM_STATS_HEADERS = {
    "GAME_ID",
    "TEAM_ID",
    "TEAM_NAME",
    "TEAM_ABBREVIATION",
    "TEAM_CITY",
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
    "TO",
    "PF",
    "PTS",
    "PLUS_MINUS",
}

EXPECTED_STARTER_BENCH_HEADERS = {
    "GAME_ID",
    "TEAM_ID",
    "TEAM_NAME",
    "TEAM_ABBREVIATION",
    "TEAM_CITY",
    "STARTERS_BENCH",
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
    "TO",
    "PF",
    "PTS",
}

EXPECTED_TEAM_STATS_ROWS = 2
EXPECTED_STARTER_BENCH_ROWS = 4
PLAYER_STATS_MIN_ROWS = 16
PLAYER_STATS_MAX_ROWS = 30


class TestBoxScoreTraditionalIntegration:
    def test_payload_top_level_shape(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        assert "resultSets" in payload
        assert isinstance(payload["resultSets"], list)
        assert len(payload["resultSets"]) == 3

    def test_resultset_names_and_order(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        names = [rs["name"] for rs in payload["resultSets"]]
        assert names == ["PlayerStats", "TeamStats", "TeamStarterBenchStats"]

    def test_player_stats_headers_match_expected_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        rs = payload["resultSets"][0]
        assert set(rs["headers"]) == EXPECTED_PLAYER_STATS_HEADERS

    def test_team_stats_headers_match_expected_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        rs = payload["resultSets"][1]
        assert set(rs["headers"]) == EXPECTED_TEAM_STATS_HEADERS

    def test_starter_bench_headers_match_expected_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        rs = payload["resultSets"][2]
        assert set(rs["headers"]) == EXPECTED_STARTER_BENCH_HEADERS

    def test_player_stats_row_count_is_plausible(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        rs = payload["resultSets"][0]
        assert PLAYER_STATS_MIN_ROWS <= len(rs["rowSet"]) <= PLAYER_STATS_MAX_ROWS

    def test_team_stats_has_two_rows(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        rs = payload["resultSets"][1]
        assert len(rs["rowSet"]) == EXPECTED_TEAM_STATS_ROWS

    def test_starter_bench_has_four_rows(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        rs = payload["resultSets"][2]
        assert len(rs["rowSet"]) == EXPECTED_STARTER_BENCH_ROWS

    def test_all_resultsets_have_consistent_row_widths(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        for rs in payload["resultSets"]:
            header_count = len(rs["headers"])
            assert all(len(row) == header_count for row in rs["rowSet"])

    def test_team_stats_rows_belong_to_game(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional(Client(), GAME_ID)

        rs = payload["resultSets"][1]
        game_id_idx = rs["headers"].index("GAME_ID")
        assert all(row[game_id_idx] == GAME_ID for row in rs["rowSet"])
