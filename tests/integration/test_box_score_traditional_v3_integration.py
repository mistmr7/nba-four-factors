"""End-to-end integration test for fetch_box_score_traditional_v3 against a recorded cassette.

Exercises the full stack against a V3 per-game endpoint:
client (with rate limiter + retry) -> endpoint wrapper -> recorded HTTP response.

Cassette: tests/fixtures/cassettes/boxscoretraditionalv3_0022500590.yaml
Recorded against: stats.nba.com/stats/boxscoretraditionalv3,
2025-26 regular season game 0022500590 (LAC @ TOR, 2026-01-16).

Schema notes:
- V3 envelope is {"boxScoreTraditional": {...}, "meta": {...}} — completely
  different from V2's {"resource": ..., "parameters": ..., "resultSets": [...]}.
- V3 is hierarchical (nested objects keyed by homeTeam/awayTeam) rather
  than tabular. Processed-layer normalization will need to flatten this.
- V3 column names are camelCase (fieldGoalsMade, reboundsOffensive,
  turnovers) versus V2's SCREAMING_SNAKE (FGM, OREB, TO).
- starters and bench are 19-key dicts that mirror statistics minus
  plusMinusPoints (which doesn't aggregate cleanly across player subgroups).
"""

from __future__ import annotations

from nba_four_factors.api.client import Client
from nba_four_factors.api.endpoints.box_score_traditional_v3 import (
    fetch_box_score_traditional_v3,
)

CASSETTE_NAME = "boxscoretraditionalv3_0022500590"
GAME_ID = "0022500590"

EXPECTED_TOP_LEVEL_KEYS = {"boxScoreTraditional", "meta"}

EXPECTED_BOX_SCORE_KEYS = {
    "awayTeam",
    "awayTeamId",
    "gameId",
    "homeTeam",
    "homeTeamId",
}

EXPECTED_TEAM_KEYS = {
    "bench",
    "players",
    "starters",
    "statistics",
    "teamCity",
    "teamId",
    "teamName",
    "teamSlug",
    "teamTricode",
}

EXPECTED_TEAM_STATISTICS_KEYS = {
    "assists",
    "blocks",
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "fieldGoalsPercentage",
    "foulsPersonal",
    "freeThrowsAttempted",
    "freeThrowsMade",
    "freeThrowsPercentage",
    "minutes",
    "plusMinusPoints",
    "points",
    "reboundsDefensive",
    "reboundsOffensive",
    "reboundsTotal",
    "steals",
    "threePointersAttempted",
    "threePointersMade",
    "threePointersPercentage",
    "turnovers",
}

EXPECTED_STARTERS_BENCH_KEYS = EXPECTED_TEAM_STATISTICS_KEYS - {"plusMinusPoints"}

EXPECTED_PLAYER_KEYS = {
    "comment",
    "familyName",
    "firstName",
    "jerseyNum",
    "nameI",
    "personId",
    "playerSlug",
    "position",
    "statistics",
}

EXPECTED_PLAYER_STATISTICS_KEYS = EXPECTED_TEAM_STATISTICS_KEYS

FOUR_FACTOR_INPUT_KEYS = {
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersMade",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "reboundsOffensive",
    "reboundsDefensive",
    "turnovers",
    "points",
}

PLAYER_COUNT_MIN = 8
PLAYER_COUNT_MAX = 15


class TestBoxScoreTraditionalV3Integration:
    def test_payload_top_level_keys(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        assert set(payload.keys()) == EXPECTED_TOP_LEVEL_KEYS

    def test_box_score_object_keys(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        assert set(payload["boxScoreTraditional"].keys()) == EXPECTED_BOX_SCORE_KEYS

    def test_game_id_matches_request(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        assert payload["boxScoreTraditional"]["gameId"] == GAME_ID

    def test_home_and_away_team_keys_match(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        bx = payload["boxScoreTraditional"]
        assert set(bx["homeTeam"].keys()) == EXPECTED_TEAM_KEYS
        assert set(bx["awayTeam"].keys()) == EXPECTED_TEAM_KEYS

    def test_team_id_denorms_match_team_objects(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        bx = payload["boxScoreTraditional"]
        assert bx["homeTeamId"] == bx["homeTeam"]["teamId"]
        assert bx["awayTeamId"] == bx["awayTeam"]["teamId"]

    def test_team_statistics_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        bx = payload["boxScoreTraditional"]
        assert set(bx["homeTeam"]["statistics"].keys()) == EXPECTED_TEAM_STATISTICS_KEYS
        assert set(bx["awayTeam"]["statistics"].keys()) == EXPECTED_TEAM_STATISTICS_KEYS

    def test_starters_and_bench_schemas(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        bx = payload["boxScoreTraditional"]
        for side in ("homeTeam", "awayTeam"):
            assert set(bx[side]["starters"].keys()) == EXPECTED_STARTERS_BENCH_KEYS
            assert set(bx[side]["bench"].keys()) == EXPECTED_STARTERS_BENCH_KEYS

    def test_four_factor_inputs_present_in_team_statistics(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        bx = payload["boxScoreTraditional"]
        assert FOUR_FACTOR_INPUT_KEYS.issubset(bx["homeTeam"]["statistics"].keys())
        assert FOUR_FACTOR_INPUT_KEYS.issubset(bx["awayTeam"]["statistics"].keys())

    def test_player_lists_have_plausible_size(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        bx = payload["boxScoreTraditional"]
        for side in ("homeTeam", "awayTeam"):
            n = len(bx[side]["players"])
            assert PLAYER_COUNT_MIN <= n <= PLAYER_COUNT_MAX

    def test_player_object_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_traditional_v3(Client(), GAME_ID)

        bx = payload["boxScoreTraditional"]
        for side in ("homeTeam", "awayTeam"):
            for player in bx[side]["players"]:
                assert set(player.keys()) == EXPECTED_PLAYER_KEYS
                assert set(player["statistics"].keys()) == EXPECTED_PLAYER_STATISTICS_KEYS
