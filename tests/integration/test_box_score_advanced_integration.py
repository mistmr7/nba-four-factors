"""End-to-end integration test for fetch_box_score_advanced against a recorded cassette.

Exercises the full stack against the V3 advanced endpoint:
client (with rate limiter + retry) -> endpoint wrapper -> recorded HTTP response.

Cassette: tests/fixtures/cassettes/boxscoreadvancedv3_0022500590.yaml
Recorded against: stats.nba.com/stats/boxscoreadvancedv3,
2025-26 regular season game 0022500590 (LAC @ TOR, 2026-01-16).

Same game as the traditional and summary integration tests, so the three
cassettes form a coherent "what one game looks like across all V3
endpoints" set.

Schema notes:
- V3 envelope is {"boxScoreAdvanced": {...}, "meta": {...}}, hierarchical
  (homeTeam/awayTeam nesting) like traditional but WITHOUT starters/bench
  aggregations. Advanced reports team-level + per-player only.
- 24 statistics keys per team, including 6 estimated* variants
  (estimatedPace, estimatedOffensiveRating, etc.).
- The estimated* variants return implausible values across the entire
  date range, including 2025-26 (e.g. estimatedPace 18.84 vs pace 94.19,
  estimatedOffensiveRating 568 vs offensiveRating 113.6). They are
  asserted PRESENT here (schema lock) but the processed layer will
  filter them out — see Section 2.4.
- Four-factor reference metrics (effectiveFieldGoalPercentage,
  turnoverRatio, offensiveReboundPercentage) are present at the team
  level and used for cross-validation against the four-factor inputs
  computed from boxscoretraditionalv3.
"""

from __future__ import annotations

from nba_four_factors.api.client import Client
from nba_four_factors.api.endpoints.box_score_advanced import (
    fetch_box_score_advanced,
)

CASSETTE_NAME = "boxscoreadvancedv3_0022500590"
GAME_ID = "0022500590"

EXPECTED_TOP_LEVEL_KEYS = {"boxScoreAdvanced", "meta"}

EXPECTED_BOX_SCORE_KEYS = {
    "awayTeam",
    "awayTeamId",
    "gameId",
    "homeTeam",
    "homeTeamId",
}

EXPECTED_TEAM_KEYS = {
    "players",
    "statistics",
    "teamCity",
    "teamId",
    "teamName",
    "teamSlug",
    "teamTricode",
}

EXPECTED_TEAM_STATISTICS_KEYS = {
    "PIE",
    "assistPercentage",
    "assistRatio",
    "assistToTurnover",
    "defensiveRating",
    "defensiveReboundPercentage",
    "effectiveFieldGoalPercentage",
    "estimatedDefensiveRating",
    "estimatedNetRating",
    "estimatedOffensiveRating",
    "estimatedPace",
    "estimatedTeamTurnoverPercentage",
    "estimatedUsagePercentage",
    "minutes",
    "netRating",
    "offensiveRating",
    "offensiveReboundPercentage",
    "pace",
    "pacePer40",
    "possessions",
    "reboundPercentage",
    "trueShootingPercentage",
    "turnoverRatio",
    "usagePercentage",
}

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

# Per-player statistics omits estimatedTeamTurnoverPercentage (a team-
# aggregate metric with no per-player analog). 23 keys vs the team's 24.
EXPECTED_PLAYER_STATISTICS_KEYS = EXPECTED_TEAM_STATISTICS_KEYS - {
    "estimatedTeamTurnoverPercentage"
}

# Reference metrics published by the API; the processed layer also
# computes its own four-factor values from boxscoretraditionalv3 and
# uses these for cross-validation.
FOUR_FACTOR_REFERENCE_KEYS = {
    "effectiveFieldGoalPercentage",
    "turnoverRatio",
    "offensiveReboundPercentage",
}

ESTIMATED_VARIANT_KEYS = {
    "estimatedDefensiveRating",
    "estimatedNetRating",
    "estimatedOffensiveRating",
    "estimatedPace",
    "estimatedTeamTurnoverPercentage",
    "estimatedUsagePercentage",
}

PLAYER_COUNT_MIN = 8
PLAYER_COUNT_MAX = 15


class TestBoxScoreAdvancedIntegration:
    def test_payload_top_level_keys(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        assert set(payload.keys()) == EXPECTED_TOP_LEVEL_KEYS

    def test_box_score_object_keys(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        assert set(payload["boxScoreAdvanced"].keys()) == EXPECTED_BOX_SCORE_KEYS

    def test_game_id_matches_request(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        assert payload["boxScoreAdvanced"]["gameId"] == GAME_ID

    def test_home_and_away_team_keys_match(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        bx = payload["boxScoreAdvanced"]
        assert set(bx["homeTeam"].keys()) == EXPECTED_TEAM_KEYS
        assert set(bx["awayTeam"].keys()) == EXPECTED_TEAM_KEYS

    def test_team_id_denorms_match_team_objects(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        bx = payload["boxScoreAdvanced"]
        assert bx["homeTeamId"] == bx["homeTeam"]["teamId"]
        assert bx["awayTeamId"] == bx["awayTeam"]["teamId"]

    def test_team_statistics_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        bx = payload["boxScoreAdvanced"]
        assert set(bx["homeTeam"]["statistics"].keys()) == EXPECTED_TEAM_STATISTICS_KEYS
        assert set(bx["awayTeam"]["statistics"].keys()) == EXPECTED_TEAM_STATISTICS_KEYS

    def test_four_factor_reference_keys_present(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        bx = payload["boxScoreAdvanced"]
        assert FOUR_FACTOR_REFERENCE_KEYS.issubset(bx["homeTeam"]["statistics"].keys())
        assert FOUR_FACTOR_REFERENCE_KEYS.issubset(bx["awayTeam"]["statistics"].keys())

    def test_estimated_variants_present_in_schema(self, recorded):
        # Locked: estimated* variants are part of the API schema even though
        # they return implausible values. Processed layer filters them.
        # This test ensures the schema doesn't drift silently — if the API
        # ever fixes or removes these, we want to know.
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        bx = payload["boxScoreAdvanced"]
        assert ESTIMATED_VARIANT_KEYS.issubset(bx["homeTeam"]["statistics"].keys())

    def test_player_lists_have_plausible_size(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        bx = payload["boxScoreAdvanced"]
        for side in ("homeTeam", "awayTeam"):
            n = len(bx[side]["players"])
            assert PLAYER_COUNT_MIN <= n <= PLAYER_COUNT_MAX

    def test_player_object_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_advanced(Client(), GAME_ID)

        bx = payload["boxScoreAdvanced"]
        for side in ("homeTeam", "awayTeam"):
            for player in bx[side]["players"]:
                assert set(player.keys()) == EXPECTED_PLAYER_KEYS
                assert set(player["statistics"].keys()) == EXPECTED_PLAYER_STATISTICS_KEYS
