"""End-to-end integration test for fetch_box_score_summary against a recorded cassette.

Exercises the full stack against the V3 summary endpoint:
client (with rate limiter + retry) -> endpoint wrapper -> recorded HTTP response.

Cassette: tests/fixtures/cassettes/boxscoresummaryv3_0022500590.yaml
Recorded against: stats.nba.com/stats/boxscoresummaryv3,
2025-26 regular season game 0022500590 (LAC @ TOR, 2026-01-16).

Same game as the traditional and advanced integration tests.

Test strategy (differs from traditional/advanced):
- The summary envelope has 34 top-level keys covering officials, line
  scores, broadcaster metadata, series context, replay flags, and other
  scheduling/presentation fields. Many of these are likely to evolve
  season-to-season as NBA adds new metadata.
- Asserting `set(...) == EXPECTED` would create constant test breakage
  on benign API additions. Instead this test asserts:
    * `.issubset()` for the keys our pipeline depends on
      (PIPELINE_REQUIRED_KEYS at the top level)
    * Tight `==` for nested structures we structurally depend on
      (homeTeam/awayTeam team-key set)
    * Type/shape checks where it matters (officials is a list of NBA
      crew size, inactives is a list under each team)
- This is intentionally a thinner contract than traditional/advanced.
  The four-factor pipeline reads structured stats from those endpoints;
  summary provides metadata for schedule reconciliation and context.
"""

from __future__ import annotations

from nba_four_factors.api.client import Client
from nba_four_factors.api.endpoints.box_score_summary import (
    fetch_box_score_summary,
)

CASSETTE_NAME = "boxscoresummaryv3_0022500590"
GAME_ID = "0022500590"

EXPECTED_TOP_LEVEL_KEYS = {"boxScoreSummary", "meta"}

# Keys our pipeline depends on. Asserted via .issubset(); the API may
# add additional metadata fields without breaking this contract.
PIPELINE_REQUIRED_KEYS = {
    "arena",
    "attendance",
    "awayTeam",
    "awayTeamId",
    "gameClock",
    "gameEt",
    "gameId",
    "gameStatus",
    "gameTimeUTC",
    "homeTeam",
    "homeTeamId",
    "lastFiveMeetings",
    "officials",
    "period",
    "seriesGameNumber",
    "seriesText",
}

EXPECTED_TEAM_KEYS = {
    "inBonus",
    "inactives",
    "periods",
    "players",
    "score",
    "seed",
    "statistics",
    "teamCity",
    "teamId",
    "teamLosses",
    "teamName",
    "teamSlug",
    "teamTricode",
    "teamWins",
    "timeoutsRemaining",
}

NBA_CREW_SIZE = 3


class TestBoxScoreSummaryIntegration:
    def test_payload_top_level_keys(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        assert set(payload.keys()) == EXPECTED_TOP_LEVEL_KEYS

    def test_pipeline_required_keys_present(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        bx = payload["boxScoreSummary"]
        assert PIPELINE_REQUIRED_KEYS.issubset(bx.keys())

    def test_game_id_matches_request(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        assert payload["boxScoreSummary"]["gameId"] == GAME_ID

    def test_home_and_away_team_schema(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        bx = payload["boxScoreSummary"]
        assert set(bx["homeTeam"].keys()) == EXPECTED_TEAM_KEYS
        assert set(bx["awayTeam"].keys()) == EXPECTED_TEAM_KEYS

    def test_team_id_denorms_match_team_objects(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        bx = payload["boxScoreSummary"]
        assert bx["homeTeamId"] == bx["homeTeam"]["teamId"]
        assert bx["awayTeamId"] == bx["awayTeam"]["teamId"]

    def test_officials_is_list_of_crew_size(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        officials = payload["boxScoreSummary"]["officials"]
        assert isinstance(officials, list)
        assert len(officials) == NBA_CREW_SIZE

    def test_inactives_is_list_per_team(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        bx = payload["boxScoreSummary"]
        for side in ("homeTeam", "awayTeam"):
            assert isinstance(bx[side]["inactives"], list)

    def test_last_five_meetings_shape(self, recorded):
        with recorded(CASSETTE_NAME):
            payload = fetch_box_score_summary(Client(), GAME_ID)

        # Replaces V2's LastMeeting (single game). V3 wraps the list in a
        # dict: {"meetings": [<game>, <game>, ...]}. May have fewer than
        # 5 entries early in a series or for first-time matchups.
        last_five = payload["boxScoreSummary"]["lastFiveMeetings"]
        assert isinstance(last_five, dict)
        assert "meetings" in last_five
        assert isinstance(last_five["meetings"], list)
        assert len(last_five["meetings"]) <= 5
