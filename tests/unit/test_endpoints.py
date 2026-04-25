"""Unit tests for nba_four_factors.api.endpoints.* wrappers.

The wrappers are thin param-shape adapters over Client.fetch. We mock the
client and verify that each wrapper calls fetch with the correct endpoint
and the parameter dict the NBA Stats API expects.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nba_four_factors.api.client import Client
from nba_four_factors.api.endpoints.box_score_advanced import fetch_box_score_advanced
from nba_four_factors.api.endpoints.box_score_advanced_v3 import fetch_box_score_advanced_v3
from nba_four_factors.api.endpoints.box_score_summary import fetch_box_score_summary
from nba_four_factors.api.endpoints.box_score_summary_v3 import fetch_box_score_summary_v3
from nba_four_factors.api.endpoints.box_score_traditional import fetch_box_score_traditional
from nba_four_factors.api.endpoints.box_score_traditional_v3 import (
    fetch_box_score_traditional_v3,
)
from nba_four_factors.api.endpoints.league_game_log import fetch_league_game_log
from nba_four_factors.config import Endpoint, SeasonType


def _mock_client(return_value: dict | None = None) -> MagicMock:
    client = MagicMock(spec=Client)
    client.fetch = MagicMock(return_value=return_value if return_value is not None else {})
    return client


class TestLeagueGameLog:
    def test_regular_season(self):
        client = _mock_client({"ok": 1})
        result = fetch_league_game_log(client, "2023_24", SeasonType.REGULAR)
        assert result == {"ok": 1}
        client.fetch.assert_called_once()
        endpoint_arg = client.fetch.call_args.args[0]
        params_arg = client.fetch.call_args.args[1]
        assert endpoint_arg == Endpoint.SCHEDULE
        assert params_arg["Season"] == "2023-24"
        assert params_arg["SeasonType"] == "Regular Season"

    def test_converts_underscore_season_to_hyphen(self):
        client = _mock_client()
        fetch_league_game_log(client, "1997_98", SeasonType.REGULAR)
        params = client.fetch.call_args.args[1]
        assert params["Season"] == "1997-98"

    def test_playoffs_season_type_value(self):
        client = _mock_client()
        fetch_league_game_log(client, "2019_20", SeasonType.PLAYOFFS)
        params = client.fetch.call_args.args[1]
        assert params["SeasonType"] == "Playoffs"

    def test_play_in_season_type_value(self):
        client = _mock_client()
        fetch_league_game_log(client, "2024_25", SeasonType.PLAY_IN)
        params = client.fetch.call_args.args[1]
        assert params["SeasonType"] == "PlayIn"

    def test_fixed_params_present(self):
        client = _mock_client()
        fetch_league_game_log(client, "2023_24", SeasonType.REGULAR)
        params = client.fetch.call_args.args[1]
        assert params["LeagueID"] == "00"
        assert params["PlayerOrTeam"] == "T"
        assert params["Sorter"] == "DATE"
        assert params["Direction"] == "DESC"
        assert "DateFrom" in params
        assert "DateTo" in params


class TestBoxScoreTraditional:
    def test_endpoint_is_v2(self):
        client = _mock_client()
        fetch_box_score_traditional(client, "0022300001")
        endpoint = client.fetch.call_args.args[0]
        assert endpoint == Endpoint.BOXSCORE_TRADITIONAL
        assert endpoint.value == "boxscoretraditionalv2"

    def test_game_id_required_and_passed_through(self):
        client = _mock_client()
        fetch_box_score_traditional(client, "0042300401")
        params = client.fetch.call_args.args[1]
        assert params["GameID"] == "0042300401"

    def test_defaults_all_range_params_to_zero(self):
        client = _mock_client()
        fetch_box_score_traditional(client, "0022300001")
        params = client.fetch.call_args.args[1]
        for key in ("StartPeriod", "EndPeriod", "StartRange", "EndRange", "RangeType"):
            assert params[key] == "0"

    def test_range_params_override(self):
        client = _mock_client()
        fetch_box_score_traditional(
            client,
            "0022300001",
            start_period="1",
            end_period="4",
            range_type="2",
        )
        params = client.fetch.call_args.args[1]
        assert params["StartPeriod"] == "1"
        assert params["EndPeriod"] == "4"
        assert params["RangeType"] == "2"


class TestBoxScoreAdvanced:
    def test_endpoint_is_v2(self):
        client = _mock_client()
        fetch_box_score_advanced(client, "0022300001")
        endpoint = client.fetch.call_args.args[0]
        assert endpoint == Endpoint.BOXSCORE_ADVANCED
        assert endpoint.value == "boxscoreadvancedv2"

    def test_game_id_passed_through(self):
        client = _mock_client()
        fetch_box_score_advanced(client, "0022300001")
        params = client.fetch.call_args.args[1]
        assert params["GameID"] == "0022300001"

    def test_param_shape_matches_traditional(self):
        client = _mock_client()
        fetch_box_score_advanced(client, "0022300001")
        params_adv = set(client.fetch.call_args.args[1].keys())

        client2 = _mock_client()
        fetch_box_score_traditional(client2, "0022300001")
        params_trad = set(client2.fetch.call_args.args[1].keys())

        assert params_adv == params_trad


class TestBoxScoreSummary:
    def test_endpoint_is_v2(self):
        client = _mock_client()
        fetch_box_score_summary(client, "0022300001")
        endpoint = client.fetch.call_args.args[0]
        assert endpoint == Endpoint.BOXSCORE_SUMMARY
        assert endpoint.value == "boxscoresummaryv2"

    def test_only_game_id_param(self):
        client = _mock_client()
        fetch_box_score_summary(client, "0042300401")
        params = client.fetch.call_args.args[1]
        assert params == {"GameID": "0042300401"}

    def test_returns_payload(self):
        client = _mock_client({"GameSummary": []})
        result = fetch_box_score_summary(client, "0042300401")
        assert result == {"GameSummary": []}


class TestBoxScoreTraditionalV3:
    def test_endpoint_is_v3(self):
        client = _mock_client()
        fetch_box_score_traditional_v3(client, "0022500001")
        endpoint = client.fetch.call_args.args[0]
        assert endpoint == Endpoint.BOXSCORE_TRADITIONAL_V3
        assert endpoint.value == "boxscoretraditionalv3"

    def test_param_shape_matches_v2(self):
        c2 = _mock_client()
        fetch_box_score_traditional(c2, "0022500001")
        v2_params = set(c2.fetch.call_args.args[1].keys())

        c3 = _mock_client()
        fetch_box_score_traditional_v3(c3, "0022500001")
        v3_params = set(c3.fetch.call_args.args[1].keys())

        assert v2_params == v3_params

    def test_range_overrides_apply(self):
        client = _mock_client()
        fetch_box_score_traditional_v3(
            client, "0022500001", start_period="1", end_period="4", range_type="2"
        )
        params = client.fetch.call_args.args[1]
        assert params["StartPeriod"] == "1"
        assert params["EndPeriod"] == "4"
        assert params["RangeType"] == "2"


class TestBoxScoreAdvancedV3:
    def test_endpoint_is_v3(self):
        client = _mock_client()
        fetch_box_score_advanced_v3(client, "0022500001")
        endpoint = client.fetch.call_args.args[0]
        assert endpoint == Endpoint.BOXSCORE_ADVANCED_V3
        assert endpoint.value == "boxscoreadvancedv3"

    def test_param_shape_matches_v2(self):
        c2 = _mock_client()
        fetch_box_score_advanced(c2, "0022500001")
        v2_params = set(c2.fetch.call_args.args[1].keys())

        c3 = _mock_client()
        fetch_box_score_advanced_v3(c3, "0022500001")
        v3_params = set(c3.fetch.call_args.args[1].keys())

        assert v2_params == v3_params


class TestBoxScoreSummaryV3:
    def test_endpoint_is_v3(self):
        client = _mock_client()
        fetch_box_score_summary_v3(client, "0022500001")
        endpoint = client.fetch.call_args.args[0]
        assert endpoint == Endpoint.BOXSCORE_SUMMARY_V3
        assert endpoint.value == "boxscoresummaryv3"

    def test_only_game_id_param(self):
        client = _mock_client()
        fetch_box_score_summary_v3(client, "0042500401")
        params = client.fetch.call_args.args[1]
        assert params == {"GameID": "0042500401"}
