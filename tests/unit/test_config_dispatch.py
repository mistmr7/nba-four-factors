"""Unit tests for the V2/V3 endpoint dispatcher in nba_four_factors.config."""

from __future__ import annotations

from nba_four_factors.config import (
    V3_CUTOVER_SEASON,
    Endpoint,
    endpoint_for_season,
)


class TestV3EnumMembers:
    def test_v3_traditional_value(self):
        assert Endpoint.BOXSCORE_TRADITIONAL_V3.value == "boxscoretraditionalv3"

    def test_v3_advanced_value(self):
        assert Endpoint.BOXSCORE_ADVANCED_V3.value == "boxscoreadvancedv3"

    def test_v3_summary_value(self):
        assert Endpoint.BOXSCORE_SUMMARY_V3.value == "boxscoresummaryv3"

    def test_v2_and_v3_are_distinct_members(self):
        assert Endpoint.BOXSCORE_TRADITIONAL is not Endpoint.BOXSCORE_TRADITIONAL_V3
        assert Endpoint.BOXSCORE_ADVANCED is not Endpoint.BOXSCORE_ADVANCED_V3
        assert Endpoint.BOXSCORE_SUMMARY is not Endpoint.BOXSCORE_SUMMARY_V3


class TestCutoverConstant:
    def test_cutover_is_2025_26(self):
        assert V3_CUTOVER_SEASON == "2025_26"


class TestEndpointForSeasonPreCutover:
    def test_traditional_2023_24_stays_v2(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_TRADITIONAL, "2023_24")
            == Endpoint.BOXSCORE_TRADITIONAL
        )

    def test_advanced_1997_98_stays_v2(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_ADVANCED, "1997_98") == Endpoint.BOXSCORE_ADVANCED
        )

    def test_summary_2024_25_stays_v2(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_SUMMARY, "2024_25") == Endpoint.BOXSCORE_SUMMARY
        )


class TestEndpointForSeasonAtAndAfterCutover:
    def test_traditional_2025_26_becomes_v3(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_TRADITIONAL, "2025_26")
            == Endpoint.BOXSCORE_TRADITIONAL_V3
        )

    def test_advanced_2025_26_becomes_v3(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_ADVANCED, "2025_26")
            == Endpoint.BOXSCORE_ADVANCED_V3
        )

    def test_summary_2025_26_becomes_v3(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_SUMMARY, "2025_26")
            == Endpoint.BOXSCORE_SUMMARY_V3
        )

    def test_traditional_2030_31_becomes_v3(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_TRADITIONAL, "2030_31")
            == Endpoint.BOXSCORE_TRADITIONAL_V3
        )


class TestEndpointForSeasonPassthrough:
    def test_schedule_pre_cutover_unchanged(self):
        assert endpoint_for_season(Endpoint.SCHEDULE, "2019_20") == Endpoint.SCHEDULE

    def test_schedule_post_cutover_unchanged(self):
        assert endpoint_for_season(Endpoint.SCHEDULE, "2025_26") == Endpoint.SCHEDULE

    def test_v3_input_pre_cutover_unchanged(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_TRADITIONAL_V3, "2018_19")
            == Endpoint.BOXSCORE_TRADITIONAL_V3
        )

    def test_v3_input_post_cutover_unchanged(self):
        assert (
            endpoint_for_season(Endpoint.BOXSCORE_TRADITIONAL_V3, "2025_26")
            == Endpoint.BOXSCORE_TRADITIONAL_V3
        )


class TestLexicographicSeasonOrdering:
    """Sanity check the underscore-string season comparison used in dispatch."""

    def test_2024_25_lt_2025_26(self):
        assert "2024_25" < "2025_26"

    def test_2025_26_lt_2026_27(self):
        assert "2025_26" < "2026_27"

    def test_1999_00_lt_2000_01(self):
        assert "1999_00" < "2000_01"
