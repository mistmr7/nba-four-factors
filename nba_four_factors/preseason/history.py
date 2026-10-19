"""Player-season history and opening-night rosters.

Opening-night roster for season S: every player active for his team's first
regular-season game plus every player listed inactive for that game. The
inactive list is published pregame and exists from 2005-06; before that the
roster is first-game actives only. Nothing about who actually got minutes
later in the season leaks in.
"""

from __future__ import annotations

import pandas as pd

from nba_four_factors.preseason import inputs as io


def player_seasons() -> pd.DataFrame:
    """One row per player-season: minutes, games, main team, minutes per team game.

    Minutes are divided by the games his main team actually played, so the
    shortened 2019-20 and 2020-21 seasons sit on the same scale as full ones.
    """
    gs = pd.read_parquet(
        io.PS / "player_game_score.parquet",
        columns=["person_id", "team_id", "minutes", "season", "season_type"],
    )
    gs = gs[gs.season_type == "regular_season"]
    by_team = gs.groupby(["season", "person_id", "team_id"]).minutes.sum().reset_index()
    main = by_team.sort_values("minutes").drop_duplicates(["season", "person_id"], keep="last")
    agg = (
        gs.groupby(["season", "person_id"])
        .agg(minutes=("minutes", "sum"), gp=("minutes", lambda m: int((m > 0).sum())))
        .reset_index()
    )
    agg = agg.merge(
        main[["season", "person_id", "team_id"]].rename(columns={"team_id": "main_team"}),
        on=["season", "person_id"],
    )
    games_played = []
    for season in sorted(gs.season.unique()):
        g = io.load_games(season)
        games_played.append(
            g.groupby("team_id").size().rename("sched").reset_index().assign(season=season)
        )
    tg = pd.concat(games_played, ignore_index=True)
    agg = agg.merge(tg, left_on=["season", "main_team"], right_on=["season", "team_id"], how="left")
    agg = agg.drop(columns="team_id")
    agg["sched"] = agg.sched.fillna(agg.season.map(io.schedule_length))
    agg["mpg"] = agg.minutes / agg.sched
    agg["gp_frac"] = agg.gp / agg.sched
    agg = agg.sort_values(["person_id", "season"])
    agg["n_prior"] = agg.groupby("person_id").cumcount()
    return agg.reset_index(drop=True)


def opening_roster(season: str) -> pd.DataFrame:
    """Players on each team's roster for its first game of `season`."""
    games = io.load_games(season)
    first = games.sort_values("game_date").drop_duplicates("team_id")[["team_id", "game_id"]]
    ro = pd.read_parquet(
        io.PS / "rosters" / f"{season}_regular_season.parquet",
        columns=["game_id", "team_id", "person_id"],
    )
    act = ro.merge(first, on=["game_id", "team_id"])[["team_id", "person_id"]]
    ina = pd.read_parquet(
        io.DATA / "features" / "inactives.parquet",
        columns=["game_id", "season", "team_id", "person_id"],
    )
    ina = ina[ina.season == season].merge(first, on=["game_id", "team_id"])
    act["out_g1"] = 0.0
    listed = ina[["team_id", "person_id"]].assign(out_g1=1.0)
    out = pd.concat([act, listed]).drop_duplicates(["team_id", "person_id"], keep="first")
    out["season"] = season
    return out.reset_index(drop=True)


def all_opening_rosters(seasons: list[str]) -> pd.DataFrame:
    return pd.concat([opening_roster(s) for s in seasons], ignore_index=True)
