# Known Anomalies

Documents season- and game-level anomalies that affect expected game counts,
schedule parsing, and downstream four-factors aggregates. The pipeline treats
the schedule endpoint as authoritative for `expected_game_count`; everything
below exists so that deviations from 82 games/team (or from clean home/away
semantics) are understood, not silently smoothed over.

Covered span: 1997-98 through 2025-26.

## Lockout seasons

### 1998-99 — Lockout

- Regular season: 50 games per team (725 league-wide)
- Dates: Feb 5, 1999 – May 5, 1999
- All-Star Game canceled; no preseason
- Pipeline notes:
  - `expected_game_count` per team = 50, not 82
  - Per-possession four-factors remain comparable across seasons;
    season totals are not
  - Back-to-back density is elevated relative to non-lockout seasons

### 2011-12 — Lockout

- Regular season: 66 games per team (990 league-wide)
- Dates: Dec 25, 2011 – Apr 26, 2012
- Pipeline notes:
  - `expected_game_count` per team = 66
  - Extreme schedule compression: elevated back-to-backs and
    3-games-in-4-nights; rest-based features will skew for this season
  - Travel/rest logic (REST_CAP=7) still applies but distribution shifts

## Canceled games preserved in schedule data

### 2012_13: BOS vs IND, April 16, 2013 (game_id: 0021201214)

The Boston Celtics home game against the Indiana Pacers was canceled by
the NBA following the Boston Marathon bombings on April 15, 2013. The
game was not rescheduled. This is the most recent NBA season with an
odd number of games played (1,229 actual; 1,230 scheduled).

The nba.com `leaguegamelog` endpoint includes the canceled game in the
schedule with `pts=0` and `opp_pts=0` for both teams. The processed
Parquet preserves this representation. Downstream analysis should
filter rows where `pts == 0 AND opp_pts == 0`.

Sources:
- https://www.si.com/nba/2013/04/16/boston-marathon-bombing-terror-attack-celtics-pacers-game-cancelled-nba
- https://www.cbc.ca/sports/basketball/nba/celtics-pacers-game-cancelled-due-to-boston-marathon-bombings-1.1364069

## COVID-19 disruptions

### 2019-20 — Bubble season

- Regular season suspended Mar 11, 2020
- Resumed Jul 30, 2020 in Orlando with 22 of 30 teams
  (8 eliminated teams did not resume)
- 8 seeding games per participating team; full playoffs in bubble
- Per-team game counts range roughly 64–75 depending on when the team
  stood at suspension and whether they went to the bubble
- Pipeline notes:
  - `expected_game_count` is per-team and non-uniform for this season —
    do NOT assume a single league-wide value
  - All bubble games are neutral-site (Jul 30 - Oct 11, 2020). nba.com schedules them with normal vs./@ MATCHUP fields, so the `is_neutral` flag is enforced by `anomalies.NEUTRAL_SITE_DATE_OVERRIDES` and applied in `processed/schedule.py::_tidy_schedule`. Home-court
    advantage features should be suppressed for all `is_neutral=True` rows, which now correctly includes the bubble seeding games and the 2019-20 playoffs  - Travel distance = 0 for all bubble games regardless of
    `METRO_MAP` lookup; REST_CAP still applies
  - First play-in game in league history (Blazers vs. Grizzlies)
    played in bubble

### 2020-21 — Shortened season, play-in introduced

- Regular season: 72 games per team (1,080 league-wide)
- Dates: Dec 22, 2020 – May 16, 2021
- Pipeline notes:
  - `expected_game_count` per team = 72
  - Heavy mid-season rescheduling due to COVID protocols; schedule
    endpoint rows for postponed games may carry stale dates until
    the rescheduled instance is published
  - Attendance was zero or limited at most venues for most of the
    season; home-court advantage is depressed and should be
    modeled as a distinct regime
  - Play-in tournament becomes permanent from this season forward

### 2021-22 — Omicron wave

- Regular season: full 82 games per team
- Mid-December 2021 Omicron wave produced widespread player
  unavailability and mass postponements; no games were ultimately
  canceled
- Pipeline notes:
  - `expected_game_count` per team = 82 (normal)
  - Hardship-exception signings inflated rosters transiently; player
    participation variance in Dec 2021 – Jan 2022 is abnormal but
    team-game rows are intact
  - Postponed games may appear as duplicate schedule rows (original
    date + rescheduled date); dedupe on `game_id`, keep the instance
    with a completed box score

## Specific game anomalies

Game-level exclusions are tracked in `nba_four_factors/anomalies.py` under
`ANOMALOUS_GAME_IDS`. No entries at this time. Populate as concrete
`game_id` values are identified during schedule reconciliation.

## Pipeline-wide handling rules

1. Schedule endpoint is truth for `expected_game_count` per team per season.
   Never hardcode 82.
2. Per-possession rates are the cross-season-comparable unit; totals are not.
3. Missing or anomalous games are excluded from four-factors aggregates —
   never interpolated. Exclusion list lives in `nba_four_factors/anomalies.py`
   as `ANOMALOUS_GAME_IDS`.
4. Neutral-site games (bubble, NBA Paris Games, Mexico City Games, etc.)
   require a `neutral_site` flag; home-court advantage features are suppressed
   for those rows.
5. Postponed/rescheduled games: dedupe on `game_id`, keep the played instance.

## Validation via reference boxscores

Traditional boxscores are the computation source for four factors; Advanced
boxscores serve as reference cross-check. Small systematic divergence
(rounding, possession-formula variant) is expected. Divergence that changes
shape across seasons is a bug signal in our computation, not a reason to
adopt Advanced numbers.

## Out of scope for v1

Tracked for future anomaly documentation but not blocking Phase 2:
- Non-bubble neutral-site regular-season games (Paris, Mexico City, London)
- Franchise relocations and ID continuity in `nba_api`
  (SEA→OKC 2008, NJN→BKN 2012, NOH→NOP 2013, CHA Bobcats→Hornets 2014)
- Vancouver→Memphis relocation (pre-2001, outside season span but worth noting)
