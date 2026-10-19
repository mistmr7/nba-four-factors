# 2026-27 preseason win totals: locked picks

Locked 19 October 2026, about 12:00 Mountain Time, before the season's
first game (20 October, 3 p.m. ET). The git commit that adds this folder is
the timestamp of record; the SHA-256 of the picks file is in
SHA256SUMS.txt.

## What is here

- preseason_win_totals_2026_27.csv: one row per team.
- inputs/live_win_totals.csv: the two line sources as collected.
- inputs/live_injuries_2026_27.csv: expected games missed, hand-converted
  from public injury reports (sources in the file).

Model code: nba_four_factors/preseason/ at the commit that adds this
folder. Method and backtest: docs/Preseason_WinTotals.md (local).

## Columns

- line_boyds, line_fanatics: Boyd's Bets consensus (best available over,
  updated 17 Oct) and Fanatics (published 16 Oct). line_consensus is
  their average and is the line the model's blend step was fed.
- proj_wins_primary: the pre-registered primary projection (base model,
  blended with the consensus line). proj_wins_model_only: the player model
  alone, before the blend.
- edge_vs_boyds, pick_primary: primary projection minus the Boyd's line;
  OVER when positive, UNDER otherwise (an edge of exactly zero is UNDER).
- price_boyds, breakeven_pct: the American price for the picked side on
  Boyd's page and the win rate needed to break even at that price. Boyd's
  lists only over prices; under prices are Boyd's own estimates from the
  over and a 4.8 percent margin (price_is_estimated = True).
- proj_wins_candidate_team_age, pick_candidate_team_age: the candidate
  that adds roster age to the calibration. It was found while looking at
  holdout residuals, so it is labeled a candidate, not the primary. The
  two disagree on one team (DEN, both within half a win of the line).

## Grading rules, set before the season

1. Over/under: every one of the 30 primary picks is graded against
   line_boyds and the final regular-season win total. Pushes are dropped.
   Reported overall, by direction (over vs under), and for picks with an
   edge of 2 or more wins.
2. Accuracy: mean absolute error of proj_wins_primary vs actual wins,
   compared with the MAE of line_consensus.
3. The candidate is graded the same way and reported beside the primary.
4. Profit claim: a pick set is called profitable only if its hit rate
   beats the average breakeven of the prices it was graded at (51.0
   percent across these 30 picks at Boyd's prices). One season is 30
   correlated picks; a single season cannot establish an edge, it can
   only add to the record.

## Honest framing of the backtest these picks rest on

The backtest's 57 percent holdout hit rate (2016-17 to 2025-26, 300 picks)
is against Covers' preseason consensus lines with no prices attached, so
it means 57 percent against a reconstructed consensus that could not
necessarily have been bet at that number. Breakeven at standard -110 juice
is 52.4 percent; at -115 it is 53.5 percent.

## Overlay: 2027 lottery reform (added 19 Oct 2026, before tip-off)

overlay_lottery_reform_2026_27.csv adjusts the primary projections for the
new lottery, which first applies to the 2027 draft: 16 lottery teams
sharing 37 balls; the three worst records get 2 balls each (5.4 percent
for No. 1) and cannot fall below No. 12; the next seven non-playoff teams
get 3 balls (8.1 percent); the No. 9 and No. 10 seeds get 2 balls; the
losers of the 7 vs 8 play-in games get 1 ball. The worst teams are now
penalized and the 4th through 10th worst have flat odds.

Sizing comes from history (scripts/preseason_tank_dip.py):
over games 61 to 82, teams in the bottom 3 by early-season margin fall
3.1 to 4.5 points a game short of what their early margin predicts, about
2.3 to 3.3 wins; teams ranked 4th to 10th fall 2.0 to 3.9 points short,
about 1.5 to 2.9 wins. Assumed recovered under the new rules: 2.5 wins for
the projected bottom 3 (the incentive now runs the other way) and 0.8 for
projected 4th to 10th (no lottery reason to sink within a flat tier, but
deadline selling and development minutes remain). Wins are zero-sum, so
the 13.1 added wins are taken evenly from all 30 teams (0.44 each).

This is a judgment overlay, not a backtested model, and is graded
separately from the primary picks. It flips three picks, all within half
a win of the line: DEN and CLE to UNDER, CHI to OVER. The second
incentive (9th and 10th seeds getting fewer balls than the 4th to 10th
worst, so teams near the play-in line may prefer to miss it) is not
applied; it is left to be measured this season.
