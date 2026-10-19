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
