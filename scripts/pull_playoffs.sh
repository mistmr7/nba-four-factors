#!/usr/bin/env bash
#
# Pull all playoff (and play-in) data and build the processed parquet layer.
#
# Run from the repo root on your machine (needs the project venv and a normal
# home internet connection; stats.nba.com blocks datacenter IPs):
#
#     bash scripts/pull_playoffs.sh            # do it
#     bash scripts/pull_playoffs.sh --dry-run  # preview, no API calls
#
# Every step is idempotent. The fetch steps skip games already on disk, so
# re-running is safe and cheap. Practically, the only gap right now is the
# 2025-26 playoffs that just finished; the other 28 seasons are already pulled
# and will no-op.
#
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from, so
# the relative data/ and scripts/ paths below always resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
  echo "DRY RUN: planning only, no API calls."
fi

CLI="uv run python -m nba_four_factors.cli"

# Current season whose playoffs just completed. incremental always re-fetches
# the schedule (the historical backfill would skip the stale partial schedule
# on resume), then pulls any pending box scores for all season types.
CURRENT_SEASON="2025_26"

echo "==> Step 1: finish the current season (re-fetch schedule + pending box scores)"
$CLI incremental --season "$CURRENT_SEASON" $DRY_RUN

echo "==> Step 2: backfill playoffs and play-in across all seasons (resume fills gaps)"
$CLI backfill --season-type playoffs play_in $DRY_RUN

# Processing is local-disk only and has no --dry-run, so skip it on a dry run.
if [[ -n "$DRY_RUN" ]]; then
  echo "DRY RUN complete. Re-run without --dry-run to fetch and process."
  exit 0
fi

echo "==> Step 3: build processed parquet for playoffs and play-in"
PLAY_IN_FIRST="2020_21"
for season_dir in data/raw/leaguegamelog/*/; do
  season="$(basename "$season_dir")"

  # Playoffs: every season that has a non-empty playoffs schedule.
  if [[ -s "${season_dir}playoffs.json" ]]; then
    echo "  process $season playoffs"
    $CLI process --season "$season" --season-type playoffs || \
      echo "    (skip: $season playoffs not processable yet)"
  fi

  # Play-in: only exists from 2020-21 onward.
  if [[ "$season" > "$PLAY_IN_FIRST" || "$season" == "$PLAY_IN_FIRST" ]]; then
    if [[ -s "${season_dir}play_in.json" ]]; then
      echo "  process $season play_in"
      $CLI process --season "$season" --season-type play_in || \
        echo "    (skip: $season play_in not processable yet)"
    fi
  fi
done

echo "==> Step 4: verify coverage"
uv run python scripts/check_missing_games.py || true
uv run python scripts/integrity_check.py || true

echo "Done. New playoff parquet should appear under data/processed/<season>/playoffs.parquet"
