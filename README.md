# NBA Four Factors

NBA analytics pipeline building four-factors-based regression baselines and LSTM-augmented models for playoff series and end-of-season win prediction.

## Status

Active development. Phase 2 (data gathering). Not yet production.

## Requirements

- macOS, Linux, or WSL
- Python 3.12 (managed via uv)
- uv 0.4+ (https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/<user>/nba-four-factors.git
cd nba-four-factors
uv sync
uv run pre-commit install
```

## Project structure

- `nba_four_factors/` — main package (fetcher, storage, orchestration)
- `manifests/` — per-(season, season_type) integrity records (committed)
- `data/` — bronze/silver/gold layers (gitignored, regenerated via fetcher)
- `tests/` — unit + integration tests
- `scripts/` — one-off investigation code

## Data pipeline

Seasons: 1997-98 through 2025-26 (regular season + playoffs).
Endpoints: LeagueGameLog, BoxScoreAdvancedV2, BoxScoreSummaryV2.
Rate limit: 30 requests/minute with jitter.

## License

Not yet licensed. Contact author for terms.
