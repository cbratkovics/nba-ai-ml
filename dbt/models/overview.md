{% docs __overview__ %}

# nba_dbt — analytics warehouse for the NBA stat predictor

The pipeline predicts NBA player points, rebounds and assists nightly from a player's earlier
games, publishes every product to a Hugging Face dataset repo, and serves a site from those
files. This is the analytics layer over the same files: a bronze / silver / gold medallion
built with dbt Core and dbt-duckdb, DuckDB locally and in CI, MotherDuck (`nba`) from the
nightly job. Grain is the game date and game id, not a week.

## Layers

**Bronze** (`brz_*`) — typed one-to-one copies of the source files pulled by
`python -m nba.warehouse.load`: per-season game logs, nightly predictions and residuals, daily
ingest reports, the holdout-season replay residuals, and the committed reports
(`reports/metrics.json`, `reports/replay_<season>.json`, `reports/replay_all_rows_<season>.json`).
Every row carries `source_file`; the game logs carry the dataset-repo revision of the load.

**Silver** (`slv_*`) — grain-enforced; the data contracts are its tests. `slv_game_logs`,
`slv_predictions` and `slv_residuals` are incremental (delete+insert with a 14-day restatement
lookback); the unique tests on `(player_id, game_id)` and `(player_id, game_id, model_revision)`
are what the writers only imply. A plausibility test warns on a box score above 70 points, 30
rebounds or 25 assists unless the `known_stat_exceptions` seed lists it with a verification.

**Snapshot** (`snp_player_team`) — SCD2 history of each player's current team.

**Gold** — contracted marts: `dim_game`, `dim_team`, `dim_player_asof` (team stints, the
roster rule's as-of answer), `dim_player_current` (snapshot view), `fct_player_game` (box
scores with the population flag), `fct_prediction` (prediction, actual, errors of the model
and of the last-10 baseline), `mart_holdout_metrics` and `mart_daily_metrics` (both
populations), `mart_season_coverage` (the 1,230-game contract with declared exceptions),
`mart_restatement_lag` (observed restatement lag against the lookback).

## Populations

Every fact and metric mart carries `population`: `min10` is the training population (at
least 10 minutes and both baselines defined), `all` is every row with a box score. The model
beats the last-10 baseline on `min10` and loses to it on `all`; the marts report both.

## How trust is established

- `assert_holdout_metrics_reconcile_to_replay_report` and
  `assert_daily_metrics_reconcile_to_all_rows_report` recompute the replay's numbers from
  the residual rows and fail on any disagreement above float noise (1e-6).
- `assert_holdout_metrics_reconcile_to_metrics_json` checks the training-population MAE
  against `reports/metrics.json` within 0.005 (the populations differ by the 169 rows the
  roster rule never slated; ADR-0011 records the observed gap).
- `assert_population_flag_matches_replay` proves the warehouse's history counts reproduce the
  feature module's population rule row for row.
- `assert_replay_roster_rule_reproduces_from_asof` proves the replay's team assignments
  reproduce from the stint table.
- Contracts on every gold model; slim CI on pull requests; docs published to GitHub Pages.

- Repository: https://github.com/cbratkovics/nba-ai-ml
- Site: https://nba-ai-ml.vercel.app

{% enddocs %}
