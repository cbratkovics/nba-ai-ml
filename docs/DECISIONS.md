# Decisions

Architecture decision records for the template pass (branch `template`, 2026-09-15 onward).
ADR-0001 to ADR-0008 are the brief's fixed decisions in its numbering; later ADRs record
what building the warehouse decided. Each ADR says whether it is implemented (code exists
and has run in GitHub Actions), prototyped (code and tests exist, has not run in Actions),
or planned.

## ADR-0001 — Decision unit: line-free directional calls (planned, Phase 2)

Not yet written: Phase 2 defines the `over` / `under` / `no_call` policy against the last-10
mean with residual-quantile bands and two causal baselines. No sportsbook lines.

## ADR-0002 — Two populations, headline stays minutes ≥ 10 (implemented, Phase 0)

The headline population is the training population: rows with at least 10 minutes and both
baselines defined (22,244 holdout rows in `reports/metrics.json`). The all-rows comparison
(every replayed row with a box score and a last-10 baseline, 26,031 rows) is published next
to it with its baseline column, from `reports/replay_all_rows_2025-26.json`, and every page
that shows a metric names its population. On all rows the last-10 mean has the lower MAE on
every target; the model's edge exists only on the training population. The warehouse carries
the same two populations as a `population` column (ADR-0009).

## ADR-0003 — Bronze loads from Hugging Face parquet at a pinned revision (prototyped)

`python -m nba.warehouse.load` pulls the five per-season parquet files, the nightly products
and the holdout-season replay residuals from the dataset repo at one revision (the `main`
commit at load time unless `--revision` is given), records it in `load_manifest.json`, and
every `brz_game_logs` row carries it. Bronze never reads the Kaggle CSV: the nightly job's
400 MB dump download stays where it is and the loader adds nothing to it (7.7 MB per load).
Hugging Face remains the system of record; the warehouse is a typed copy that reconciles to
the committed reports. On MotherDuck the same SQL reads the loader's local files through
DuckDB's hybrid execution; that path has not run yet (owner: create database `nba`, add
`MOTHERDUCK_TOKEN`).

## ADR-0004 — Silver restatement lookback: 14 days (prototyped)

`slv_game_logs`, `slv_predictions` and `slv_residuals` are incremental (delete+insert on
their grain keys). An incremental run reprocesses every row whose game date is within 14
days of the newest date already loaded: the daily ingest re-reads 7 days
(`DAILY_LOOKBACK_DAYS`), and 7 more cover a dump refresh that lands late. The value is
chosen, not calibrated: `mart_restatement_lag` records the lag the ingest actually observes
(ADR-0008) and `assert_lookback_covers_restatement_lag` fails when the observed maximum
exceeds the lookback. Recalibrate after 30 in-season daily reports: set the lookback to the
largest observed lag plus 7, and record the counts here. `tests/test_dbt_incremental.py`
proves incremental equals full refresh over the same input, that a restatement inside the
lookback is picked up, and that one outside it is not until `--full-refresh`.

## ADR-0005 — Provenance by file hash (implemented, Phase 0)

`reports/provenance_b20b5601.json` records the SHA-256 of every parquet file at dataset
revision `b20b5601`, of the model files and `metrics.json` at model revision `fb427de`, of
every committed report, and of the replay products the site reads from `main`. One model
identity is displayed everywhere: `commit 50a3b2e / HF fb427de` (`nba/config.py`
`MODEL_COMMIT`, `MODEL_REVISION`, mirrored in `frontend/lib/data.ts` and `dbt_project.yml`,
checked by tests).

## ADR-0006 — Thresholds and bands are in-sample on 2025-26 (planned, Phase 2)

Not yet written.

## ADR-0007 — MotherDuck compute guard (prototyped)

The nightly job builds only the incremental silver models with their parents (bronze copies
of the loaded files, seconds of work) and children (snapshot, gold, tests), plus the three
tiny report copies; a full refresh is the weekly `warehouse.yml` (Sundays 06:00 UTC, or
dispatch). The build runs only when the night ingested, scored or slated something
(`python -m nba.warehouse.gate` reads `data/nightly_summary.json`); an off-season zero-row
run prints `WAREHOUSE skip` and exits 0 without touching the warehouse. Without
`MOTHERDUCK_TOKEN`, or with `DBT_TARGET=local`, both jobs build into `.duckdb/nba.duckdb`
and upload it as a workflow artifact, so the warehouse never blocks on the owner's MotherDuck
setup. CI builds the local target only and never holds the token.

## ADR-0008 — Restatement lag is measured by the ingest (implemented in code, prototyped in the warehouse)

`nba/ingest/kaggle_daily.py` writes `restatement_lag_days` (run date minus game date) on
every changed-row example and a `restatement_lag` summary (count, max, p50, p90 over every
changed row) into `daily_report.json`. `slv_restatements` unnests the examples,
`mart_restatement_lag` summarises them per run date against the lookback, and the silver test
compares the observed maximum with `lookback_days`. Today: four daily reports, zero changed
rows, maximum lag none, lookback 14 — the test passes trivially, which is the point of
logging it.

## ADR-0009 — Population as the warehouse's cohort (prototyped)

There is one model per target and no player cohort, so the cohort dimension of the template
is the population: `min10` (training population) or `all`. Fact rows carry the narrowest
population they belong to (`fct_player_game`: minutes ≥ 10 and at least one earlier game in
the season; `fct_prediction`: the writer's `in_metrics_population` flag), and the metric marts
report one row per population, where `all` aggregates every row and `min10` the flagged
subset. `assert_population_flag_matches_replay` proves the warehouse's history counts
reproduce the feature module's rule on all 26,031 replayed rows with a box score.

## ADR-0010 — Grain is the game date and game id (prototyped)

The template's ordinal `period` seam is replaced by `game_date` (a date) and `game_id`
everywhere: incremental lookbacks are in days, the snapshot's as-of column is a game date,
and `dim_player_asof` gives each team stint an effective date range. Seasons are the
pipeline's labels (`2025-26`), derived in SQL by the same rule as the package
(`season_of` macro: October starts the season).

## ADR-0011 — Reconciliation tolerances and the observed differences (prototyped)

Measured on the first local build (2026-09-15, DuckDB 1.5.5, four threads):

| Report | What is compared | Observed difference | Tolerance |
|---|---|---|---|
| `reports/replay_2025-26.json` | model MAE and n on both populations (22,075 and 26,031 rows) | n exact; MAE within 7.1e-15 (pts), 4.0e-15 (reb), 8.9e-16 (ast) | 1e-6 |
| `reports/replay_all_rows_2025-26.json` | model and baseline MAE and n on 164 dates × 3 targets | n exact on every date; MAE within float noise (see checkpoint) | 1e-6 |
| `reports/metrics.json` | training-population model MAE and last-10 baseline MAE | model +0.002123 (pts), +0.000793 (reb), +0.001011 (ast); baseline −0.000437, −0.001623, +0.000222; n 22,075 vs 22,244 | 0.005; n within 1% |

The replay reports are recomputed from the very rows they summarise, so their tolerance is
float noise (DuckDB's parallel aggregation differs from pandas' mean at the 1e-15 level;
`NBA_DUCKDB_THREADS=1` makes a build reproducible). `metrics.json` scored every holdout row
with both baselines; the replay only the rows the roster rule slated, so the 169 rows never
slated (post-trade debuts, players absent from their team's previous ten games) keep the two
apart by up to 0.0021 and exact equality is not the target. Stated plainly: the warehouse
cannot recompute `metrics.json`, because the training evaluation scored 169 rows the slate
path never produces; `assert_holdout_metrics_reconcile_to_metrics_json` is a
population-restricted comparison, and its 0.005 tolerance was set after observing 0.0021,
not before. The site's pinning asymmetry is
left as-is and recorded here: `metrics.json` is read at the model revision `fb427de`, the
replay files at the dataset repo's `resolve/main`, which holds a later run of the same replay
with identical numbers (provenance file, `hub_replay_json_equals_committed_report: false`).

## ADR-0012 — Season contract with declared exceptions (prototyped)

`mart_season_coverage` and `assert_season_game_counts` require 1,230 distinct games per
completed season unless `dbt/seeds/season_exceptions.csv` declares otherwise: 2024-25 is
1,223 because seven scheduled games have no box-score rows in the dump
(`known_missing_games.csv`, from `docs/reconciliation.md` finding 7), and
`assert_known_missing_games_are_absent` keeps the seed honest. The Cup final is excluded
(`dim_game.is_cup_final` must be false everywhere); every game has 13 to 30 players with a
box score (`accepted_range` on `dim_game.n_players`; observed 16 to 26).

## ADR-0013 — Plausibility warns and cites (prototyped)

`assert_plausible_box_scores` warns (never fails) on a box score above 70 points, 30 rebounds
or 25 assists unless `known_stat_exceptions.csv` lists it with a verification. Seeded: game
`0022500938`, player `1628389`, 83 points, verified with `nba_api boxscoretraditionalv3` on
2026-09-15. Four further rows warn today (Mitchell 71 on 2023-01-02, Lillard 71 on
2023-02-26, Dončić 73 on 2024-01-26, Nurkić 31 rebounds on 2024-03-03); they are added to the
seed only once the owner has verified them.

## ADR-0014 — No dbt packages (prototyped)

The project ships its own two generic tests (`unique_combination`, `accepted_range`) instead
of dbt_utils / dbt_expectations: two macros are cheaper than a package install in every job
and the `dbt deps` quirks the template documents. Revisit if a third package feature is
needed.
