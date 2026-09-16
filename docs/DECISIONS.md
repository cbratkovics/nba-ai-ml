# Decisions

Architecture decision records for the template pass (branch `template`, 2026-09-15 onward).
ADR-0001 to ADR-0008 are the brief's fixed decisions in its numbering; later ADRs record
what building the warehouse decided. Each ADR says whether it is implemented (code exists
and has run in GitHub Actions), prototyped (code and tests exist, has not run in Actions),
or planned.

## ADR-0001 — Decision unit: line-free directional calls (prototyped, Phase 2)

The decision is one call per (player, game, target) against the player's own last-10-game
mean, the number every slate row already carries: `edge = prediction − baseline_last10`;
`over` when `edge > threshold`, `under` when `edge < −threshold`, else `no_call`. Once the
box score exists the call resolves against the same last-10 mean: `hit` on the called side,
`miss` on the other, `push` when the actual equals the mean (pushes are excluded from the hit
rate and counted). There are no sportsbook lines anywhere in the repo. Two causal baselines
are scored on the rows the model calls: a coin flip (0.5 by definition, with its 95% half
width at that many resolved calls) and the sign of `season_mean − baseline_last10`, the
direction a mean-reverting forecaster would call from the player's season-to-date mean,
also known before tip-off. The rule lives in `nba/decisions/policy.py`; the warehouse
reimplements it in `fct_decision_policy` (v1, aliased to the plain name, exported to
`gold/fct_decision_policy.parquet`) and the nightly job writes `decisions/<date>.json` and
`decisions/latest.json` for the slate from the committed artifact (`nba/decisions/decide.py`),
which `/decisions` reads; off-season it renders the replay evaluation from the artifact and
says so. Population is a column: every row appears under `all`, training-population rows
again under `min10` with that population's threshold and bands (ADR-0009).

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

## ADR-0006 — Thresholds and bands are in-sample on 2025-26 (prototyped, Phase 2)

There is one evaluated season, so the thresholds and bands are chosen on the same replay
rows they are scored on: 22,075 training-population rows and 26,031 all rows
(`reports/replay_2025-26.json` `n_restricted` / `n_with_actuals`; a test pins both).
`reports/policy_2025-26.json` says `in_sample: true` in its own words, and every hit rate
on `/decisions` is labelled in-sample. Bands are quantiles of `actual − prediction` per
target and population (q10, q25, q75, q90 → the 80% and 50% bands around a prediction), so
their in-sample coverage is 0.800 and 0.500 by construction, which the mart recomputes and
the artifact records rather than presents as evidence. The selection rule is ADR-0015. The
first untouched season (2026-27, scored nightly with the same thresholds) is the
out-of-sample test; `mart_policy_metrics` reports the nightly rows next to the replay rows.

## ADR-0007 — MotherDuck compute guard (prototyped)

The nightly job builds only the incremental silver models with their parents (bronze copies
of the loaded files, seconds of work) and children (snapshot, gold, tests), plus the five
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
and `dim_player_asof` gives each team stint an effective date range: derived from the game
logs alone on the first local build (2026-09-15), 2,003 stints for 1,027 players, of which
976 are team changes (a player's second or later stint). Seasons are the
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
2026-09-15. The four other rows above the thresholds (Mitchell 71 on 2023-01-02, Lillard 71
on 2023-02-26, Dončić 73 on 2024-01-26, Nurkić 31 rebounds on 2024-03-03) were verified the
same way on 2026-09-15 and seeded; the warning now reports nothing.

## ADR-0014 — No dbt packages (prototyped)

The project ships its own two generic tests (`unique_combination`, `accepted_range`) instead
of dbt_utils / dbt_expectations: two macros are cheaper than a package install in every job
and the `dbt deps` quirks the template documents. Revisit if a third package feature is
needed.

## ADR-0015 — Threshold selection by a coverage floor, reconciled in both directions (prototyped, Phase 2)

**Rule.** Per target and population the threshold is the largest value on a fixed grid
(`POLICY_THRESHOLD_GRID`: pts 0–8 by 0.25, reb and ast 0–4 by 0.1) whose coverage, the
share of the population's rows called, is still at least `POLICY_MIN_COVERAGE` = 0.25: the
strictest policy that keeps a quarter of the slate. The rule has one constant and no
objective that rewards the model: maximising net correct calls picks threshold 0 (call
everything) on every target because coverage falls faster than the hit rate rises, and
maximising the hit rate picks a handful of rows. The whole coverage curve (every grid
point, both baselines) is in the artifact and on the page so any other point can be read
off. Measured 2026-09-15 on the first local build (`reports/policy_2025-26.json`):

| Population | Target | Threshold | Coverage | Resolved (the one n) | Model hit | Season-mean sign, same rows (abstentions) | Season-mean sign, own threshold (own n) | Coin flip ±95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| min10 (22,075) | pts | 1.75 | 0.306 | 6,680 | 0.667 | 0.592 (356 ties) | 0.588 (7,639 called) | 0.012 |
| min10 | reb | 0.70 | 0.300 | 6,516 | 0.660 | 0.590 (357 ties) | 0.601 (7,122) | 0.012 |
| min10 | ast | 0.50 | 0.274 | 5,914 | 0.633 | 0.595 (310 ties) | 0.602 (6,969) | 0.013 |
| all (26,031) | pts | 2.25 | 0.268 | 6,890 | 0.547 | 0.566 (642 ties, 120 missing) | 0.604 (6,086) | 0.012 |
| all | reb | 0.90 | 0.267 | 6,780 | 0.539 | 0.566 (627 ties, 87 missing) | 0.617 (5,660) | 0.012 |
| all | ast | 0.50 | 0.323 | 8,108 | 0.537 | 0.564 (612 ties, 119 missing) | 0.605 (7,606) | 0.011 |

Both baselines are scored on exactly the model's resolved rows, so each comparison has one
n. Where the season-mean sign has no side it abstains and is scored as a coin flip (0.5):
a *tie* when the season-to-date mean equals the last-10 mean (mostly players whose season
is ten games or fewer, where the two means coincide), and *missing* when there is no season
mean (a season debut; impossible on `min10`, which requires an earlier game in the season).
Both counts are in the artifact (`n_tie`, `n_missing`) and `hit_rate = (n_hit + 0.5 ×
(n_tie + n_missing)) / n`. Scoring abstentions as misses instead would move the baseline by
at most 0.03 and change no verdict. On the training population the model's calls beat both
baselines on every target. On all rows they do not: the season-mean sign hits more often on
the very rows the model calls, and as a policy with its own threshold it is the better call
on all rows (0.60–0.62 at a similar coverage). The artifact's `verdict` says so in one
sentence per target and `/decisions` prints it; the nightly decisions file carries both
populations' calls and the page labels which policy it shows.

**Bands** (residual quantiles, in-sample): min10 pts q10/q25/q75/q90 = −7.05 / −4.11 /
+3.60 / +7.98, reb −2.86 / −1.70 / +1.41 / +3.28, ast −2.00 / −1.21 / +1.00 / +2.41; all
rows pts −7.30 / −5.11 / +2.77 / +7.31, reb −3.18 / −2.10 / +1.07 / +2.98, ast −2.04 /
−1.28 / +0.77 / +2.18. The asymmetry (the model over-predicts on all rows: the median
residual is negative) is the same finding as the all-rows MAE (ADR-0002).

**Where the numbers are computed.** The artifact is written by
`python -m nba.decisions.evaluate` from the built gold marts, `fct_prediction` joined to
`fct_player_game` for the season-to-date mean (not from raw replay files), in pandas.
`fct_player_game.<stat>_mean_season_prior` is the feature module's `<stat>_mean_season`
recomputed in SQL; checked equal on all 130,414 game-log rows (null pattern identical,
maximum difference 0.0) and on the fixture in `tests/test_dbt_gold.py`. The dbt marts
read the artifact through `brz_policy_report` / `brz_policy_curve` (optional, empty typed
relations until the file exists, so a fresh warehouse builds before the artifact and again
after it), apply the thresholds row by row in `fct_decision_policy`, and recompute every
published number in `mart_policy_metrics` and `mart_policy_sweep` (all 230 curve points).
`assert_policy_metrics_reconcile_to_policy_report` and
`assert_policy_sweep_reconciles_to_policy_report` compare counts exactly and rates within
1e-6 (`tol_policy`; the observed difference is 0) and pass on the real build and, inside
`pytest`, on fixture residuals with a fixture artifact, so CI proves SQL and pandas agree
without the Hugging Face load. `frontend/lib/policy_summary.json` is derived from the
artifact and a test recomputes it, as for the all-rows baseline.

**Sequence.** `make dbt-full` → `make policy` → `make dbt-full`: the first build makes the
inputs, the artifact is written from them, the second build reconciles to it. The nightly
selection adds `brz_policy_report+` and `brz_policy_curve+`. The artifact records the code
commit that wrote it (`git_sha`), which is the commit before the one that adds it; the same
is true of every committed report.
