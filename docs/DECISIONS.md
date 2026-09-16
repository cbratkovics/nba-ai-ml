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

## ADR-0016 — Drift is PSI per model feature against a reference built from the gold marts (prototyped, Phase 3)

The nightly job checks the 14-day window of games played before the run date
(`DRIFT_WINDOW_DAYS`): training-population rows (`min10`, ADR-0009), features from the one
feature module over the stored game logs, one population stability index per feature over
decile bins with an explicit missing-value bin (`nba/drift/psi.py`). The reference is
`reports/drift_reference_<feature_version>_<model_revision>.json`, written by
`python -m nba.drift.reference` from `gold.fct_player_game` (population `min10`, training
seasons; the features are computed by the feature module over those rows, never by a second
SQL implementation, per the one-feature-module rule) and re-included in git; the file name
and content carry `asof_v1` and `fb427de`, so a new model or feature version gets a new
reference. Bin edges are deciles over every training row (86,814); a feature with at most
ten distinct values gets one bin per value. Fewer than `DRIFT_MIN_ROWS` = 500 rows in the
window (the first days of a season, the off-season) is `insufficient`: no PSI, no verdict.
Every run writes `drift/<date>.json` to the dataset repo (per-feature PSI, the window, the
reference block used, the thresholds and whether they were calibrated, the verdict and its
reasons, the slate outcome and the no-schedule streak), which the warehouse copies into
`mart_drift`; logs expire, artifacts are what the next investigation reads. The reference
population is the evaluation population (86,814 rows), not the 88,257 rows the trainer
saw: the trainer keeps season debuts, whose season-to-date mean is undefined, and the
population rule does not (ADR-0009).

## ADR-0017 — The reference is aligned by season day, from seasons with history (prototyped, Phase 3)

Calibrating on the 164 replay dates (every one of them normal data the model was evaluated
on, so any firing is a false positive) found three failures of a naive reference, in order:

1. **A season-long reference fires everywhere, not only at the opening.** `games_played_season`
   is a season counter (PSI 1.4 to 6.4 at every position), the vs-opponent means fill in as
   opponents are met, and `days_rest` moves around the calendar gaps. This is the failure the
   fantasy-football warehouse hit at a season boundary (its ADR-0031), and here it is not
   confined to the boundary.
2. **A week bucket is not enough.** A 21-day reference span for a 14-day window still biased
   the counter (PSI up to 2.1 during the Cup). The reference therefore stores fixed bin
   edges per feature and expected proportions for every season day *s* (days since the
   season's first game date) from the training rows whose season day falls in
   [s − 14, s − 1]: exactly the days the check compares. The check picks `day_NNN` from the
   run date (clipped to the last day the training seasons reached; `all` off-season).
3. **The first season in the data is history-truncated.** 2021-22 has no earlier season in
   the data, so its career-long `<stat>_mean_vs_opp` are missing on 85% of rows at season
   day 15, against 9% to 12% in 2022-23, 2023-24, 2024-25 and 2025-26 alike; PSI 0.35 to 0.38
   on all three at the opening and through the Cup, with the model-relevant distribution
   unchanged. The day-aligned expectations therefore use `DRIFT_REFERENCE_SEASONS`, the
   training seasons with an earlier season in the data (2022-23 to 2024-25); the bin edges and
   the season-long `all` block keep every training season. This departs from the brief's
   "seasons 2021-22 to 2024-25" for the expectations only, and for a reason the numbers show.

With that reference, per position (`reports/drift_calibration_2025-26.json`; PSI over all
date × feature pairs; false positives under the three-feature rule):

| Position | Dates (scored) | PSI median / p90 / max | Largest feature | FP at 0.05 / 0.10 / 0.15 / 0.20 |
|---|---:|---|---|---|
| opening (first 10 game dates) | 10 (3) | 0.021 / 0.040 / 0.046 | pts_mean_last10 | 0 / 0 / 0 / 0 |
| cup (2025-10-31 to 2025-12-15) | 45 (45) | 0.022 / 0.040 / 0.325 | games_played_season | 0 / 0 / 0 / 0 |
| deadline week (2026-02-02 to 02-08) | 7 (7) | 0.025 / 0.058 / 0.077 | games_played_season | 7 / 0 / 0 / 0 |
| All-Star return (7 dates from 02-19) | 7 (7) | 0.035 / 0.078 / 0.161 | days_rest | 7 / 0 / 0 / 0 |
| april | 11 (11) | 0.025 / 0.066 / 0.113 | minutes_mean_last20 | 11 / 0 / 0 / 0 |
| regular (the rest) | 84 (84) | 0.024 / 0.058 / 0.334 | days_rest | 44 / 1 / 0 / 0 |

Seven of the ten opening dates are `insufficient` (under 500 rows in the window); the
remaining single-feature maxima (`games_played_season` 0.33 in the Cup week, `days_rest`
0.33 around the Christmas gap) are what the three-feature rule is for. Regular season only:
the data has no playoffs. The season-position labels come from `SEASON_CALENDAR` (Cup
window, trade-deadline week, All-Star break) and the first ten game dates; only the labels
depend on that calendar, the reference does not.

## ADR-0018 — WARN by default, HOLD only once calibrated, never blocking; no-schedule streak (prototyped, Phase 3)

**Rule.** `HOLD` when at least `min_features` = 3 features have PSI at or above the threshold
in one window; `WARN` when one or two do, or when three or more do while the thresholds are
uncalibrated; `ok` otherwise; `insufficient` under 500 rows. The threshold is the smallest
candidate of 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 with zero false positives on every position:
**0.15** (0.05 would have fired on 69 of 157 scored dates, 0.10 on one, 2026-03-05). The
policy is pure Python with a test per branch (`nba/drift/policy.py`,
`tests/test_drift.py`). Until `reports/drift_calibration_<season>.json` exists the job uses
the provisional 0.20, says "uncalibrated" in every report and cannot HOLD; committing the
calibration (this phase does) is what flips it. **HOLD never blocks the slate**: the
predictions and decisions are written and pushed as usual, the verdict goes to
`drift/<date>.json`, the run summary and `mart_drift`, and the workflow opens one issue
labelled `nightly-hold` (`gh label create --force` first, so the label never has to
pre-exist; an open issue with the same title is reused, so a HOLD that persists opens one
issue, not one per night). The step needs `issues: write`.

**No-schedule streak.** Every drift report records the slate outcome; the streak is the
number of consecutive runs, newest first, that ended at the no-schedule line (AUDIT.md risk
5: the Kaggle author stops publishing `LeagueScheduleYY_YY.csv`). At
`NO_SCHEDULE_STREAK_WARN` = 14 the run is a `WARN` with an issue labelled `nightly-warn`.
Fourteen because nothing in the repo records when `LeagueSchedule25_26.csv` first appeared
in the dump (it was already there on 2026-09-12, the first live run); revise when 2026-27's
file shows up. The off-season "no games" days do not count: only the season's own
no-schedule line does.

**First run.** The dispatched nightly run is an owner step (the branch is not pushed); the
local fallback `python -m nba.nightly --date 2026-09-16 --local-dump data_dump` wrote
`drift/2026-09-16.json`: `insufficient`, 0 rows in the window (off-season), calibrated
threshold 0.15, streak 0, no issue, and `mart_drift` built from it holds the one run row.

**Sensitivity and the insufficient dates.** The calibration carries an injected-drift probe
(`sensitivity`): on 2026-03-24 (regular position, 2,071 rows) shifting three features
(`pts_mean_last10`, `reb_mean_last10`, `ast_mean_last10`) up by half their reference
q10-to-q90 range gives PSI 3.3 to 4.4 and the rule HOLDs naming all three; shifting only
`pts_mean_last10` gives a WARN that names it and does not HOLD. Writing the calibration
fails if either outcome changes. The seven opening dates with fewer than 500 rows in the
window report `insufficient` (no PSI, no verdict) and count as neither a false positive
nor a pass; the artifact lists them per position and the validator refuses one counted as
a false positive.

**The first out-of-sample test.** The rule was calibrated on 2025-26 only. The 2026-27
opening (the first windows with 500 rows, about the season's second week) is the first
time it runs on a season it has not seen. A HOLD there is to be reviewed against the
reference and the calibration, not treated as a bug in the rule or in the data: the
opening reference comes from three prior openings and a fourth may sit outside them.

## ADR-0019 — Agent tools read the exported gold marts, with the files as fallback; return shapes are versioned (prototyped, Phase 4)

The seven-tool contract stands (`get_daily_report`, `get_upstream_freshness`,
`get_residuals`, `get_rolling_metrics`, `get_player_recent`, `get_team_context`,
`list_data_gaps`). Where an exported mart is the better source the tool reads it from
`<root>/gold/<mart>.parquet`, which `pull_products` now brings down with the other product
folders (the nightly job pushes the exports there after the warehouse build), and names it
in `source`; without the export the tool computes the same numbers from the product files,
which is what the Actions runs do until the first warehouse build has pushed `gold/`:

| Tool | Version | Mart when exported | Fallback |
|---|---|---|---|
| `get_daily_report` | 2: adds `drift`, `restatement` | `mart_drift` (latest run on or before the date: status, flagged features by name, streak), `mart_restatement_lag` | `drift/<date>.json` products, then the calibration's per-date row for replay dates; restatement unavailable |
| `get_upstream_freshness` | 2: adds `dump_max_game_date_any_type`, `dump_rows_excluded_by_rules`, `rules_applied` | (stored logs) | the dump, now under the backfill's regular-season rules (game type in `GAME_TYPES`, Cup final excluded), so playoff rows never set the newest date (AUDIT.md risk 9) |
| `get_rolling_metrics` | 2: adds `decisions` | `mart_daily_metrics` (row-weighted over the window, population `all`), `fct_decision_policy` (training-population calls and hit rate to date) | residual files, then the replay daily file; decisions from the residual files with `reports/policy_<season>.json` through the same `nba.decisions.policy` rule |
| the other four | 1 | | unchanged |

Changing a return shape is a versioned, additive change: version-2 results carry
`tool_version`, every version-1 key is still returned, and `tests/test_agent_tools.py`
freezes the version-1 key set per tool. The decisions fallback was checked equal to the
mart on two golden dates (every count and rate identical for all three targets), which is
the reconciliation the marts already carry. `brief/index.json` is built from the dataset
repo's file listing unioned with the local folder (`hf.list_brief_dates`), so a partial
pull can no longer drop dates (risk 8); the listing failing is logged, never fatal.

## ADR-0020 — Golden set version 2: one decision fact and one drift fact per date (prototyped, Phase 4)

`reports/agent_golden.json` is version 2: each date keeps its largest-points-residual
player and gains a decision fact (the policy's pts hit rate to date on the training
population, from the decisions block of `get_rolling_metrics`, with the run date one day
after the brief date as the runners use) and a drift fact (the status word from the drift
block of `get_daily_report`; on off-season dates that is the `insufficient` / streak line).
Both are computed by the tool functions themselves (`python -m nba.agent.golden`), so the
fact is whatever the tool returns. A date passes only with every fact; the evals report
counts each fact separately. The system prompt requires one `decision_policy` and one
`drift` finding, each citing its tool. Pass-rate measurements record the golden version
they were made under; a date measured under an older version is incomplete for the
spreader (`agent-eval.yml`), which re-measures it, and the README row says how many are
stale until then.
