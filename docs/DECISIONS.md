# Decisions

Architecture decision records for the template pass (branch `template`, 2026-09-15 onward)
and the analytics-engineering pass (branch `analytics-engineering`, 2026-10-09 onward).
ADR-0001 to ADR-0008 are the brief's fixed decisions in its numbering; later ADRs record
what building the warehouse decided. Each ADR says whether it is implemented (code exists
and has run in GitHub Actions; the run ids are in `reports/runs.json`), prototyped (code and
tests exist, has not run in Actions), or planned. Statuses were re-checked against the run
evidence on 2026-10-09 (ADR-0022); "implemented" below means a run id exists for it.

## ADR-0001 — Decision unit: line-free directional calls (implemented; decisions are written only on nights with a slate, so none before 2026-27)

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

## ADR-0003 — Bronze loads from Hugging Face parquet at a pinned revision (implemented; ran on MotherDuck in warehouse.yml, run ids in reports/runs.json)

`python -m nba.warehouse.load` pulls the five per-season parquet files, the nightly products
and the holdout-season replay residuals from the dataset repo at one revision (the `main`
commit at load time unless `--revision` is given), records it in `load_manifest.json`, and
every `brz_game_logs` row carries it. Bronze never reads the Kaggle CSV: the nightly job's
400 MB dump download stays where it is and the loader adds nothing to it (7.7 MB per load).
Hugging Face remains the system of record; the warehouse is a typed copy that reconciles to
the committed reports. On MotherDuck the same SQL reads the loader's local files through
DuckDB's hybrid execution; that path first ran on 2026-09-16 (`warehouse.yml` run
`35048092357`, target `motherduck`, PASS=102) and weekly since (`reports/runs.json`).

## ADR-0004 — Silver restatement lookback: 14 days (implemented; the in-season incremental path is rehearsed, not yet run live: ADR-0028)

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

## ADR-0006 — Thresholds and bands are in-sample on 2025-26 (implemented; the out-of-sample test begins with the 2026-27 season)

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

## ADR-0007 — MotherDuck compute guard (implemented; the gate skipped every off-season night, so the nightly build first runs live in 2026-27)

The nightly job builds only the incremental silver models with their parents (bronze copies
of the loaded files, seconds of work) and children (snapshot, gold, tests), plus the five
tiny report copies; a full refresh is the weekly `warehouse.yml` (Sundays 06:00 UTC, or
dispatch). The build runs only when the night ingested, scored or slated something
(`python -m nba.warehouse.gate` reads `data/nightly_summary.json`); an off-season zero-row
run prints `WAREHOUSE skip` and exits 0 without touching the warehouse. Without
`MOTHERDUCK_TOKEN`, or with `DBT_TARGET=local`, both jobs build into `.duckdb/nba.duckdb`
and upload it as a workflow artifact, so the warehouse never blocks on the owner's MotherDuck
setup. CI builds the local target only and never holds the token.

## ADR-0008 — Restatement lag is measured by the ingest (implemented; the mart builds weekly on MotherDuck)

`nba/ingest/kaggle_daily.py` writes `restatement_lag_days` (run date minus game date) on
every changed-row example and a `restatement_lag` summary (count, max, p50, p90 over every
changed row) into `daily_report.json`. `slv_restatements` unnests the examples,
`mart_restatement_lag` summarises them per run date against the lookback, and the silver test
compares the observed maximum with `lookback_days`. Today: four daily reports, zero changed
rows, maximum lag none, lookback 14 — the test passes trivially, which is the point of
logging it.

## ADR-0009 — Population as the warehouse's cohort (implemented)

There is one model per target and no player cohort, so the cohort dimension of the template
is the population: `min10` (training population) or `all`. Fact rows carry the narrowest
population they belong to (`fct_player_game`: minutes ≥ 10 and at least one earlier game in
the season; `fct_prediction`: the writer's `in_metrics_population` flag), and the metric marts
report one row per population, where `all` aggregates every row and `min10` the flagged
subset. `assert_population_flag_matches_replay` proves the warehouse's history counts
reproduce the feature module's rule on all 26,031 replayed rows with a box score.

## ADR-0010 — Grain is the game date and game id (implemented)

The template's ordinal `period` seam is replaced by `game_date` (a date) and `game_id`
everywhere: incremental lookbacks are in days, the snapshot's as-of column is a game date,
and `dim_player_asof` gives each team stint an effective date range: derived from the game
logs alone on the first local build (2026-09-15), 2,003 stints for 1,027 players, of which
976 are team changes (a player's second or later stint). Seasons are the
pipeline's labels (`2025-26`), derived in SQL by the same rule as the package
(`season_of` macro: October starts the season).

## ADR-0011 — Reconciliation tolerances and the observed differences (implemented; the tests pass on every weekly MotherDuck build, PASS=102 on 37201223219)

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

## ADR-0012 — Season contract with declared exceptions (implemented)

`mart_season_coverage` and `assert_season_game_counts` require 1,230 distinct games per
completed season unless `dbt/seeds/season_exceptions.csv` declares otherwise: 2024-25 is
1,223 because seven scheduled games have no box-score rows in the dump
(`known_missing_games.csv`, from `docs/reconciliation.md` finding 7), and
`assert_known_missing_games_are_absent` keeps the seed honest. The Cup final is excluded
(`dim_game.is_cup_final` must be false everywhere); every game has 13 to 30 players with a
box score (`accepted_range` on `dim_game.n_players`; observed 16 to 26).

## ADR-0013 — Plausibility warns and cites (implemented)

`assert_plausible_box_scores` warns (never fails) on a box score above 70 points, 30 rebounds
or 25 assists unless `known_stat_exceptions.csv` lists it with a verification. Seeded: game
`0022500938`, player `1628389`, 83 points, verified with `nba_api boxscoretraditionalv3` on
2026-09-15. The four other rows above the thresholds (Mitchell 71 on 2023-01-02, Lillard 71
on 2023-02-26, Dončić 73 on 2024-01-26, Nurkić 31 rebounds on 2024-03-03) were verified the
same way on 2026-09-15 and seeded; the warning now reports nothing.

## ADR-0014 — No dbt packages (implemented; reaffirmed by ADR-0027)

The project ships its own two generic tests (`unique_combination`, `accepted_range`) instead
of dbt_utils / dbt_expectations: two macros are cheaper than a package install in every job
and the `dbt deps` quirks the template documents. Revisit if a third package feature is
needed.

## ADR-0015 — Threshold selection by a coverage floor, reconciled in both directions (implemented)

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

## ADR-0016 — Drift is PSI per model feature against a reference built from the gold marts (implemented; drift/<date>.json pushed by every nightly run since 2026-09-16)

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

## ADR-0017 — The reference is aligned by season day, from seasons with history (implemented)

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

## ADR-0018 — WARN by default, HOLD only once calibrated, never blocking; no-schedule streak (implemented; the streak reached 9 on 2026-10-09)

**Rule.** `HOLD` when at least `min_features` = 3 features have PSI at or above the threshold
in one window; `WARN` when one or two do, or when three or more do while the thresholds are
uncalibrated; `ok` otherwise; `insufficient` under 500 rows. The threshold is the smallest
candidate of 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 with zero false positives on every position:
**0.15** (of 164 replay dates, 157 were scoreable; 0.05 would have fired on 69 of those
157 and 0.10 on one, 2026-03-05). Zero alerts at 0.15 is an observed calibration-set
result, not a prospective false-positive guarantee. The
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

**First run.** The dispatched nightly run `35048189295` (2026-09-16) and every scheduled run
since wrote and pushed `drift/<date>.json` (first at dataset revision `17fd9fb1`): `insufficient`,
0 rows in the window (off-season), calibrated threshold 0.15, no issue. By 2026-10-09 the
dataset repo held 24 drift reports and the no-schedule streak stood at 9 (`drift/2026-10-09.json`),
so the WARN issue is expected around 2026-10-14 unless the 2026-27 schedule file appears first
(docs/RUNBOOK.md).

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

## ADR-0019 — Agent tools read the exported gold marts, with the files as fallback; return shapes are versioned (implemented; the export they read was frozen at 2026-09-16 until ADR-0022)

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

## ADR-0020 — Golden set version 2: one decision fact and one drift fact per date (implemented; 25 of 25 briefs in agent-eval.yml)

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

## ADR-0021 — The agent traces are committed and replayed in CI (implemented, Phase 5)

Found 2026-09-16 during Phase 4: `tests/traces/*.trace.json` matched the repository's
`*.json` ignore rule, so the six traces recorded on 2026-09-12 were never tracked and the
grounding claim in the README and `docs/agent.md` ("6 of 6 briefs" on "the committed
traces") was, before `3874a29`, backed by files only the recording machine held;
`test_committed_traces_replay_cleanly` returned without asserting when the folder was
empty, so CI passed vacuously and never replayed a trace. What changed: `.gitignore`
re-includes `tests/traces/*.trace.json`; the six traces were re-recorded under the
version-2 golden set and committed (144 KB, key-scrubbed and scanned for `gsk_` and the
organisation id); the test now fails when the folder holds no trace and asserts every
committed trace replays into a grounded brief that passes every golden fact; CI runs it
on every push. The same class of failure (a global or repo-wide ignore silently dropping
files a claim depends on) is what `test_no_warehouse_file_is_git_ignored` guards for the
dbt project (Phase 1). Reading: an artifact a claim rests on must be in the repository,
and a test that skips on "nothing to check" must fail instead.

## ADR-0022 — Run evidence is an artifact; status words are checked against it (implemented, 2026-10-09)

**Finding.** The README, ADR titles, `warehouse.yml` header and `docs/STORIES.md` said
"prototyped" and "no committed run record proves a MotherDuck run" while `warehouse.yml` had
four green runs on target `motherduck` (`35048092357` dispatched 2026-09-16; `35506971007`,
`36317104881`, `37201223219` scheduled) and the nightly had run green every day since
2026-09-12, pushing `drift/<date>.json` each night. Two undersold claims were nonetheless true:
the `gold/` export on the dataset repo was frozen at its first push (`66404be0`, 2026-09-16)
because `warehouse.yml`'s publish step was gated on `inputs.publish != false`, which is false
on `schedule` events (the `inputs` context is empty and both sides coerce to 0), and the
nightly's own warehouse steps had never executed because every off-season night stops at
the gate (ADR-0007).

**Decision.** `reports/runs.json` records every cited run (id, workflow, event, branch,
commit, conclusion, start time, and where known the warehouse target and the Hugging Face
revision it pushed); `scripts/record_run.py <run id>` appends one from `gh api`;
`tests/test_runs.py` fails when a run id cited anywhere in the README, docs or workflow
headers is missing from it. Status words now have one meaning each: **ran** needs a run id
in that file; **deployed** needs a URL that answers; **implemented, not verified** is the
word for code that exists and has not executed in Actions. The README's dbt counts are
rendered from `dbt/target/manifest.json` between `<!-- dbt-counts -->` markers
(`scripts/render_dbt_counts.py`; `tests/test_dbt_counts.py` parses the project and fails
on drift), the same mechanism as the agent pass-rate row. The publish condition is now
`github.event_name == 'schedule' || inputs.publish`; the owner approved the resulting
`gold/` push, and the first run after the fix is to be dispatched and recorded here.

**Status after this ADR.** Weekly MotherDuck build: ran (ids above). Gold export: pushed
once on 2026-09-16, not since; fresh pushes resume with the first scheduled or dispatched
refresh on `main` after this change. Nightly incremental build on MotherDuck: implemented,
rehearsed on a synthetic night (ADR-0028), not yet run live.

## ADR-0023 — The nightly selection is a named selector, and it is the whole project (implemented, 2026-10-09)

The ten-item `--select` list lived twice (`Makefile`, `nightly.yml`) and omitted
`slv_restatements`: `+slv_daily_reports+` selects that model's parents and children, and
`slv_restatements` is a sibling, so on a nightly run `mart_restatement_lag` would have been
rebuilt against a `slv_restatements` last refreshed by the weekly job. `dbt/selectors.yml`
now defines `nightly` once; `make dbt-nightly` and `nightly.yml` use `--selector nightly`;
`tests/test_dbt_selectors.py` resolves it with `dbt ls` and asserts it covers every model,
seed and the snapshot. The definition is deliberately the whole project (`fqn:*`): the
bronze copies and gold marts are seconds of work, the four seeds are a few dozen rows, and
the season relationship tests (ADR-0024) reference the `seasons` seed, so a nightly build
must not depend on a weekly refresh having loaded it. ADR-0007's compute guard remains the
gate that decides *whether* to build; this selector is where to narrow *what* if MotherDuck
compute ever matters. A caveat found on the way: dbt ignores `--exclude` next to
`--selector`, so a build that needs exclusions spells the selection out.

## ADR-0024 — The season list is a seed, tested against the pipeline's rule (implemented, 2026-10-09)

`var('seasons')` in `dbt_project.yml` was read by no model, test or macro; the only season
gate was a hand-typed `accepted_values` list on `slv_game_logs.season` that someone had
already extended to 2026-27 by hand. The var is removed. `dbt/seeds/seasons.csv` is the one
list (season, start year, status, note); `slv_game_logs.season`, `slv_predictions.season` and
`slv_residuals.season` carry `relationships` tests to it with severity `warn` (ADR-0026),
so the first rows of a season the seed does not list warn and open an issue rather than
fail the build; `tests/test_project_config.py` asserts the seed's complete seasons equal
`config.SEASONS`, every row obeys the October rule, and the current season is present
(so the test, not the warehouse, is what fails when a season is forgotten). A date/season
dimension (Phase 4) will be built from this seed.

## ADR-0025 — One silver contract: slv_game_logs (implemented, 2026-10-09)

Contracts were enforced on gold only. `slv_game_logs` now has one too, and the other four
silver models do not, because: every gold mart and the feature parity proof (Phase 3) read
`slv_game_logs`, so a column type drifting there would move published numbers; it is
incremental with `on_schema_change='fail'`, so a contract is the only *typed* statement of
what the table is between refreshes; and its columns are the canonical game-log schema
(`nba/schema.py`), already fixed. `slv_predictions` and `slv_residuals` are typed copies of
the product files whose shapes the `product_columns` macros already pin, and
`slv_daily_reports` / `slv_restatements` are small operational tables; contracting them would
add forty lines of YAML and no new guarantee. Revisit if a second consumer reads them.

## ADR-0026 — Severity policy: a wrong published number blocks, an operational threshold warns (implemented, 2026-10-09)

**error** (fails the build, nothing exports): grain uniqueness, contracts, the
reconciliation tests to the committed reports, `assert_known_missing_games_are_absent`,
`assert_season_game_counts`. **warn** (build and export continue; the workflow opens one
idempotent issue labelled `warehouse-warn` per warning test through `nba/warehouse/notify.py`,
the same protocol as the drift notifier): `assert_plausible_box_scores` (already),
`assert_lookback_covers_restatement_lag` (changed from error: a lag beyond the lookback means
"raise `lookback_days` and full-refresh", docs/RUNBOOK.md, not "the numbers are wrong"), and
the season relationship tests (ADR-0024). `nightly.yml` and `warehouse.yml` run the notifier
after every build (`issues: write`). `store_failures` stays off until Phase 4, where the
reconciliation tests get it so a failure leaves the disagreeing rows behind.

## ADR-0027 — sqlfluff lints the dbt project; ADR-0014 stands (implemented, 2026-10-09)

`sqlfluff==4.4.0` with `sqlfluff-templater-dbt==4.4.0` resolve against the pinned
`dbt-core==1.12.5` / `dbt-duckdb==1.11.0` and run in `make lint-sql` and CI. The config
(`.sqlfluff`) matches the style the models already had: lower-case keywords, four-space
indents, 120-character lines, Jinja blocks that do not indent their contents; rules that
fight DuckDB struct access, quoted keywords or the project's deliberate one-line selects
are excluded by name with the reason recorded in the file. The first lint found 326
violations under default settings and 27 under this config, all whitespace; `sqlfluff fix`
applied them to eight files and a token-level comparison of every compiled model and test
before and after showed no change. sqlfluff is a Python dev dependency, not a dbt package,
so ADR-0014 (no dbt packages) stands unchanged.

## ADR-0028 — The nightly's warehouse steps are rehearsed on a synthetic game night (implemented, 2026-10-09; run id to be recorded)

**Problem.** Steps 10–14 of `nightly.yml` (gate, target, loader, incremental `dbt build`,
`export_gold`, `publish`) had executed zero times: every run since 2026-09-16 was an
off-season night the gate skipped. Opening night (2026-10-20) would have been their first
execution, with the ledger depending on it.

**Design.** `nightly.yml` gains a `rehearsal` dispatch input. `python -m nba.rehearsal`
pulls the real game logs (a read), writes a dump directory with the real `TeamHistories.csv`,
a header-only `PlayerStatistics.csv` (the daily ingest sees an empty window and changes
nothing) and a synthetic `LeagueSchedule26_27.csv` that re-dates the most recent real game
night to the rehearsal date with `0022600001`-style ids, then runs `nba.nightly.run` with
push disabled and no agent key. The slate, decisions, daily report and drift report are
written locally; the summary says "slate written", so the gate says build; the target step
is forced to `local`; a first `dbt build --full-refresh` creates the scratch warehouse from
the published files, `nba.rehearsal stage-products` copies the night's products where
`pull_products` would have put them, and the normal `dbt build --selector nightly` then runs
as a genuine incremental pass over new nightly rows; `export_gold` runs; the publish step
runs `--dry-run`; the DuckDB file, the export and both dbt logs are uploaded as an artifact.
Nothing can reach Hugging Face: `nba.storage.hf` refuses to build a write client while
`NBA_REHEARSAL` is set, and `tests/test_rehearsal.py` proves it.

**What it proves and what it does not.** It proves the code path of steps 10–14 end to end
on real data shapes, including the bronze families switching from empty typed relations to
real files and the incremental models accepting a season no stored row carried.
`tests/test_season_readiness.py` proves the same on fixtures in CI, plus the second night's
residuals and a restatement inside the lookback. It does **not** exercise `delete+insert` on
MotherDuck itself (the weekly full refresh does exercise MotherDuck, but with
`--full-refresh`); a rehearsal against a scratch MotherDuck database (`NBA_MOTHERDUCK_DATABASE`)
would close that gap and is offered to the owner as an option, since it writes to MotherDuck.

**Record.** Dispatched from the branch on 2026-10-10 (`reports/runs.json`): run `38008812416`
failed in the rehearsal step (the downloaded `TeamHistories.csv` was copied onto itself; fixed
in the next commit); run `38008927282` **succeeded**: synthetic schedule of 15 games, 483
player predictions and a decisions file, `WAREHOUSE build: slate written`, target `local`,
first pass `PASS=106`, three product files staged, incremental pass `PASS=106`, export
(`fct_prediction` 38,855 rows = 38,372 replay + 483 nightly, `mart_drift` 25), publish
dry-run listing 14 files. Nothing reached Hugging Face.

**Scratch MotherDuck (owner decision, 2026-10-10).** `rehearsal_target=motherduck` builds into
the scratch database `nba_rehearsal` (`CREATE DATABASE IF NOT EXISTS` before the build,
`DROP DATABASE … CASCADE` in a final `always()` step; the helpers refuse any name outside
`nba_rehearsal*`). Run `38011203767` **succeeded**: the same synthetic night, both passes on
`target='motherduck'` (`PASS=106` each; the second a real `delete+insert` over the staged
products), export, dry-run publish, database dropped. The gap named above is closed; the
MotherDuck incremental path has run, on a scratch copy, against real data shapes.

## ADR-0029 — Opening night depends on one Kaggle uploader for both the schedule and the box scores (proposed, 2026-10-09; decision in Phase 2)

Checked with the owner's own Kaggle API key on 2026-10-09: the dump
(`eoinamoore/historical-nba-data-and-player-box-scores`) is at version 515, **last updated
2026-06-15**, and lists `LeagueSchedule24_25.csv` and `LeagueSchedule25_26.csv` but no
`LeagueSchedule26_27.csv`. Two consequences, in order of severity: (1) without the schedule
file there is no slate and so no ledger (`SLATE: no schedule file`, streak WARN at 14);
(2) if the author does not resume daily updates once games are played, no box scores
arrive either, so residuals, settlement of paper bets, drift windows and the incremental
build all stall; the pipeline would exit 0 every night with nothing to do.

Free, legitimate fallbacks found (docs/AE_AUDIT.md §9): for the **schedule**, The Odds API
`GET /v4/sports/basketball_nba/events` costs 0 credits under its documented terms and returns
upcoming games with teams and tip-off times, and balldontlie's `/games` endpoint is on its
free tier with permissive terms; either can be mapped to the schedule loader's shape through
a team-name seed. For **box scores** no documented free source reachable from a runner was
found (balldontlie stats need a paid tier; nba.com hosts are blocked; ESPN's JSON is
undocumented and Disney's terms forbid automated access), so settlement depends on the
Kaggle author resuming, and the ledger must be designed to settle late rather than never.
**Decision (owner, 2026-10-10): The Odds API `/events` is the schedule fallback.** Implemented
in Phase 2 (`nba/lines/schedule.py`, `nba/nightly.py` step 3): when the dump has no schedule
file for the date's season and `ODDS_API_KEY` is set, the night's events (0 credits) become
a schedule frame in the loader's layout with the Eastern-time game date, both teams mapped
to the game logs' abbreviations (`nba/lines/teams.py`, errors on an unknown name), a
provisional game id `evt:<event id>` that can never collide with an nba.com id, the tip-off
in UTC and `schedule_source = odds_api`; the Kaggle file wins whenever it exists. Event ids
are not NBA game ids, so the mapping to the real id happens when the box scores arrive: by
(game date, team), since a team plays at most once per date (`resolve_game_ids`, tested in
`tests/test_lines_schedule.py`); the residual step keys the box score by the resolved id,
carries it as `resolved_game_id` next to the slate's id, and reports the rows that stayed
provisional (`n_unresolved_provisional_games` in the residual line and the run summary). The
warehouse joins `fct_prediction.box_score_game_id` for every settlement. Every slate row now
also carries `tip_off_utc`, `event_id` and `schedule_source` from either source.

The box-score side has no fallback (ADR-0031 designs for it: bets settle whenever the box
scores arrive, and a `source-stalled` issue is distinct from no-schedule).

## ADR-0030 — The pre-registered paper-bet rule, version 1 (FINAL, 2026-10-10: approved by the owner with five amendments, before the first live date 2026-10-20)

The rule is data in `nba/ledger/rule.py` (`RULE`, hash `ae7b1a2e00534761ff4814fc9fa536d4a10282c370a9ccb1ba176b52931a917f`), mirrored as
dbt vars that `tests/test_project_config.py` pins to the same hash, so the recommendations
view (Python) and `fct_paper_bet` (SQL) apply one rule. It is committed before the first live
date it covers and is never tuned on 2026-27; a change is `ledger_rule_v2` with its own ADR.

**Scope.** Markets: through the API, `player_points` only; by hand (`source = manual`),
points, rebounds and assists. Books: DraftKings through the API (Caesars is paid-only on The
Odds API free plan, docs/AE_AUDIT.md §9.1); DraftKings or Caesars by hand. Line source and
book are columns on every bet and the marts report `api`, `manual` and `all` separately.

**Snapshot.** One scheduled capture a day at 20:00 UTC (16:00 EDT, 15:00 EST), every game
of the night. Games that commenced before the snapshot are out of scope by rule and counted
(`mart_ledger_coverage.games_out_of_scope`); a manual dispatch uses its own time as the
snapshot. Predictions are stamped by the nightly at 10:00 UTC, so `prediction_generated_at
< snapshot_at_utc < tip_off_utc` holds by construction and is tested
(`assert_predictions_precede_their_lines` fails the build; `assert_no_bet_violates_the_timestamp_rule`).

**Eligibility (all three known before tip-off), from the 2025-26 replay
(`reports/ledger_rule_evidence_2025-26.json`):**
- at least **1 prior game this season** (`games_played_season >= 1`): the training
  population requires an earlier game (both baselines defined), and on opening night this
  alone makes every row ineligible, which is expected: 2026-10-20 is a smoke test of
  capture, matching and timestamps, not a betting night;
- **last-10 minutes mean ≥ 15.0** as the pre-game proxy for the 10-minute cut the model was
  evaluated on. At that floor **95.7% of the proxy's rows are min10 rows** (precision) and it
  **keeps 86.3% of the min10 population** (recall); the model's points MAE on the proxy
  set is 4.989 against the last-10 baseline's 5.062 (keys
  `proxy.table[minutes_floor=15]`). The neighbours: a 12-minute floor is 93.4% / 92.6%, a
  20-minute floor 98.1% / 70.5%. Fifteen is where the proxy stops admitting
  rows the model never saw in evaluation (under 5 % impurity) while keeping most of the
  population; it is a proxy, and the marts report the realised population separately;
- **inputs at most 10 days old** (`input_staleness_days`, the feature module's `days_rest`
  at prediction time). In the training population the median staleness is 2 days,
  p99 16; 98.2% of rows have inputs at most 10 days old. The cap is **not** about
  accuracy — the model still beats the baseline at 8–10 days (4.658 vs 5.048, n=431) and
  11–14 days (4.256 vs 4.750, n=135) — it is about what a gap longer than the
  All-Star break (about 8 days) means: the player was out, and the model has no
  availability or minutes-restriction information. Ten days keeps the season's scheduled
  gaps and excludes most injury returns.

**Decision.** `edge = prediction − line`. **Over** when `edge ≥ threshold`, **under** when
`−edge ≥ threshold`, else pass. Thresholds per market are the committed `min10` policy
thresholds already pre-registered on the replay (`reports/policy_2025-26.json`
`populations.min10.targets.<target>.threshold`): points **1.75**, rebounds **0.70**, assists
**0.50**. No historical line was available at $0 to tune an edge against a book, so the one
pre-registered edge constant per target is reused unchanged; that is the honest position and
it is what makes the season out of sample.

**Stake (amendment 1).** One unit is risked per bet, every bet; the profit on a win comes
from the captured American price on the bet side (−110 pays 0.909 units, +120 pays 1.2);
a loss costs the unit. `units_staked` and `units_profit` are additive sums; `roi_units` is
their ratio and is never averaged across rows.

**Settlement (amendment 2).** `pending` while the game's box score is **absent** from the
stored logs; `void` only when the game's box score is **present and the player has no row
in it** (did not play); a row with any minutes played, even under one, means the action
stands and the bet settles on the line; `won` / `lost` on the bet side of the line; `push`
when the actual equals a whole-number line. Settlement is computed in `fct_paper_bet` from
the facts on every build, never stored, so it is idempotent and backfills itself when a late
box score arrives. The dbt unit test `fct_paper_bet_settles_pending_void_and_sub_minute_action`
covers the absent game (pending), the present game without the player (void), a 0.7-minute
row (settled on the line) and an exact whole-line hit (push).

**Side bias (amendment 3).** The model is a LightGBM regressor trained on squared error
(`nba/config.py` `LGBM_PARAMS`, LightGBM's default `regression` objective), that is, a
conditional **mean**. A book's line sits near the **median**, and points are right-skewed,
so a mean-type prediction tends to sit above the line and the rule may lean to overs.
Nothing is adjusted for it; `mart_ledger_by_side` reports hit rate, n and ROI for overs and
unders separately so the lean and its cost, if any, are visible.

**Reporting and the reading rule (amendment 4).** `mart_ledger` (hit rate against the
line with a 95 % half-width, the break-even hit rate implied by the prices actually taken,
ROI in units with a half-width, and a `verdict`), `mart_ledger_by_side`, `mart_ledger_by_edge`,
`mart_ledger_coverage` (captured nights against game nights, bets per night, unmatched lines,
pending counts). Rates are ratios of sums; every row carries its n and
`evaluation = out_of_sample`; in-sample results stay in the replay marts and are never
pooled. **Headline numbers are `source_group = api` only (`is_headline`); manual lines never
enter them.** The verdict: **"too early" under 100 settled bets**; above that, the interval
and the break-even rate are shown and a verdict ("beats break-even" / "below break-even") is
given only when the hit rate's 95 % interval excludes the break-even rate, else
"inconclusive". How long that takes: to separate a true 55 % hit rate from the 52.4 %
break-even of −110 prices at 95 % confidence needs
n = (1.96 × √(0.55 × 0.45) / (0.55 − 0.524))² = (1.96 × 0.4975 / 0.026)² ≈ 37.5² ≈ **1,400
settled bets**; at 100 settled bets the half-width is ±0.10, at 400 it is ±0.05. A season of
every-night DraftKings points lines will not reach 1,400 qualifying bets, so the expected
honest verdict for 2026-27 is "inconclusive" unless the edge is large.

**Privacy guard (amendment 5).** `nba.lines.store.push` reads the lines repo's visibility
before every commit and refuses to push unless it is private; `ensure_private_repo` creates
it private and refuses a public one; `tests/test_lines_store.py` proves both.

## ADR-0031 — Budgeted, bias-guarded line capture and the private ledger (implemented, 2026-10-10)

**Budget.** The Odds API free plan: 500 credits a month; `/events` is free; a per-event
props call costs `markets × region units` and a `bookmakers` filter of up to ten books is
one unit (docs/AE_AUDIT.md §9.1–9.2). With points only and DraftKings only, one game costs
one credit, an average night 7.5 and a 15-game night 15; every game every night fits with
margin (about 200 of 500 a month), which is why the owner allowed a schedule. The guard
(`nba/lines/guard.py`) never plans past `425` (the cap less a 15 % margin); a scheduled run
stops at `375`, keeping `50` in reserve for a manual dispatch with `--use-reserve`. Before
the night's priced calls it computes the plan's cost, reads the month's spend from the
private usage log, reads the provider's own `x-requests-remaining` from the last answer (the
provider's cycle is not the calendar month, so the smaller figure wins), and refuses in
writing when either would be crossed. Every call and every refusal is logged with the cost
the provider reported (`usage/odds_api.jsonl`). Nothing retries: a failed event call is
logged once and skipped. `tests/test_lines_guard.py` proves the refusal paths;
`tests/test_lines_capture.py` proves the refusal happens before any priced call.

**Season guard (owner question, 2026-10-10).** No priced call is made for anything that is
not a 2026-27 regular-season game. Two independent checks, both before the plan is costed:
the event's `sport_key` must be `basketball_nba` (The Odds API lists preseason, All-Star and
summer-league games under their own keys, `basketball_nba_preseason` and so on), and the
event's Eastern game date must fall inside the season's configured regular-season window
(`config.REGULAR_SEASON_WINDOWS`: 2026-27 opens 2026-10-20; the end date is set when the
league schedule is published, and until then the window is open-ended and must be closed
before the play-in). A capture dated outside the window ends `out_of_season` with the
events listed and no priced call; an out-of-season event on a regular-season date is counted
and never priced. The same classifier filters the nightly's schedule fallback, so a preseason
game never produces a slate either. `tests/test_lines_capture.py::test_no_priced_call_out_of_season_and_preseason_keys_are_never_priced`
and `tests/test_lines_schedule.py::test_regular_season_window_and_fallback_drop_preseason`
prove both. Every run logs one `EVENT …` line per event the endpoint returned with its verdict.

**No cherry-picked nights.** The capture writes and pushes `captures/<date>.json` (date,
policy version, trigger, snapshot time, `decided_before_fetch: true`) before the first
priced call; once that record exists every qualifying line of the night enters the ledger.
The policy is `capture_policy_v1`: the schedule runs every day at the fixed time; a manual
dispatch exists only to capture a night the schedule missed, and is recorded as such.
`mart_ledger_coverage` reports captured nights against game nights on every ledger page,
and a captured night cannot be deleted from the record (the private repo keeps history).

**Private storage.** Raw lines, prices, capture records, the usage log and the
recommendations live in the private dataset repo `cbratkovics/nba-lines` (The Odds API's
terms permit storing the data and publishing derived values; they prohibit redistributing it
as a raw data product). The warehouse loader pulls them with the token; CI has no token and
builds the bronze copies empty and typed. `brz_lines`, `slv_lines` and `fct_paper_bet` are
`meta: export: false` and never leave the warehouse; `mart_ledger*` carry aggregates only.

**Matching.** Sportsbook names to `player_id` deterministically (`nba/lines/matching.py`):
a normaliser (accents, periods, apostrophes, Jr/Sr/II/III), an exact match on the normalised
name within the player's team roster for this and last season, then league-wide when unique,
then the committed alias file (`nba/lines/aliases.csv`, confirmed mappings with a date and
note). Anything else is `unmatched`, counted in the capture record and in
`mart_ledger_coverage.lines_unmatched_player`, never guessed. Team names map through
`nba/lines/teams.py`, which errors on an unknown name.

**Manual lines.** `python -m nba.lines.manual ingest <csv>`: the template is validated row by
row (ISO timestamp with zone, date, book, market, half-or-whole line in range, American
prices, no duplicates), matched by the same step, written as `lines/<date>_manual.parquet`
with `source = manual`, and reported separately as well as together.

**Source stall.** Distinct from no-schedule: when slated nights go by without a single box
score (`stall_streak` from the rolling residual lines, `STALL_WARN = 2`), the nightly logs
`SOURCE … (source stalled)` and the workflow opens one issue labelled `source-stalled`.
Predictions and captures continue; bets stay `pending` and settle whenever the dump resumes
(settlement is computed, never stored). `mart_ledger_coverage.bets_pending` is on every page.

**Staleness and eligibility stamps.** Every slate row carries `minutes_mean_last10`,
`input_staleness_days` (the feature module's `days_rest` for the pending row) and
`prediction_generated_at`, so the rule's eligibility is decided from the prediction itself.

**Recommendations (private).** `python -m nba.ledger.recommend` writes
`recommendations/<date>.json` for a captured night: player, market, line, prices, prediction,
edge, decision, the reasons a row does not qualify, and on every row and in the header the
plain statement that the model has no availability information.

**Workflow.** `.github/workflows/capture.yml`: scheduled at 20:00 UTC and dispatchable
(date, `use_reserve`, `dry_run`); needs the `ODDS_API_KEY` secret (owner step); until it
exists a scheduled run stops green with a notice. Nothing in it writes to the public repo.

