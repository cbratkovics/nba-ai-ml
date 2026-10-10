# Runbook

What each signal from the nightly and weekly jobs means, what to do about it, and when a
full refresh is required. Signals arrive as GitHub issues (labels below), as lines in the
job summary, or as a red run. Every number a decision rests on should be read from the
artifact named, not from the log.

## Where to look first

| Where | What it tells you |
|---|---|
| Actions → the run → "Job summary" | the `NIGHTLY / INGEST / RESIDUALS / AGENT / SLATE / DECISIONS / DRIFT` outcome lines, the `WAREHOUSE build/skip` verdict, and dbt's `Done. PASS=… WARN=… ERROR=…` line |
| `data/nightly_summary.json` (run artifact `nightly-<run id>`) | the same facts as JSON, plus the list of product files pushed and the dataset revision |
| `drift/<date>.json` in the dataset repo | the drift verdict, PSI per feature, the slate outcome and the no-schedule streak |
| `gold/_export_manifest.json` in the dataset repo | which run exported the marts, when, from which commit, on which target |
| `reports/runs.json` | every cited run with its conclusion; append with `scripts/record_run.py <run id>` |

## Issues the jobs open

All issues are idempotent: an open issue with the same title is reused, so a condition that
persists for days produces one issue. Close the issue when the condition is gone.

### `nightly-warn` — "Nightly WARN: no schedule file for the season"

**Means:** `NO_SCHEDULE_STREAK_WARN` (14) consecutive nightly runs ended at `SLATE: no
schedule file for season <season>`; the Kaggle dump has no `LeagueSchedule<yy>_<yy>.csv` for
the current season. Expected around 2026-10-14 for 2026-27 (the streak was 9 on
2026-10-09) unless the file appears first. Without a schedule there is no slate, no
decisions file and no paper ledger.

**Do:** check the dump's file list (the nightly log line `NIGHTLY <date>: downloaded […];
schedule file …` names what it found). If the author has published the file, the next run
clears the streak by itself. If not, the schedule fallback decided in ADR-0029 (The Odds
API `/events`, 0 credits, or balldontlie `/games`) is the way forward; until it exists the
job will keep exiting 0 and reusing this issue.

### `nightly-hold` — "Nightly HOLD: feature drift"

**Means:** at least three model features had PSI ≥ 0.15 against the day-aligned reference
in the 14-day window (ADR-0018). The slate and decisions were still written; HOLD never
blocks. Early in a season the reference for `day_NNN` comes from three earlier openings and
a fourth may sit outside them (ADR-0018, "the first out-of-sample test").

**Do:** open `drift/<date>.json`: the `flagged` list and each feature's PSI. Compare the
window's distribution with the reference block for that season day
(`reports/drift_reference_asof_v1_fb427de.json`). A HOLD on `games_played_season` or
`days_rest` alone is calendar, not data; a HOLD on the rolling means of pts/reb/ast after a
rule change or a data outage is the one to take seriously. Record what you concluded in the
issue; the rule and thresholds are not to be re-tuned on this season (they are the
pre-registered out-of-sample test).

### `warehouse-warn` — "Warehouse WARN: <test name>" (ADR-0026)

One issue per dbt test with severity `warn` that returned rows. The build and the export
were not blocked.

- **`assert_lookback_covers_restatement_lag`** — the ingest restated a row older than
  `lookback_days` (14), so the incremental silver models did not pick it up.
  **Do:** run a full refresh (below), then raise `lookback_days` in `dbt/dbt_project.yml` to
  the observed maximum plus 7 and note the counts in ADR-0004, as it says. The observed lags
  are in `mart_restatement_lag` (`max_lag_days`, `p90_lag_days`).
- **`relationships_slv_*_season__season__ref_seasons_`** — rows carry a season the
  `seasons` seed does not list (ADR-0024). **Do:** add the row to `dbt/seeds/seasons.csv`
  (season, start year, status, note); `tests/test_project_config.py` will have been failing
  on the current season too. The next build loads the seed (it is in the nightly selection).
- **`assert_plausible_box_scores`** — a box score above 70 points, 30 rebounds or 25
  assists that the `known_stat_exceptions` seed does not list. **Do:** verify the line
  against another source (the owner's machine can reach nba.com; runners cannot) and seed it
  with the verification, as `docs/reconciliation.md` finding 9 did.

### `source-stalled` — "Nightly WARN: box scores not arriving (source stalled)" (ADR-0031)

**Means:** two or more consecutive slated nights scored no box score at all
(`n_with_actuals = 0` in `predictions/rolling_metrics.json`). The Kaggle dump has not
published those games; this is the second ADR-0029 risk, distinct from "no schedule" (the
schedule came from The Odds API, the slate ran, the lines were captured, nothing settles).

**Do:** check the dataset's `lastUpdated` on Kaggle. Nothing in the pipeline needs
changing: predictions and captures continue, paper bets stay `pending` and settle on the
first build after the box scores arrive (settlement is computed from the facts, never
stored). `mart_ledger_coverage.bets_pending` shows how much is waiting. If the dump has
stopped for good, there is no free, documented box-score source (docs/AE_AUDIT.md §9.5) and
the ledger cannot settle; say so on the ledger page rather than finding a scraper.

## Lines and the paper ledger (ADR-0030, ADR-0031)

| Signal | Means | Do |
|---|---|---|
| `capture.yml` summary `CAPTURE <date>: REFUSED by the credit guard (...)` | the night's plan would cross the monthly ceiling (375 scheduled, 425 manual) or the provider reports fewer credits left than the margin keeps back | nothing tonight; the night is recorded as `refused` and counts against coverage. Read `usage/odds_api.jsonl` in the private repo. A manual `Run workflow` with `use_reserve` may capture it if the reserve allows |
| `CAPTURE ... no in-scope games` | every game had tipped before the snapshot (a holiday matinee slate) | expected; counted as `games_out_of_scope` |
| `CAPTURE ... out of season` | the date is outside `config.REGULAR_SEASON_WINDOWS` (preseason, or the window's end has passed) | expected before 2026-10-20; **set the 2026-27 end date in `nba/config.py` as soon as the league schedule is published**, so the play-in and playoffs are never priced |
| `Capture not armed: missing repository secret(s)` | `ODDS_API_KEY` (or `HF_TOKEN`) is not set | add the secret (Settings → Secrets → Actions); the free key comes from the-odds-api.com |
| `n_unmatched > 0` in the capture record | a sportsbook name did not match a player deterministically | look at `lines/<date>.parquet` rows with `match_method = unmatched`; add a confirmed row to `nba/lines/aliases.csv` (raw name, player_id, date, how you confirmed it); never a fuzzy match |
| `UnknownTeamError` in the capture log | The Odds API used a team name not in `nba/lines/teams.py` | add it (abbreviation as the game logs spell it; San Antonio is SAN) |
| `RECOMMEND <date>: 0 paper bets` early in the season | every player is a season debut (`no prior game this season`) | expected on opening night and the first few days; the rule needs one prior game |
| a hand-read line to add | | `python -m nba.lines.manual template > lines.csv`, fill it from your own account (prices as American odds, time read in UTC with `Z`), then `python -m nba.lines.manual ingest lines.csv` |
| `assert_predictions_precede_their_lines` fails | a prediction was stamped after its line was captured: the nightly and the capture ran out of order | find why (a re-run of the nightly after 20:00 UTC?); do not change the test; the bet is already excluded |

How many bets before the numbers mean anything (ADR-0030): under 100 settled bets the
`verdict` column reads "too early"; above that the interval and the break-even rate implied
by the prices taken are shown, and a verdict is given only when the interval excludes
break-even. Separating a true 55 % hit rate from the 52.4 % break-even of −110 takes about
1,400 settled bets ((1.96 × √(0.55 × 0.45) / 0.026)² ≈ 1,406); at 100 the half-width is
±0.10, at 400 ±0.05. Headline numbers are API lines only; manual lines are reported in their
own rows.

## Red runs

| Symptom | Likely cause | Do |
|---|---|---|
| `nightly.yml` fails in "Require secrets" | a repository secret was deleted or renamed | restore `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` (`GROQ_API_KEY` is optional) |
| `INGEST` raises a schema error | the dump changed a column name or type | the error names the column; fix `nba/ingest/kaggle_dump.py` and add a fixture row; do not relax `nba/schema.py` |
| download failure for `PlayerStatistics.csv` | Kaggle outage or a renamed file | re-run later; if the file was renamed, `kaggle_dump.BOX_SCORE_FILE` |
| dbt `ERROR` in a reconciliation test (`assert_*_reconcile_*`) | a mart no longer reproduces a committed report | nothing publishes until it passes; read the failing rows (`dbt test --select <test> --store-failures` locally), compare with the report's keys; a legitimate change to a published number is a new report plus a new ADR, never a tolerance change |
| dbt `ERROR` in a contract (`slv_game_logs`, any gold model) | a column changed type or name | the model's SQL or YAML changed; fix the one that is wrong; incremental models need a full refresh after a column change |
| dbt `ERROR: on_schema_change='fail'` on a silver model | an incremental model's columns changed | full refresh (below) |
| `publish` fails with "HF_TOKEN is not set" or "writes are disabled" | token missing, or `NBA_REHEARSAL` is set outside a rehearsal | check the secret; make sure the step is not running with the rehearsal environment |
| `relationships … ref('seasons')` reports **ERROR** (not warn) | the `seasons` seed table does not exist in the target | the selection skipped seeds; run `dbt seed` or a full refresh once; the nightly selector includes seeds for this reason (ADR-0023) |

## When a full refresh is required

`warehouse.yml` runs one every Sunday 06:00 UTC and pushes `gold/`; dispatch it (Actions →
Warehouse full refresh → Run workflow) when you cannot wait:

1. After any change to an incremental model's SQL or columns (`slv_game_logs`,
   `slv_predictions`, `slv_residuals` have `on_schema_change='fail'`).
2. After a restatement older than `lookback_days` (the `warehouse-warn` above).
3. At the start of a season, and after the first in-season nightly build, to confirm the
   incremental and full paths agree (`mart_season_coverage`, `mart_holdout_metrics` nightly
   rows).
4. After a change to a seed, a var in `dbt_project.yml`, or the model/dataset revision pins.
5. When the `gold/` export must be refreshed now (the nightly only exports on nights it
   built).

A dispatched run with `publish: true` (the default) pushes `gold/`; record the run with
`python scripts/record_run.py <run id> --target motherduck --pushed cbratkovics/nba-game-logs@<revision>`.

## Opening-night checklist (2026-27, first game 2026-10-20)

- [ ] `LeagueSchedule26_27.csv` present in the dump, or `ODDS_API_KEY` set so the ADR-0029 fallback (`SLATE … schedule from The Odds API events`) produces the slate.
- [ ] `ODDS_API_KEY` set and one `capture.yml` dispatch with `dry_run: true` showing `plan of N credits allowed`; then the 20:00 UTC schedule captures opening night (expect `0 paper bets`: every player is a season debut).
- [ ] The first in-season `nightly.yml` run shows `SLATE … games, … player predictions`,
      `DECISIONS …`, `WAREHOUSE build: slate written`, a dbt `Done. PASS=` line with
      `ERROR=0`, and a `PUBLISH pushed gold marts` line; `gold/_export_manifest.json` carries
      that run's commit. Record the run id.
- [ ] The next night shows `RESIDUALS <date>: N predicted, M with actuals`; `M` close to `N`
      means the dump author published the box scores (if `M = 0` for several days the dump
      has stopped updating: ADR-0029's second risk, nothing settles until it resumes).
- [ ] Drift stays `insufficient` until a window holds 500 training-population rows (about
      the second week); the first `ok` is the first out-of-sample verdict.
- [ ] `/predictions` and `/decisions` on the site show the slate (the pages fall back to the
      replay until `predictions/latest.json` and `decisions/latest.json` exist).

## Rehearsing before a change (ADR-0028)

Actions → Nightly → Run workflow → `rehearsal: true` (optionally a `date`). It runs the full
nightly path on a synthetic game night: nothing is pushed, the warehouse builds into a local
file uploaded as `warehouse-local-<run id>`, and the publish step only lists files. Use it
after any change to the warehouse steps of `nightly.yml`, the loader, the selector or the
publisher.
