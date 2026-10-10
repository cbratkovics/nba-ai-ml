# Analytics-engineering audit (Phase 0, read-only)

Audited `main` at `4f16543` on 2026-10-09 on the branch `analytics-engineering`. Every claim
below cites a file and line, a GitHub Actions run id, a Hugging Face commit, or a command run
in this audit. Where a point could not be verified it says so.

## 0. What this audit ran

| Command | Result |
|---|---|
| `git pull --ff-only` | the working copy was three commits behind `origin/main` (`bbfff6f` → `4f16543`, PR #6 merged 2026-09-23); fast-forwarded before reading |
| `make dbt-load` | dataset revision `949cbd39` (HF commit "Nightly 2026-10-09"): 5 game-log files, 0 predictions, 0 residuals, 28 daily reports, 24 drift reports, 164 replay residual files |
| `make dbt-full` | `Done. PASS=102 WARN=0 ERROR=0 SKIP=0 TOTAL=102` — 31 models (12 bronze, 5 silver, 14 gold; 3 incremental, 27 table, 1 view), 3 seeds, 1 snapshot, 67 data tests (12 singular, 55 generic), 13.8 s |
| `make dbt-docs` | catalog built; `check_dbt_descriptions.py`: "every model, seed, snapshot, column (617) and source has a description" |
| `pytest` (minus `test_model.py`, `test_train.py`) | 175 passed, 0 failed, 3 m 32 s. The two LightGBM files fail at collection on this Intel Mac because `libomp` is not installed (`/usr/local/opt/libomp` absent), exactly as README line 129 says |
| `make lint` | `All checks passed!`, 87 files already formatted |
| `gh run list` per workflow | §4 |
| HF listing of `cbratkovics/nba-game-logs` and `cbratkovics/nba-stat-predictor` | §3, §4 |
| `curl` of the five Vercel pages and the Pages site | §4 |

Local warehouse facts from the build (DuckDB `.duckdb/nba.duckdb`): `fct_player_game` 130,414
rows; `slv_game_logs` 2021-10-19 to 2026-04-12; `mart_season_coverage` five seasons, all
`meets_contract` (2024-25 at the declared 1,223); `dim_player_asof` 2,003 stints for 1,027
players; `mart_drift` 24 rows (one per off-season run); `fct_prediction` 38,372 rows, all
`run_kind = replay`; `mart_holdout_metrics` min10 pts 4.7665 vs 4.9075 (n 22,075), all pts
4.8639 vs 4.6793 (n 26,031).

## 1. Verdict on the eight gap claims

### 1. "The warehouse is not on the product's path" — confirmed, with one worse detail

- Features are built in pandas: `nba/features/asof.py:66-98` (`build_features`) and
  `:113-141` (`features_for_pending`), called by the slate (`nba/predict/slate.py:141`).
  Nothing in `dbt/` computes a model feature; `fct_player_game.sql:27-38` recomputes only the
  season-to-date mean, as the decision policy's second baseline.
- Bronze reads what the pipeline already wrote: `dbt/models/bronze/_sources.yml:9-86` (game
  logs, predictions, residuals, replay residuals, daily reports, drift reports, the loader
  manifest, and the four committed reports).
- The site never reads `gold/`: every fetch goes through `frontend/lib/data.ts:61-73`
  (`DATASET_PATHS`) and `:75` (the model repo's `metrics.json`); `grep gold frontend/` finds
  only a definition string inside `policy_summary.json`. If `dbt/` were deleted, the site
  would render unchanged.
- Only the agent reads gold, with fallbacks: `nba/agent/tools.py:93-96` (`_mart` reads
  `<root>/gold/<mart>.parquet`), used at `:128` (`mart_drift`), `:230` (`mart_restatement_lag`),
  `:393` (`fct_decision_policy`), `:469` (`mart_daily_metrics`).
- **Worse detail:** the `gold/` the agent pulls from Hugging Face is frozen at
  2026-09-16. HF commit `66404be0` ("Export gold marts", from the dispatched run
  `35048092357` on `f27e660`) is the last commit touching any `gold/*` file;
  `gold/_export_manifest.json` says `mart_drift` has 0 rows, while the 2026-10-04 run built 18
  and the local build has 24. Cause in §2 item 1. So the one consumer the warehouse has is
  reading a three-week-old export and silently falling through to the file fallbacks.

### 2. "The marts answer model-monitoring questions only; no metric has a written definition" — confirmed

- The fourteen gold models (`dbt/models/gold/_gold.yml`) are: four dimensions
  (`dim_game`, `dim_team`, `dim_player_asof`, `dim_player_current`), three facts at the
  prediction / box-score / decision grain, and seven marts whose questions are all "how is the
  model or the pipeline doing" (`mart_holdout_metrics`, `mart_daily_metrics`,
  `mart_season_coverage`, `mart_restatement_lag`, `mart_policy_metrics`, `mart_policy_sweep`,
  `mart_drift`). Nothing answers a team, player or game question.
- Definitions exist only as column descriptions (`_gold.yml`) and, for the decision policy,
  as `nba/decisions/policy.py:272-293` (`DEFINITIONS`, carried in the artifact's `definitions`
  key). None states grain, additivity or an anti-use. There is no `docs/METRICS.md`.

### 3. "dbt features a reviewer looks for are absent" — confirmed on every item, counts from the built manifest

From `dbt/target/manifest.json` after `make dbt-full`:

| Item | Finding |
|---|---|
| unit tests | `unit_tests: 0` |
| exposures | `exposures: 0` |
| source freshness | 0 of 11 sources have `freshness` or `loaded_at_field` (`_sources.yml`); `brz_load_manifest.loaded_at` exists (`brz_load_manifest.sql:4`) and is unused for this |
| intermediate layer | none; silver → gold directly (`dbt_project.yml:63-76`) |
| date / season dimension | none; season is derived by the `season_of` macro (`dbt/macros/season_of.sql`) in two silver models and copied through |
| selectors | `selectors: []`; the ten-item `--select` list is duplicated verbatim in `Makefile:44` and `.github/workflows/nightly.yml:145` (diffed: identical today) |
| SQL linting | `pyproject.toml:44` excludes `dbt` from ruff; no sqlfluff anywhere; CI lints Python only (`ci.yml:27-31`) |
| groups / access | no `groups`; every model is the default `access: protected` |
| `store_failures` | `false` on all 67 tests (`dbt_project.yml:82-84`) |
| severity | 66 tests `ERROR`, one `warn`: `assert_plausible_box_scores` (`dbt/tests/silver/assert_plausible_box_scores.sql:4`) |
| packages | none (ADR-0014); two home-made generic tests (`dbt/tests/generic/`) |

### 4. "Dimensional modelling is thin" — partly right

- Denormalised strings on facts: confirmed. `player_name`, `team`, `opponent` on
  `fct_player_game.sql:47-49`, `fct_prediction.sql:23-25`, `fct_decision_policy_v1.sql:132-134`.
- `dim_team` keyed by `(team, season)` with no surrogate key: confirmed (`_gold.yml:59-74`,
  `unique_combination` on the pair, `not_null` constraints, no key column).
- "No relationship tests I could find": **wrong by one.** There is exactly one:
  `fct_player_game.game_id → dim_game.game_id` (`_gold.yml:85`;
  `relationships_fct_player_game_game_id__game_id__ref_dim_game_` in the manifest). There are
  none from `fct_prediction` or `fct_decision_policy` to `dim_game`, none to `dim_team`, none
  from any fact to a player dimension.
- Not mentioned in the brief but relevant: `dim_game` and the two facts do declare
  `primary_key` model-level constraints (`_gold.yml:43-45, 80-82, 122-124`), which DuckDB
  enforces at build time; so the grain is enforced even where the relationship is not tested.

### 5. "Docs contradict each other and undersell what ran" — confirmed; the gap is larger and in a different place than stated

- Counts: `README.md:103` says "24 models, 3 seeds, 1 snapshot, 53 tests". The build says 31
  models, 3 seeds, 1 snapshot, 67 tests. (The Pages site generated 2026-09-23 also says 31.)
- "Prototyped / no run record" wording that is now false:
  - `README.md:56-58` ("the committed evidence does not show a completed MotherDuck build or a
    pushed `gold/` export"), `:67-68` (two table rows), `:103` ("prototyped: built locally,
    not yet run in Actions"), `:140` (Deployment boundary bullet), `:145-149` (Owner TODO 1, 2
    and 4).
  - `.github/workflows/warehouse.yml:9-10` ("No committed run record currently proves that
    this workflow has completed against MotherDuck").
  - `docs/DECISIONS.md`: ADR-0003, 0004, 0006, 0007, 0009–0020 are titled "prototyped";
    ADR-0003:46-48 "that path has not run yet"; ADR-0018:335 "the dispatched nightly run is
    an owner step".
  - `docs/STORIES.md:31` ("MotherDuck, pushed gold exports, and nightly decision/drift
    publication remain unverified by committed run evidence").
  - `docs/agent.md:49-51` ("until the first warehouse build has run in Actions").
- What actually ran (§4): `warehouse.yml` has four green runs on `main`, all on target
  `motherduck` (`35048092357` dispatch 2026-09-16; `35506971007`, `36317104881`,
  `37201223219` scheduled). The 2026-10-04 log says `Concurrency: 4 threads
  (target='motherduck')` and `Done. PASS=102`. `nightly.yml` has run green every day since
  2026-09-12 (28 runs), pushing `daily_reports/` and `drift/` to HF daily (24 drift files,
  `drift/2026-09-16.json` … `drift/2026-10-09.json`). The dbt docs are live on Pages.
- **But the brief's framing "the gap is that no run evidence is committed, not that nothing
  ran" is itself incomplete.** Two of the undersold claims are still true in substance:
  `gold/` on HF is stale (item 1 above) because the scheduled full refresh never pushes it
  (§2 item 1), and the nightly's warehouse steps have never executed (§2 item 2). So Phase 1
  must not swing the README to "deployed" wholesale; it has to split "built on MotherDuck
  weekly (run ids)" from "gold export pushed once on 2026-09-16, not since" from "nightly
  incremental path not yet exercised".

### 6. "First impression leads with ML" — confirmed

`README.md:1` title "NBA Stat Predictor", `:3` tagline about batch predictions, `:15-19` three
bullets (Predict, Validate, Audit) with no warehouse, `:21-52` architecture diagram with
`Warehouse` as the fifth subgraph (`:42-45`). The GitHub About text (live, read with
`gh repo view`) is "Zero-cost nightly NBA player-stat prediction pipeline: GitHub Actions,
Hugging Face, LightGBM, and a tool-calling agent that audits each run."; topics are
`sports-analytics, lightgbm, github-actions, huggingface, llm-agents, mlops, nextjs, dbt,
duckdb` (dbt and duckdb last).

### 7. "Hygiene" — confirmed on the repo items; the About/topics page you saw is stale

- `dbt/.user.yml` is tracked (`git ls-files dbt/.user.yml`); it holds a dbt anonymous-usage id.
- `ci.yml:7-10` grants `contents: read, pages: write, id-token: write` at workflow level, so
  `lint-and-test` and `dbt` carry the Pages/OIDC permissions on every push and pull request;
  only `pages` (`ci.yml:111-121`) needs them.
- `ci.yml:78`: the slim path runs `--select state:modified+ --defer --state … --full-refresh`,
  so CI never exercises an incremental run through dbt. The incremental path is proven only by
  `tests/test_dbt_incremental.py` (pytest shelling out to dbt on fixture rows).
- About text and topics: **live values are not the cached ones.** No "R² 0.94", no `mlflow`,
  `fastapi` or `xgboost` (the old audit, `AUDIT.md` §15 item 5, already recorded their
  removal). What is live is accurate but ML-led; proposed replacement in §6 Phase 1.
- Also found: `frontend/lib/policy_summary 2.json` exists locally, untracked and git-ignored by
  the `*.json` rule (a stray duplicate on this machine only); `frontend/.env.local` holds unused
  `NEXT_PUBLIC_CLERK_*` / `NEXT_PUBLIC_ACCURACY_RATE` variables from the pre-rebuild site
  (git-ignored, read by nothing). Both are local hygiene, not repo hygiene.

### 8. "The 2026-27 season is about to start" — partly right; the named risk is a dead variable, the real risks are elsewhere

- `var('seasons')` (`dbt_project.yml:30`) stops at 2025-26, **but no model, test or macro
  reads it** (`grep -rn "var('seasons')" dbt/` returns nothing). Its only use is
  `tests/test_project_config.py:26`, which mirrors it to `config.SEASONS`. The gate that
  matters is the `accepted_values` list on `slv_game_logs.season` (`_silver.yml:14`), which
  already includes 2026-27. `config.SEASONS` itself is a default argument of
  `kaggle_dump.prepare` (`nba/ingest/kaggle_dump.py:433, 507`) that the daily ingest overrides
  with `seasons=None` (`nba/ingest/kaggle_daily.py:240`), so 2026-27 rows will be ingested and
  labelled `2026-27` by `season_from_date` (`kaggle_dump.py:250-253`; the SQL macro
  `season_of` applies the same October rule).
- The nightly already treats the current season as 2026-27: `drift/2026-10-09.json` on HF has
  `"season": "2026-27"`, `"position": "off_season"`, `"no_schedule_streak": 9`.
- Real opening-night risks, in order:
  1. **The nightly's warehouse steps have never run.** Every nightly since 2026-09-16 hit
     `WAREHOUSE skip` (`nba/warehouse/gate.py:42`), so `nightly.yml:122-159` (choose target,
     loader, incremental `dbt build` on MotherDuck, `export_gold`, `publish`) have executed zero
     times in that workflow. The first slate night is the first execution.
  2. **No-schedule WARN fires about 2026-10-14.** The streak was 9 on 2026-10-09 and
     `NO_SCHEDULE_STREAK_WARN = 14` (`nba/config.py:172`). Unless `LeagueSchedule26_27.csv`
     appears in the Kaggle dump first, the run around 2026-10-14 opens an issue labelled
     `nightly-warn` (`nightly.yml:89-102`; the label does not exist yet and is created on
     demand). This is the designed behaviour (ADR-0018), but it will happen before opening
     night and should be in the runbook.
  3. **Bronze optional families switch shape on the first in-season night.**
     `brz_predictions.sql:3-9`, `brz_residuals.sql:3-9` and the daily/drift copies go from
     `empty_typed_relation` to a real `select` when the first file appears
     (`files_exist`, `dbt/macros/files_exist.sql:5-11`). They are tables, so this is a
     rebuild, not a schema change; the downstream incremental `slv_predictions` /
     `slv_residuals` have `on_schema_change='fail'` but the column list is fixed by the
     macros (`product_columns.sql`), so it should pass. It has never been exercised against
     real nightly files; `test_dbt_incremental.py` covers only `slv_game_logs`.
  4. **Season-position labels for 2026-27 collapse to opening / april / regular** because
     `SEASON_CALENDAR` (`nba/config.py:163-168`) has only the 2025-26 Cup, deadline-week and
     All-Star windows. The drift reference does not depend on the labels (ADR-0017), so this
     affects reporting only.
  5. The drift window needs 500 training-population rows (`DRIFT_MIN_ROWS`), so the first
     real verdict lands about the second week; until then every report is `insufficient`.
     Expected and documented (ADR-0016).

## 2. What the brief missed, ranked by how much a reviewing analytics engineer would care

1. **Scheduled full refreshes never push `gold/` (expression bug).** `warehouse.yml:73`
   gates the publish step on `${{ inputs.publish != false }}`. On a `schedule` event the
   `inputs` context is empty; GitHub coerces both sides to numbers (`'' → 0`, `false → 0`),
   so the expression is false and the step is skipped. Evidence: in run `37201223219` step 7
   "Push the exported gold marts to the dataset repo" is `skipped` while steps 5–6 succeeded;
   HF `gold/` is unchanged since `66404be0` (2026-09-16, the dispatched run whose `publish`
   input defaulted to true). The nightly's publish is gated on the build gate and has never
   fired. Net: the warehouse rebuilds weekly on MotherDuck and throws the export away.
2. **The nightly warehouse path is implemented, not verified** (§1 item 8.1). The README
   claims it as the nightly's step 6 (`README.md:50`, `nightly.yml:11-17`).
3. **The served `/decisions` page does not carry `4f16543`.** PR #6 changed one sentence
   (`frontend/app/decisions/page.tsx:127-128`, "in-sample policy-fit results, not
   out-of-sample validation…"); `curl https://nba-ai-ml.vercel.app/decisions` returns
   `x-vercel-cache: HIT`, `date: Thu, 24 Sep 2026`, and the sentence is absent. Either Vercel
   has not deployed `4f16543` (merged 2026-09-23 23:45 UTC) or it is serving a stale ISR copy
   (`revalidate = 3600`, `decisions/page.tsx:17`, should have refreshed). The `/replay` page
   does carry `70d19aa` (164 `<option>`s, "Did not play"). README Owner TODO 3 is therefore
   half-done; the owner must read the deployment commit in the Vercel dashboard.
4. **The HF model card is behind `docs/MODEL_CARD.md`.** HF revision `7dc0eafc` (2026-09-16)
   lacks PR #6's eligible-cohort wording (`MODEL_CARD.md:39-41, 57-62, 99-101`); `diff`
   shows 21 changed lines. README Owner TODO 5 is still open. Pushing is an owner step.
5. **Silver has no contracts.** Contracts are enforced on gold only (`dbt_project.yml:75-76`);
   silver relies on the grain `unique_combination` and `not_null` tests. Defensible (ADR-0004
   explains silver as grain-enforced), but a reviewer will ask why the incremental models,
   whose `on_schema_change='fail'` is the only schema guard, are not contracted. Decide and
   record, rather than add contracts reflexively.
6. **Three docstrings describe readers that do not exist.** `nba/warehouse/publish.py:3-5`
   ("so the site and the analyst tools can read a mart") — the site does not;
   `dbt/models/gold/_gold.yml:4` ("The nightly job exports these tables … and pushes them") —
   it has not yet; `nba/storage/hf.py:190` (`push_products` "uploads only predictions and
   residuals") — it accepts any product prefix.
7. **Two implementations of the season rule** (`kaggle_dump.season_from_date` and the
   `season_of` macro) and a dead `seasons` var. A season/date dimension (Phase 3) is the place
   to make one of them the source and test the other against it.
8. **The dedup rule in `slv_game_logs` is untested in isolation.** `slv_game_logs.sql:44-53`
   (higher minutes wins, ties by `source_file asc`) has no fixture that produces a duplicate;
   today the publisher guarantees uniqueness so the branch never executes. A unit test
   (Phase 4) is the right proof.
9. **The agent misreads `pushed: false`.** `brief/latest.json` (2026-10-08) summarises
   "Ingest shows no changes and push failed; data is 180 days stale". `pushed: false` in the
   daily report means nothing needed pushing (`kaggle_daily.py:277, 304`). This is agent
   prompt/tool wording, outside this brief's scope ("changes to the agent beyond pointing its
   tools at renamed marts"); noted so it is not mistaken for a pipeline fault.
10. **Actions deprecation warnings.** Every run logs "Node.js 20 is deprecated" for
    `actions/checkout@v4`, `setup-python@v5`, `upload-artifact@v4`. Cosmetic until GitHub
    enforces it; bump when convenient.

## 3. Lineage walk

### Site pages (`frontend/lib/data.ts:50-51`: dataset files at `resolve/main`, `metrics.json` at the pinned model revision; `fetch` at `:444` with `revalidate` 3600; a 404 returns null, any other error throws with no page-level handler)

| Page | Reads | Fallback when absent | Mart that already holds the same numbers |
|---|---|---|---|
| `/` (`app/page.tsx:24-29`) | model repo `metrics.json`; `replay/2025-26/replay.json`; committed `frontend/lib/all_rows_baseline.json` | "not published at {url}" messages (`page.tsx:116, 241`) | `mart_holdout_metrics` (both populations; replay rows reconcile to both reports) |
| `/predictions` (`predictions/page.tsx:17-23`) | `predictions/latest.json`, `predictions/rolling_metrics.json` (JSON Lines), `replay/2025-26/daily_mae.json` | "No slate yet" + last replayed date (`:44-57`) | `mart_daily_metrics` (nightly rows, in season) |
| `/replay` (`replay/page.tsx:23-36`) | `replay/2025-26/daily_mae.json`, `replay/2025-26/slates/index.json`, `slates/<date>.json`; `all_rows_baseline.json` | "replay files are not published" (`:88`) | `mart_daily_metrics` population `all` (reconciled day by day) |
| `/decisions` (`decisions/page.tsx:112-114`) | `decisions/latest.json`, `replay/2025-26/daily_mae.json`; committed `frontend/lib/policy_summary.json` | "No slate today" + replay evaluation (`:154-162`) | `mart_policy_metrics`, `mart_policy_sweep` |
| `/brief` (`brief/page.tsx:61-70`) | `brief/index.json`, `brief/<date>.json` or `brief/latest.json`, `daily_mae.json` | "No brief is published" (`:169-170`) | none (agent output) |

Neither `predictions/`, `residuals/` nor `decisions/` exists in the dataset repo today (the
2025-26 run pre-dates these prefixes and the off-season writes none), so `/predictions` and
`/decisions` are on their fallbacks.

### Agent tools (`nba/agent/tools.py`; registry `:659-667`; `TOOL_RETURN_VERSIONS` `:45-53`)

| Tool | Mart (from `<root>/gold/`) | Fallback order |
|---|---|---|
| `get_daily_report` v2 | `mart_drift` (`:128`), `mart_restatement_lag` (`:230`) | report: `daily_reports/<date>.json` → `data/daily_report.json` (`:269-272`); drift: mart → `drift/<date>.json` (`:158-190`) → calibration row (`:191-224`); restatement: mart only (`:231-232`, no fallback) |
| `get_upstream_freshness` v2 | none | stored game logs + the dump under regular-season rules (`:300-323`) |
| `get_residuals` v1 | none | `residuals/<date>.parquet` → `replay/<season>/residuals/<date>.parquet` (`:103-108`) |
| `get_rolling_metrics` v2 | `mart_daily_metrics` (`:469`, population `all`), `fct_decision_policy` (`:393`, `min10`) | residual files → `replay/<season>/daily_mae.json` (`:519-568`); decisions: residual files + `reports/policy_<season>.json` (`:428-463`) |
| `get_player_recent`, `get_team_context`, `list_data_gaps` v1 | none | stored game logs (+ `KNOWN_MISSING_GAMES`) |

In Actions the tools see the HF `gold/` of 2026-09-16 (`pull_products`, `nba/storage/hf.py:168-181`
includes `gold/`); `mart_drift` there is empty, so the drift block always falls through to the
`drift/<date>.json` products. Reconciled on two golden dates per ADR-0019, so the numbers are
right; the lineage is just not what `docs/agent.md:47-54` describes.

### Product prefixes in `cbratkovics/nba-game-logs`

| Prefix | Writer | Reader |
|---|---|---|
| `game_logs/` (5 files) | `hf.push_dataset` from the backfill / daily ingest | loader (`nba/warehouse/load.py:83`), slate, agent |
| `predictions/`, `residuals/`, `decisions/` (absent today) | nightly (`nba/nightly.py:179, 124-126, 189-191`) | site, agent, bronze |
| `replay/2025-26/` (332 files) | `replay.yml` run `35539759857` | site, agent, bronze (`replay_residuals`) |
| `daily_reports/` (28), `drift/` (24) | nightly, daily | agent, bronze |
| `brief/` (70) | nightly agent step | site `/brief` |
| `gold/` (13 parquet + manifest, 2026-09-16) | `nba/warehouse/publish.py` via `warehouse.yml:74` (bug, §2.1) and `nightly.yml:148-150` (never fired) | agent only |

## 4. Status claims against run evidence

Run ids are GitHub Actions run ids on `cbratkovics/nba-ai-ml`; all listed runs concluded
`success` unless noted.

| Claim | Where | Evidence found | Verdict |
|---|---|---|---|
| Ingest ran in Actions (`34668092730`) | README:62 | run exists (dispatch, 2026-09-12, `cf4ebac`); nightly has run on schedule every day since: `34695782154` (09-12) … `37960899102` (10-09), 28 runs, all success | true, understated |
| Train and evaluate ran, model at HF `fb427de` | README:63 | `train.yml` runs `34665925555`, `34666354516` (2026-09-12, `50a3b2e`); HF model commit `fb427de1` 2026-09-12 | true |
| Replay ran (`34668262142`), 164 dates | README:64 | run exists; a later run `35539759857` (2026-09-20, branch `replay-slates`) pushed the slates the site reads | true |
| Agent evaluation `35073930571` … `35499665725`, 25 of 25 | README:65, 111 | runs exist (09-16 … 09-20); daily scheduled runs continue to `37917032257` (10-09), exiting early as designed | true |
| Core site deployed on Vercel; newer page revisions not verified | README:66, TODO 3 | all five pages HTTP 200; `/replay` carries `70d19aa`; `/decisions` does not carry `4f16543` (§2.3) | partly true; one commit behind or cached |
| Decision policy and drift: "no committed run record proves nightly decision/drift products were pushed to HF" | README:67, 99, 140; STORIES:31 | `drift/` has 24 files pushed by the nightly (first `17fd9fb1` 2026-09-16); `decisions/` cannot exist until a slate is written | drift: false (ran and pushed); decisions: true but trivially (off-season) |
| Warehouse: "no committed record proves a completed MotherDuck run or pushed `gold/` exports" | README:57-58, 68, 103, 140, 145; warehouse.yml:9-10; ADR-0003:46-48; ADR-0007 title | four `warehouse.yml` runs on `motherduck`: `35048092357`, `35506971007`, `36317104881`, `37201223219` (log: `target='motherduck'`, `PASS=102`); `gold/` pushed once, HF `66404be0` 2026-09-16, not since (§2.1) | MotherDuck: false (it ran four times); gold push: half-true (once, stale) |
| "`dbt build`: 24 models, 3 seeds, 1 snapshot, 53 tests" | README:103 | 31 models, 3 seeds, 1 snapshot, 67 tests (local build and run `37201223219`: TOTAL=102) | false |
| dbt docs to GitHub Pages (TODO 4 "enable if not already") | README:148; ci.yml:111-121 | Pages enabled (`build_type: workflow`), `https://cbratkovics.github.io/nba-ai-ml/` HTTP 200, manifest `generated_at 2026-09-23T23:46:25Z` from CI run `35935163627` | true; TODO 4 is done |
| Nightly step 6 builds the warehouse on MotherDuck | README:50; nightly.yml:11-17; overview.md:8-9 | steps 11–14 skipped on every nightly run (gate); never executed | implemented, not verified |
| "Pass rates … row rendered from the report" | README:111 | `agent-eval.yml:133-144` commits the row; `tests/test_agent_pass_rates.py` checks it | true |
| Model card pushed (TODO 5) | README:149 | HF card `7dc0eafc` 2026-09-16 differs from `docs/MODEL_CARD.md` after PR #6 | open |
| ADR status words "implemented / prototyped / planned" (DECISIONS.md:5-7) | ADR titles | no ADR was re-statused after 2026-09-16; 0003, 0004, 0007, 0009–0020 say prototyped | stale |
| About/topics "R² 0.94", `mlflow`/`fastapi`/`xgboost` | brief | live About and topics carry none of these | the page you saw was stale |

## 5. Observations a reviewer would make that are not gaps

- The reconciliation design is sound and the tolerances are honest about their origin
  (ADR-0011 says the 0.005 was set after observing 0.0021). Keep it.
- Rates are already ratio-of-sums where they exist: `mart_holdout_metrics` /
  `mart_daily_metrics` average per-row absolute errors; policy hit rates are `n_hit /
  n_resolved` (`mart_policy_metrics.sql:38-39`). There is no average-of-averages today.
- `dim_player_asof` is good dimensional work: a player who returns to a former team gets a
  new stint (`lag(team)` by player, `dim_player_asof.sql:19-36`), and the roster rule is proven
  against it (`assert_replay_roster_rule_reproduces_from_asof`).
- The incremental proof (`tests/test_dbt_incremental.py`) tests three things most projects
  skip: equality after append, a restatement inside the lookback, and one outside it.
- Every column is described (617) and CI fails otherwise. Rare.

## 6. Phase recommendations

Sizes: S = under half a day, M = one to two days, L = three or more, at my pace with your
review between phases.

### Phase 1 — truth and hygiene (do it; S–M)

Scope, in commit order:
1. `reports/runs.json` (or `docs/RUNS.md` rendered from it): one row per run id above with
   workflow, commit, target, conclusion, date, and what it pushed; a test that every run id
   cited in README/ADRs appears in it.
2. README, ADR statuses, `warehouse.yml:9-10`, `STORIES.md:31`, `agent.md:49-51`,
   `_gold.yml:4`, `publish.py:3-5`, `hf.py:190` rewritten to the three-way truth: MotherDuck
   weekly build **ran** (ids); gold export **pushed once 2026-09-16, not since** (bug);
   nightly incremental path **implemented, not verified**. Owner TODO rewritten (4 done, 3
   and 5 reworded).
3. Fix `warehouse.yml:73` to `if: ${{ github.event_name == 'schedule' || inputs.publish }}`.
   **Ask first:** once merged, the next Sunday run pushes fresh `gold/` to HF, which is a
   write to HF you said to clear with you.
4. README counts rendered from `dbt/target/manifest.json` between markers (same mechanism as
   the `pass-rates` row, which `agent-eval.yml` rewrites and `test_agent_pass_rates.py`
   checks); a test fails when the rendered line drifts from the manifest. Requires the
   manifest in CI, which the `dbt` job already produces.
5. `git rm --cached dbt/.user.yml`; add to `.gitignore`.
6. `ci.yml`: `permissions: contents: read` at top; `pages: write, id-token: write` on the
   `pages` job only.
7. `dbt/selectors.yml` with a `nightly` selector; `Makefile:44` and `nightly.yml:145` use
   `--selector nightly`; a test that the selector's resolved set covers every incremental
   model and every gold model (`dbt ls --selector nightly`).
8. sqlfluff for `dbt/`: `sqlfluff==4.4.0` and `sqlfluff-templater-dbt==4.4.0` resolve against
   the pinned `dbt-core==1.12.5` / `dbt-duckdb==1.11.0` (checked with `uv pip install
   --dry-run -c constraints.txt`; nothing installed). Config: dialect duckdb, templater dbt,
   lower-case keywords, 100-char lines, trailing commas as the models already do; rules tuned
   so the first run passes with at most a handful of fixes, not a rewrite. Pins go in
   `constraints.txt` and the `dev` extra. This is a Python dev tool, not a dbt package, so
   ADR-0014 stands; I would say so in a one-line ADR note.
9. About text to set (the live one is accurate but ML-first):
   "dbt + DuckDB/MotherDuck warehouse behind a nightly NBA player-stat predictor: contracted
   gold marts that reconcile to every published number, LightGBM scoring, and a tool-grounded
   agent, all on free tiers." Topics: `dbt, duckdb, motherduck, analytics-engineering,
   data-modeling, sports-analytics, github-actions, huggingface, lightgbm, nextjs` (drops
   `llm-agents`, `mlops`; adds the first four).

Cut: nothing. Main design choice: where run evidence lives. Alternatives: (a) a JSON under
`reports/` tested like the other artifacts (recommended; it is what the repo already does
for numbers); (b) a `## Runs` table typed into README (no); (c) a workflow that appends to the
file on every run (contents: write on the nightly — more permission, more noise; defer).

### Phase 2 — put dbt on the path (do it; L; the hardest phase)

**2a. Features in SQL with a parity proof.** Scope: `int_player_game_features` (new
`intermediate` folder, schema `intermediate`, not exported, no contract because it is
internal — stated) computing all 23 `asof_v1` features over `slv_game_logs`; a parity test
over all 130,414 rows.

Main design choice: *what the SQL is compared against*. Alternatives:
- (i) pytest only on fixture rows, like `test_dbt_gold.py` — cheap, proves the rules on ~400
  rows, weak on real ties and gaps.
- (ii) pytest on the real local warehouse, gated on `data/warehouse` existing, run in the CI
  `dbt` job after the build — proves 130k rows but the proof is invisible in dbt docs.
- (iii) **recommended:** the loader writes `data/warehouse/features/asof_v1.parquet` from
  `nba.features.asof.build_features` over the loaded game logs (the Python module stays the
  one feature implementation and the reference, so the one-feature-module rule holds); a
  source + `brz_features_asof_v1`; a singular test `assert_int_features_match_feature_module`
  compares all 23 columns row for row and fails on any difference above tolerance; plus (i)
  for CI without the HF load. The proof then appears in the lineage graph and in the test
  count, and it runs on MotherDuck too.

Tolerance: `1e-9` absolute on the twelve rolling means, four season means and three vs-opp
means (`test_dbt_gold.py:212` already uses `atol=1e-9` for the season mean and ADR-0015
observed 0.0; the difference is summation order inside a window of at most 20 values, far
below 1e-9); exact equality on `games_played_season`, `days_rest`, `back_to_back`, `home`;
identical null pattern.

The details that make it hard, and how each maps:
- "strictly before this game": `rows between N preceding and 1 preceding` ordered by
  `(game_date, game_id)` within `player_id`, matching `asof.py:75` sort and `:56-63`
  `shift(1)` then `rolling(w, min_periods=1)`. The rolling windows span seasons
  (`test_asof.py:82-83` asserts it); the season mean and `games_played_season` partition by
  `(player_id, season)`; the vs-opponent mean partitions by `(player_id, opponent)` across
  seasons with an unbounded preceding window.
- ties: NBA players do not play twice on one date, but the ordering rule must still be
  `(game_date, game_id)` so that the SQL and pandas agree if a tie ever appears; a unit test
  (Phase 4) can construct one.
- `days_rest`: pandas `diff().dt.days` is a float with NaN on the first game; SQL
  `date_diff('day', lag(game_date), game_date)` is an integer with null. Cast to double in
  the int model to match the contract the model file would eventually use, or compare as
  integers after dropping nulls — decide in the ADR.
- `back_to_back`: `(days_rest == 1).astype(int)` turns NaN into 0; SQL must `coalesce(..., false)`.
- first-game nulls: `min_periods=1` means the first row is null and the second row is the
  first value; SQL `avg` over an empty preceding frame is null, matching.
- `home`: int in pandas, boolean in silver; cast.

ADR: what switching the source of truth would take (slate reads `int_player_game_features`
for history and still needs `features_for_pending` semantics for unplayed games — a SQL
"append pending rows" step — plus a re-pinned `FEATURE_VERSION` and the model's
`feature_names` check; not this season, because the model is pinned).

**2b. The site reads gold.** Pre-requisite: Phase 1 item 3 (fresh `gold/`). Scope: `export_gold`
also writes `gold/<mart>.json` for the small marts (`mart_holdout_metrics` 6 rows,
`mart_policy_metrics` 6, `mart_policy_sweep` 230, `mart_daily_metrics` 978) — the parquet
files are the wrong shape for a browser fetch and `fct_decision_policy` (181k rows) should
never be fetched by a page. `/` reads `mart_holdout_metrics.json` (both populations; drops
the committed `all_rows_baseline.json` to a fallback), `/decisions` reads
`mart_policy_metrics.json` + `mart_policy_sweep.json` (fallback `policy_summary.json`),
`/replay` reads `mart_daily_metrics.json` (fallback `daily_mae.json`). The page names its
source. Exposures in `dbt/models/exposures.yml`: five pages (`type: dashboard`, url, owner),
the agent (`type: application`, depends on the four marts), the HF `gold/` export
(`type: application`), and `nba.decisions.evaluate` (`type: analysis`, depends on
`fct_prediction`, `fct_player_game`). Then the docs DAG ends at things a person looks at.

Cut from 2b: do not point `/predictions` or `/brief` at gold (one has no in-season rows yet,
the other is agent output). Main design choice: JSON export vs the site reading parquet with
a WASM DuckDB — the latter is a new runtime dependency on the serving path; no.

### Phase 3 — dimensional model and analyst-facing marts (do it; M–L)

Star: `dim_date` (one row per calendar date from 2021-10-01 to 2027-06-30: `season`,
`season_day` from the season's first game date, `is_game_date`, `position` from
`SEASON_CALENDAR` for seasons that have one, `is_off_season`) — the drift reference
(`nba/drift/check.py:48-71, 79-86`) needs exactly this and currently derives it in Python,
so the dim can be tested against that derivation; `dim_season` folded into `dim_date` or a
five-row table (your call). Keys: keep natural keys (`game_id`; `(team, season)`;
`player_id`) and add a `team_season_key` string on `dim_team` and the facts so a
`relationships` test can run on one column; a surrogate hash adds nothing in DuckDB at this
size and hides the natural key in the docs. Relationship tests: `fct_prediction.game_id` and
`fct_decision_policy.game_id → dim_game`; `fct_player_game.team_season_key → dim_team`;
`fct_player_game.game_date → dim_date`; `fct_player_game.player_id → dim_player_asof`
(distinct players). Names on facts: **keep, and record why** — `player_name` and `team` on a
fact are the point-in-time values at that game (a player's name string and team *as of that
game* are what the slate and the roster rule use); replacing them with a lookup to a current
dimension would be less correct, and the as-of dimension already exists for history. That is
a "degenerate dimension / SCD on the fact" argument and it fits an ADR.

Marts to pick from (each with grain, formula, additivity and anti-use in `docs/METRICS.md`):

1. **`mart_error_by_segment`** — grain (season, run_kind, model_revision, population,
   target, segment_kind, segment); segments: rest-days bucket (0, 1, 2, 3+), home/away,
   minutes band (10–20, 20–30, 30+), season position from `dim_date`, player tenure
   (`prior_games` quartile). Model and baseline MAE as sum(|err|)/n, plus n. Shows where the
   model is weak; the strongest reviewer story. Not additive across segments of different
   kinds.
2. **`fct_team_game`** and **`mart_team_season`** — team-game rollup from `fct_player_game`
   (pts, fgm/fga, fg3m/fg3a, ftm/fta, reb, ast, tov, players used, result from `dim_game`), then
   season rates as ratio of sums (FG%, 3P%, FT%, AST/TOV). The test for average-of-averages:
   a singular test recomputes one season's FG% as `sum(fgm)/sum(fga)` from the fact and fails
   if the mart's value differs, and a second check asserts the mart's value differs from
   `avg(game_fg_pct)` on a season where the two are known to diverge (so the test proves it
   catches the wrong formula, not just that the right one was used).
3. **`mart_player_form`** — grain (player_id, game_date) for game dates only: last-10 mean vs
   season mean per target, games in the team's last 10 (availability, the roster rule's own
   number), days since last game. Reuses the Phase 2 intermediate layer so the numbers are the
   model's own features, not a third implementation.
4. **`mart_policy_by_segment`** — hit rate by the same segments as (1); only if (1) lands
   cleanly. Lowest marginal value.

Recommend 1, 2 and 3. Semantic layer: **not worth adding.** `dbt-metricflow==0.15.0`
resolves against the pins (dry run), so it is installable at $0, but on dbt Core it is a
separate CLI (`mf query`) with no API, no docs-site integration and no consumer in this
repo; it would be the third package feature ADR-0014 said to wait for, with nothing reading
it. `docs/METRICS.md` plus marts whose formulas are tested is the defensible choice; say in
the ADR that MetricFlow is the upgrade path if a BI consumer ever appears.

### Phase 4 — testing depth (do it; M)

- Native unit tests (dbt ≥ 1.8 `unit_tests:` YAML with `given` / `expect`; works on
  dbt-duckdb; for incremental models set `overrides: macros: is_incremental: false`):
  `fct_decision_policy` outcome (over-hit, over-miss, under-hit, push, no actual → null, no
  threshold → null decision); `dim_player_asof` (A→B trade, A→B→A return gives three stints
  with correct effective ranges); `fct_player_game.population` (minutes exactly 10.0, first
  game of season → `all`, second game ≥ 10 → `min10`); `slv_game_logs` dedup (two rows for one
  key: higher minutes wins; tie → `source_file asc`). One caveat to state: unit tests run on
  the model's SQL with mocked inputs, so they cannot exercise the `files_exist` branches.
- Source freshness: not on `game_logs.game_date` (it would error all off-season) but on the
  pipeline heartbeat: `daily_reports` with `loaded_at_field: generated_at`, `warn_after 2 days`,
  `error_after 4 days` (the nightly runs daily in every season), and `drift_reports`
  likewise; `load_manifest.loaded_at` as a third (warn if the warehouse was built from a load
  older than 8 days — the weekly refresh). `dbt source freshness` in the nightly job as a
  warning step and the result (`sources.json`) uploaded with the run artifacts.
- Severity policy, written down: **error** = grain, contracts, reconciliation to artifacts,
  known-missing-games absent; **warn** = plausibility, freshness, restatement lag (today
  `error`; a lag beyond the lookback should alert the owner to raise `lookback_days`, not
  block the slate's warehouse build — propose changing it, with the ADR-0004 recalibration
  note). `store_failures: true` on the reconciliation tests and the plausibility test, into
  schema `test_failures`; cheap in DuckDB and exactly the rows you would want when one fails.
- CI incremental job: on pull requests drop `--full-refresh` from the slim path
  (`ci.yml:78`) so `state:modified+` runs incrementally against the restored `main` DuckDB
  file; keep the full refresh on `main`. That exercises the real incremental SQL through dbt
  on every PR. One risk: a PR that changes an incremental model's columns will fail with
  `on_schema_change='fail'`, which is the right signal (then `--full-refresh` is the fix and
  the PR says so).
- Data-quality report on the docs site: dbt docs does not show test results. A zero-package
  option is a small script that renders `run_results.json` + `sources.json` to
  `site/tests.html` (pass/warn/fail per test, freshness per source) in the CI `dbt` job and
  links it from the docs overview. Propose; skip if it feels like a tick-box.

### Phase 5 — season readiness (do it; M)

The dry run that matters is the one that has never happened: a nightly with a slate and the
warehouse gate open. Fixtures: a synthetic `LeagueSchedule26_27.csv` over the fixture game
logs; the nightly runner with `--local-dump`, `--push` off, models stubbed (the harness
`tests/test_nightly.py` already injects `load_models`); then the warehouse path on a scratch
DuckDB: loader `--offline`, `dbt build --selector nightly` (not full refresh) over a
warehouse that already holds the replay rows, `export_gold`, `publish --dry-run`. Cases:
schedule absent (no_schedule, streak increments), present with no games, present with games
(slate, decisions, residuals the next night), a restated row inside the lookback on the
second night, a drift window under 500 rows, and a season label the dead `seasons` var does
not list (expect nothing to break; then delete or use the var). Also add 2026-27 to
`SEASON_CALENDAR` when the dates are known (labels only).

`docs/RUNBOOK.md`: one entry per signal — `nightly-warn` no-schedule issue (expected around
2026-10-14; what to check on Kaggle; close when the file appears), `nightly-hold` drift
issue (compare flagged features against the reference; the opening is the first
out-of-sample window), freshness warnings, restatement-lag warning (raise `lookback_days`,
full refresh), `assert_season_game_counts` failing after 150 days (declare an exception or
find the games), when a full refresh is required (schema change on an incremental model,
restatement older than 14 days, new season, model revision change), and how to read
`_export_manifest.json`.

Out-of-sample accumulation: `mart_policy_metrics` and `mart_drift` already separate
`run_kind = nightly` rows from replay rows. Add a gold view `mart_policy_out_of_sample`
(nightly rows only, season 2026-27, column `evaluation = 'out_of_sample'`, exported) and a
README section rendered from its export next to the in-sample table, with the in-sample
table relabelled "in-sample (thresholds chosen on these rows)". No new artifact format is
needed; the labelling is the deliverable.

### Phase 6 — presentation (do it; S–M)

README first screen: problem → warehouse design (lineage picture ending in the Phase 2
exposures) → how trust is established (contracts, reconciliation, unit tests, freshness) →
ML and agent as consumers. Keep the `pass-rates` markers (`agent-eval.yml` rewrites between
them; `test_agent_pass_rates.py` checks them) and the Limitations section as it is. Evidence
tables move to `docs/EVIDENCE.md`. "How this was built" paragraph. `docs/STORIES.md` gains
the new citable facts with keys; `docs/RESUME.md` holds the two bullets and three stories.

## 7. Open questions

1. **Branch base.** The working copy was at `bbfff6f`; I fast-forwarded `main` to `4f16543`
   and created `analytics-engineering` from it. Fine?
2. **Fix the `warehouse.yml` publish condition in Phase 1?** It is hygiene and it makes the
   README truthful, but once merged the Sunday 2026-10-11 run will push fresh `gold/` to the
   HF dataset repo. You said nothing goes to HF without your say-so: say so, or I leave the
   condition alone and document it.
3. **Vercel.** Please read the current deployment's commit in the Vercel dashboard; the
   served `/decisions` lacks the `4f16543` sentence (§2.3). If it is a stale ISR cache, a
   redeploy settles it; if the deploy is behind, the Owner TODO stays open.
4. **Phase 3 marts:** 1, 2 and 3 as proposed, or a different three?
5. **Names on facts:** keep with the point-in-time argument (recommended), or normalise?
6. **Restatement-lag test severity:** change to `warn` (recommended) or keep `error`?
7. **Silver contracts:** add (the incremental models are the schema risk) or record why not?
8. **Dead `seasons` var:** delete it and the mirroring assertion, or make `dim_date` consume it?
9. **HF model card push** (Owner TODO 5): yours to do after review, as before?
10. **Agent wording** ("push failed" on `pushed: false`): out of this brief's scope; do you want
    a one-line prompt fix queued separately?
11. **`frontend/lib/policy_summary 2.json`** is a stray local file on your machine (ignored by
    git); delete it yourself, or shall I?

## 8. Phase 1 record (2026-10-09, branch `analytics-engineering`)

What changed, in commit order: `dbt/.user.yml` untracked and anonymous usage stats off;
CI permissions scoped to the `pages` job; `warehouse.yml` publish condition fixed
(ADR-0022); `dbt/selectors.yml` with the `nightly` selector used by the Makefile and
`nightly.yml` (ADR-0023; the hand-typed list had omitted `slv_restatements`); sqlfluff with
a config matching the existing style, 27 whitespace-only fixes, token-identical compiled SQL
(ADR-0027); the `seasons` seed replacing the dead var, with `relationships` tests at
severity warn (ADR-0024); a contract on `slv_game_logs` (ADR-0025); the severity policy and
the `warehouse-warn` notifier (ADR-0026); `reports/runs.json` with `scripts/record_run.py` and
`tests/test_runs.py`; README counts rendered from the manifest with a parse-based test;
README, ADR statuses, `warehouse.yml` header, `docs/STORIES.md`, `docs/agent.md` and three
docstrings rewritten to the three-way status; `nba/rehearsal.py` and the `rehearsal` dispatch
input (ADR-0028); `tests/test_season_readiness.py` (two opening nights end to end on
fixtures: new season label, bronze families switching on, incremental build, residuals, a
restatement inside the lookback, export, dry-run publish); `docs/RUNBOOK.md`.

Verification: `make dbt-full` 106 of 106 (31 models, 4 seeds, 1 snapshot, 70 tests);
`make dbt-docs` with every column described; `make lint`, `make lint-sql`; `pytest` (the two
LightGBM files still fail at collection on this machine without `libomp`). Run ids for the
rehearsal dispatch and the post-merge `warehouse.yml` dispatch are recorded in
`reports/runs.json` when they exist.

Choices made on my recommendation (overrule any): restatement-lag test → warn (ruling);
silver contract on `slv_game_logs` only (ruling); the `nightly` selector is the whole project
including seeds (ADR-0023: a nightly must not depend on the weekly refresh having loaded the
`seasons` seed); the season `relationships` tests warn rather than error; the stray
`frontend/lib/policy_summary 2.json` on your machine was left alone (git-ignored, yours to
delete); the agent's "push failed" wording was left alone (out of scope).

Schedule dependency (ADR-0029): the Kaggle dump is at version 515, last updated
**2026-06-15**, with no `LeagueSchedule26_27.csv` (checked with your Kaggle key on
2026-10-09). The free, legitimate schedule fallback is in §9; the box-score dependency has no
free fallback and is the largest risk to the ledger.

## 9. Phase 0b — free-tier feasibility for the paper ledger (read-only, 2026-10-09)

Sources are the providers' own pages, fetched 2026-10-09; where only a third party states a
fact, it says so. Nothing was signed up for and no API was called. Full citations are in the
research log for this phase (summarised here with URLs).

### 9.1 Odds source

**The Odds API v4** (`the-odds-api.com`). The pricing table's "Starter" row is **FREE, 500
credits per month**, all sports, "all betting markets" (`https://the-odds-api.com/#get-access`).
Whether a card is needed is **not stated on any primary page**; third-party comparisons say
no card. Quota reset timing is not documented. Costing, from the v4 guide
(`https://the-odds-api.com/liveapi/guides/v4/`): `/odds` costs `markets × regions`; the
per-event odds endpoint costs `unique markets returned × regions`; `GET /v4/sports`,
`GET /v4/sports/{sport}/events` and the participants endpoint **do not count against the
quota**; scores cost 1 (2 with `daysFrom`); responses with no data do not consume quota;
"every group of 10 bookmakers is the equivalent of 1 region", and a `bookmakers` filter takes
priority over `regions`, so `bookmakers=draftkings` is one region-unit. Historical odds cost
10 per region per market and the historical endpoints are "only available on paid usage
plans" (the homepage row lists "Historical Odds" under Starter; the docs are authoritative,
so assume none at $0). Response headers `x-requests-remaining`, `x-requests-used`,
`x-requests-last` report the quota, which is what the Phase 2 credit guard reads.

NBA player props exist "one event at a time" through `/v4/sports/basketball_nba/events/{eventId}/odds`
with market keys `player_points`, `player_rebounds`, `player_assists` (plus `_alternate`
variants, threes, blocks, steals, turnovers, combos, `player_fantasy_points`)
(`https://the-odds-api.com/sports-odds-data/betting-markets.html`). Bookmakers in
`regions=us` include `draftkings`; **Caesars (`williamhill_us`) and Fanatics are marked
"Only available on paid subscriptions"**
(`https://the-odds-api.com/sports-odds-data/bookmaker-apis.html`). So at $0 the Maine-legal
pair collapses to **DraftKings only**; Caesars lines can enter only by hand (§9.3).

Terms (`https://the-odds-api.com/terms-and-conditions.html`, updated 2026-08-31) permit, in
their words, "Storing our data and retaining it indefinitely", "Calculating and displaying
values you derive from our data", "Displaying our data in a UI, website, or mobile app", and
training models on it; they prohibit reselling or redistributing the data "as a standalone
data product", including "downloadable files … intended to serve as a source of raw data for
others". Reading for this project: a **private** HF dataset repo or MotherDuck table of raw
lines is storage (allowed); public pages showing hit rates, ROI and coverage are derived
values (allowed); publishing the raw line snapshots publicly as files would be the prohibited
case, which the brief already rules out. Attribution is "not required, but is always
appreciated"; the terms also say to verify with the operator before acting on a line.

**Alternatives.** balldontlie: free tier (5 requests/minute) covers games, teams and players;
**odds and injuries require the GOAT tier ($39.99/month)**, and paid tiers come with a
48-hour trial that converts — excluded (`https://nba.balldontlie.io/`,
`https://www.balldontlie.io/terms.html`). OddsPapi: the vendor's blog claims 250 free
requests/month with no card, but the pricing page is a JavaScript calculator and the docs do
not list NBA prop keys or DraftKings coverage — unverifiable from primary sources, so not a
basis for the ledger. SportsGameOdds: a free "Amateur" plan of 2,500 *objects*/month with
props included; object metering makes it tight and the card question is unstated. odds-api.io:
"new free API keys are paused indefinitely". **Verdict: The Odds API free plan, DraftKings
only, is the one option whose costs and terms are documented well enough to build on.**

### 9.2 Credit budget, with arithmetic

Definitions: one *capture* = one night's lines for the chosen markets and games; the free
`/events` call lists the night's games at 0 credits; each per-event props call costs
`markets × 1` with `bookmakers=draftkings`. The 2025-26 regular season had 1,230 games on 164
game dates, **7.5 games per night on average** (`reports/replay_2025-26.json`
`n_dates`), typically 2 to 15.

| Option | Credits per average night (7.5 games) | Worst night (15 games) |
|---|---:|---:|
| three markets (pts, reb, ast), every game | 22.5 | 45 |
| points only, every game | 7.5 | 15 |
| three markets, half the games (by a fixed rule, e.g. the earliest tip-offs) | 11.25 | 22.5 |
| one book vs two | same (10 books = 1 region-unit); Caesars is not available free anyway | |

Budget: 500 credits/month minus a **15 % margin = 425**; I would hold a further fixed
**reserve of 50** for retries that must never happen automatically (so 375 planned).

- Three markets, every game: 375 / 22.5 ≈ **16 captures a month on average**, with a hard
  floor of 375 / 45 = 8 if every night were a 15-game night. A month has about 27 game
  nights, so this is roughly every other game night.
- Points only, every game: 375 / 7.5 = **50**, i.e. every game night of the month with room
  to spare (27 × 7.5 ≈ 200).
- Three markets, every game, **every night** would cost 27 × 22.5 ≈ 610 > 500: does not fit.

Over a season (≈ 164 game nights, 6 months, 3,000 credits): points-only every night ≈ 1,230
credits (fits); three markets every night ≈ 3,700 (does not fit); three markets on 16
pre-chosen nights a month ≈ 2,160 (fits with the margin). **Recommendation for the Phase 2
ADR:** three markets, DraftKings, every game on the night, on a **fixed set of weekdays**
(e.g. Tuesday, Friday, Sunday ≈ 13 nights a month ≈ 290 credits; or add Wednesday ≈ 17
nights ≈ 380 credits, right at the planned 375) — fixed in advance so the nights are chosen
by the calendar and not by you, which is the no-cherry-picking rule the brief asks for, and
the credit guard refuses any capture that would cross 425. Hand-entered nights
(`source = manual`, §9.3) cost nothing and are reported separately.

### 9.3 Manual fallback at zero credits

CSV template (`lines_manual.csv`, one row per player-market line, read from your own
sportsbook account before tip-off):

```
read_at_utc,game_date,book,player_name,team,market,line,over_price,under_price,notes
2026-10-20T20:05:00Z,2026-10-20,caesars,Jayson Tatum,BOS,player_points,27.5,-115,-105,
```

Validation on load (Phase 2): `read_at_utc` parses and is before the game's tip-off and after
the prediction's timestamp; `book` ∈ {draftkings, caesars}; `market` ∈ the three keys; `line`
is a half-point or whole number in a plausible range per market; prices are American odds
(≤ −100 or ≥ +100); `player_name` passes the same normaliser and mapping seed as API names,
unmatched rows go to the counted queue; duplicates on (game_date, book, player, market) are
rejected. **Is this alone enough for the ledger?** Yes for correctness: a hand-read line with
its timestamp settles exactly like an API line, and it is the only way Caesars lines enter.
No for scale and for trust: it is a few minutes of typing per market per night, and the
`read_at_utc` stamp is self-reported, so manual rows must be labelled `source = manual` and
reported separately from API rows as the brief requires (the API rows carry a server
timestamp and the response's `last_update`).

### 9.4 DraftKings salaries and scoring

The DraftKings rules and terms pages (`https://www.draftkings.com/help/rules/nba`,
`/help/terms`) returned **HTTP 403 to the research fetcher**, so the scoring values below are
from secondary sources and must be **confirmed by you in a browser** before Phase 5 uses
them: Point +1, three-pointer made +0.5, rebound +1.25, assist +1.5, steal +2, block +2,
turnover −0.5, double-double +1.5 (max 1), triple-double +3 (max 1); roster PG, SG, SF, PF,
C, G, F, UTIL; salary cap $50,000. The draft screen's "Export to CSV" (player, position,
salary, game info, DK average points) is referenced by third parties, not by a DraftKings
page I could read; confirm the link exists in your contest view. DraftKings' terms (mirrored
copy, 2026-09-13) prohibit "automated means … to obtain, collect or access any information
on the Website"; a manual export by a logged-in user is not automated, and that is the only
way salaries enter this project. No clause about personal use of one's own export was found.

### 9.5 Availability (injuries, expected starters)

No $0 source with documented terms reachable from a runner was found. Ranked: (1) the
official NBA injury report PDFs (hub `https://official.nba.com/nba-injury-report-2025-26-season/`;
files at `ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf`; deadlines
5 p.m. local the day before and 1 p.m. day-of) are the legitimate source, but the host is an
Akamai sibling of the blocked `cdn.nba.com`. **Probed from a runner on 2026-10-10 (run
`38010924812`): the PDF host answers (HTTP 200, `application/pdf`, 63,591 bytes for a 2025-26
report) while the `official.nba.com` hub page returns 403, and `stats.nba.com` / `cdn.nba.com`
are still blocked.** nba.com's Terms of Use allow downloading material "to any single computer
only for your personal, noncommercial use" and prohibit reproduction or redistribution; they
name nba.com and its sub-sites, not the Akamai host. Nothing is built on this (owner
decision): it would need PDF parsing, and a scheduled job is automated access the terms do not
address;
(2) balldontlie injuries: GOAT tier only; (3) ESPN's injuries JSON: undocumented, and Disney's
terms forbid "automated means" (`https://disneytermsofuse.com/english/`); (4) Rotowire and
similar: paid or scraping. **Recommendation:** the recommendations view carries the plain
flag "the model has no availability information" (already required by the brief), the ledger
settles did-not-play as the pre-registered rule says (void, counted and reported), and the
injury-report probe is run once to decide whether a PDF reader is worth building later.

### 9.6 Timing

- **Box scores → dump → ingest.** The dump's in-season cadence is undocumented (its page
  says "updated daily"; the last update was 2026-06-15, after the Finals). The nightly runs
  at 10:00 UTC (06:00 ET) and re-reads the last 7 days, so a box score the author publishes
  within a day lands the next morning; settlement of a paper bet is therefore **T+1 at the
  earliest** and later if the dump lags. ADR-0029 covers the case where it never arrives.
- **Predictions.** Stamped by the nightly at ~06:00 ET on game day (`generated_at` on every
  slate row), from game logs through the previous night.
- **Lines.** DraftKings posts NBA player props the morning of game day, sometimes the night
  before; a capture dispatched by you between **~15:00 and 18:00 ET** sees most games'
  props posted and precedes the earliest tip-off (19:00 ET); each API row carries the
  response's `last_update` and the request time.
- **Salaries.** Contest salaries are available once the slate is posted (typically the day
  before); a hand export any time before lock is fine.
- **Timestamp test (Phase 2):** `prediction.generated_at < line.read_at < game.tip_off`
  for every ledger row; a row that fails is excluded and counted, never silently reordered.

### 9.7 Every other free limit this plan touches

| Service | Limit (source) | Use today | Headroom / what exhausts it |
|---|---|---|---|
| GitHub Actions | free for public repos on standard runners; cache 10 GB per repo, entries evicted after 7 days without access; artifacts 90 days by default; 6 h per job (`docs.github.com/.../github-actions`, `dependency-caching-reference`) | ~8 nightly + ~1 weekly + CI runs; slim-CI state in cache | minutes: none for a public repo; the 7-day cache eviction already means the first PR after a quiet week does a full build (acceptable) |
| MotherDuck Lite | 10 GB storage, 10 h Pulse compute per month, no card (`motherduck.com/pricing`) | weekly refresh ≈ 1 min; nightly builds in season ≈ 1 min each → ≈ 35 min/month | compute 10 h: ~17× headroom; a ledger table adds kilobytes; a scratch rehearsal database would add one more ≈ 90 MB |
| Hugging Face | private storage 100 GB for free accounts; ≤ 100k files per repo (`huggingface.co/docs/hub/storage-limits`) | public dataset repo ≈ 50 MB; private lines repo would add ≈ 50 KB per captured night | nothing in sight |
| Groq free | `openai/gpt-oss-120b`: 30 RPM, 1,000 RPD, 8k TPM, **200k tokens per day** (`console.groq.com/docs/rate-limits`) | one brief ≈ 5k tokens/night | a pass-rate run (25 briefs) is half the daily cap; nothing new here uses Groq |
| The Odds API free | 500 credits/month | 0 today | §9.2 |
| Kaggle API | no documented quota for single-file downloads at this volume | one 390 MB download per night | an undocumented limit would surface as a download failure (red run) |
| Vercel Hobby | personal, non-commercial; builds and bandwidth within the hobby plan | 5 static pages | a private recommendations view adds one page; keep it behind the same static-file model (no server) |

### 9.8 What Phase 2 can and cannot promise at $0

Can: DraftKings points/rebounds/assists lines on a pre-declared subset of game nights (about
16 a month with three markets, or every night with points only), captured on demand with a
guard that fails closed; hand-entered Caesars or DraftKings lines on any night; a paper ledger
settled T+1 or later from the Kaggle box scores; public aggregates with sample sizes. Cannot:
Caesars lines through the API; historical odds for a backtest of the betting rule (paid
only — the rule is pre-registered without a line backtest, which is the honest position);
an availability signal; settlement independent of the Kaggle author. The last item is the
risk to name on every ledger page.

## 10. Phase 2 record (2026-10-10, branch `analytics-engineering`)

Owner steps completed: `warehouse.yml` dispatched after the merge (run `38010452785`,
MotherDuck, `PASS=106`, `gold/` pushed at HF `f197d840` — the first fresh export since
2026-09-16; the `seasons` seed is now on MotherDuck); the scratch-MotherDuck rehearsal ran
(`38011203767`, `delete+insert` on `nba_rehearsal`, dropped afterwards; ADR-0028); the
injury-report probe ran (`38010924812`, §9.5).

Built (ADR-0030, ADR-0031): The Odds API client and the fail-closed credit guard with its
usage log; the capture command and `capture.yml` (scheduled 20:00 UTC, guard-gated, decision
record before any priced call, out-of-scope games counted, no retries); the private lines
repo layout and loader pull; deterministic name matching with an alias file and unmatched
counts; the manual CSV path; the Odds API schedule fallback with provisional ids resolved by
(date, team) when box scores arrive; `tip_off_utc`, `minutes_mean_last10`,
`input_staleness_days`, `prediction_generated_at` and schedule provenance on every slate row;
the source-stall signal and issue; the pre-registered rule as data with a hash pinned in dbt;
`fct_paper_bet` (settlement computed, never stored; never exported), `mart_ledger`,
`mart_ledger_by_edge`, `mart_ledger_coverage`; the two timestamp tests; the private
recommendations view; the replay evidence artifact `reports/ledger_rule_evidence_2025-26.json`.

Expected on 2026-10-20: a slate (from the Kaggle file if it appears, else the fallback), a
capture of DraftKings points lines for every game after 16:00 EDT, matched names, **zero
qualifying bets** (every player is a season debut), and the first `pending` rows a day or two
later as players reach one prior game. The owner must add the `ODDS_API_KEY` secret; the
private repo is created on the first capture.

Cut or deferred: a public `/ledger` page (Phase 3 reads the gold JSON exports; the marts are
ready); `docs/METRICS.md` entries for the ledger metrics (Phase 4); native unit tests of the
settlement SQL against `nba.ledger.rule.settle` (Phase 4); alternate lines, live odds,
Caesars through the API (paid), any availability signal.

## 11. Phase 3 record (2026-10-10, branch `analytics-engineering`)

Scope as approved: SQL features with the row-for-row parity proof, the site reading gold
through JSON exports with the current files as fallback, and exposures. The `/ledger` page
deferred from Phase 2 is included.

**Features in SQL (ADR-0032).** `int_player_game_features` (new `intermediate` layer) renders
all 23 `asof_v1` features over `slv_game_logs`. The loader now also runs the Python feature
module over the loaded game logs and writes `features/asof_v1.parquet`
(`nba/warehouse/features.py`, ~75 s of pandas on 130k rows); `brz_features_asof_v1` types it,
and `assert_int_features_match_feature_module` full-outer-joins the two and fails on a row on
one side only, a null on one side only, any integer difference, or a double difference above
`tol_feature_parity`. Measured over 130,414 rows (`reports/feature_parity_asof_v1.json`,
`make parity`): 0.0 on 19 features, 2.13e-14 at most on the four minutes means, no null
mismatch; at tolerance 0 the test fails on 42,828 rows, all on those four columns. Tolerance
set at 1e-12 from that measurement (the audit's 1e-9 was a guess). `tests/test_feature_parity.py`
holds the var to the report, the SQL feature list to the module's, and proves the dbt test
fails on exactly one row when one value of the module's file is perturbed by 0.01. The Python
module still serves; the ADR records what switching would take.

**The site reads gold (ADR-0033).** `export_gold` writes `gold/<mart>.json` for the ten marts
with `meta.export_json` (sizes 6 B to 264 KB), the publisher pushes them, and
`frontend/lib/data.ts` gained `getGoldMart`, `pinnedRows`, `dailyRowsToDays` and
`goldSourceNote`. `/` (all-rows comparison and a training-population cross-check), `/replay`
(daily chart, season figures) and `/decisions` (hit rates, coverage, baselines, band coverage,
coverage curve) read the marts first and the files they read before as fallback; the headline
holdout table stays on `metrics.json`; thresholds, band quantiles and verdict text stay the
artifact's; every section names the source it rendered. New `/ledger` page: the rule from a
committed mirror (`frontend/lib/ledger_rule.json`, `python -m nba.ledger.rule --mirror`,
tested), coverage and pending counts, headline and manual ledgers, the split by side, the
"what this is not" paragraph; honestly empty until 2026-10-20. `next build` passes with every
gold file absent (all pages on their fallbacks).

**Exposures (ADR-0033).** `dbt/models/exposures.yml`: six pages, the agent, the Hugging Face
export (exactly the exported set), `nba.decisions.evaluate`, the drift reference/calibration
scripts, and the private recommendations view. `tests/test_dbt_exposures.py` keeps them in step
with the pages (every `page.tsx` has one; every mart a page fetches is listed and exported as
JSON), the export set, the scripts' `gold.<table>` reads, and the README count.

Verification: `dbt build --full-refresh` locally `PASS=140` (41 models, 4 seeds, 1 snapshot,
93 data tests, 1 unit test, 11 exposures); `dbt docs generate` with every column described;
`sqlfluff`; `ruff`; `pytest` (see the checkpoint message for the count); `next build`. Run ids
for CI and the post-merge `warehouse.yml` dispatch go into `reports/runs.json` when they exist.

Choices made on my recommendation: the parity tolerance 1e-12 (50× the largest observed
difference); the headline holdout table stays on `metrics.json` rather than the mart; the
mart's JSON exports include `mart_season_coverage` and `mart_restatement_lag` though no page
reads them yet (small, and the agent's fallback-free marts should be fetchable); `/ledger`
built now rather than in Phase 6; `/predictions` and `/brief` unchanged (the audit's cut).

## 12. Contingency: box scores for 2026-27 if the Kaggle dump stays silent (read-only, 2026-10-10)

Question from the owner: if the dump has not resumed nightly updates by 2026-10-22, what are
the free options with documented licences for 2026-27 player box scores, what would each take
to adopt, and what does waiting cost? Nothing below was built; every fact was read on
2026-10-10 with the owner's Kaggle key (REST `datasets/view` and `datasets/list`) or from the
provider's public pages.

**The primary.** `eoinamoore/historical-nba-data-and-player-box-scores`: CC0, version 515,
`lastUpdated 2026-06-15`, page still says "updated nightly" / "updated daily"; files list
`LeagueSchedule24_25.csv` and `LeagueSchedule25_26.csv`, no 2026-27 schedule. The page states
no plan beyond "updated daily"; the June pause matches the previous off-season pattern (the
2025-26 schedule file appeared before that season), so silence until 10-20 is not yet
evidence of abandonment.

**Options that are not $0 or not licensed for automated access (ruled out).**
balldontlie: per-game player stats need the ALL-STAR tier ($9.99/month); the free tier has
teams, players and games only (docs.balldontlie.io). Sports Reference: the data-use policy
forbids "any automated means to access or use the Site" without written permission and a
competing data store (sports-reference.com/data_use.html). MySportsFeeds: personal tier from
CAD 5/month. SportsDataIO: trial that converts. API-Sports: free plan exists (100 requests a
day) but its terms page returned 403 from here, so the licence is unverified and it is not
listed as an option. stats.nba.com / nba_api: blocked from runners (probe) and against the
owner's rule in any case.

**Free, licensed candidates on Kaggle (all currently paused for the off-season, like the
primary; all ultimately derived from NBA.com by their authors, as the primary is).**

| Dataset | Licence | Player box-score file | Last updated | Stated cadence | Adoption cost |
|---|---|---|---|---|---|
| `chevronronson/nba-stats-dataset` | CC0 | `csv/player_boxscores.csv` (767 MB, 1996–present) | 2026-08-01 | "Updates: Daily … occasionally backfilled" | a second loader mapping its columns to `nba/schema.py` (one module next to `kaggle_dump.py`), one 767 MB single-file download a night instead of 390 MB, a row-for-row reconciliation of the 2025-26 overlap against the stored game logs before any 2026-27 row is accepted, dataset-card attribution; same licence as today |
| `szymonjwiak/nba-traditional` | CC BY-NC-SA 4.0 | `traditional.csv` (94 MB, 1997–2026) | 2026-07-19 | not stated (slug says 1997-2023; title 1997-2026, so periodic rebuilds) | the smallest file and the closest shape (traditional player box scores); same loader work plus share-alike on the derived parquet on Hugging Face (the public dataset repo would need the same licence) and non-commercial, which this project already is; cadence is the unknown |
| `wyattowalsh/basketball` | CC BY-SA 4.0 | none (team `game.csv`, 2.3 GB play-by-play) | 2026-06-29 | daily and monthly updater notebooks published | player lines would have to be derived from play-by-play: not a box score source for this pipeline |
| `deocheng/nba-data-1946-2026` | Apache 2.0 | ESPN boxes as JSON columns inside a 92-table export | 2026-08-18 | not stated | mixed provenance (Basketball-Reference crawl ids, ESPN JSON, dunksandthrees), heavy; ruled out |

**Cost of simply waiting.** Predictions continue (the Odds API `/events` schedule fallback,
ADR-0029) but no bet settles, and the pre-registered rule's staleness cap turns a stall into
an empty ledger, not a wrong one: `input_staleness_days` grows a day at a time, every row
crosses the 10-day cap by about 10-30, and from then on the ledger places nothing (the
coverage mart shows `lines_ineligible` rising and `bets_total` flat). Bets placed in the first
ten days stay `pending` and settle idempotently on the first build after the dump resumes.
The drift check reports `insufficient`; the `source-stalled` issue opens after two slated
nights without a box score (`STALL_WARN = 2`). Every day of waiting after about 10-30 is a
day of lost out-of-sample sample size (≈ 40-60 eligible lines a night at the rule's floor,
per the 2025-26 replay proxy), and the 1,400-bet power arithmetic in ADR-0030 already says one
season of points lines is not enough; losing weeks makes the final verdict "too early" more
likely. Nothing is corrupted by waiting; the loss is sample size.

**Recommendation.** Wait to 2026-10-22 as asked, then decide on one number: whichever of the
primary and `chevronronson/nba-stats-dataset` shows a `lastUpdated` on or after 2026-10-21.
If only the latter has resumed, adopt it as a second source behind the existing ingest
contract (same schema, same reconciliation test, same CC0 terms, attribution on the dataset
card), with the overlap reconciliation as the gate; budget two working days. If neither has
resumed by 2026-10-29 (ten days, the rule's staleness cap), re-run this check against
`szymonjwiak/nba-traditional` and accept the share-alike consequence if it has resumed. Do not
build a fallback before 10-22: both candidates are as silent as the primary today, and a
loader for a source that may never resume is work the owner has not asked for.

## 13. Done, cut, and what to open first

Filled in at the end of the pass.
