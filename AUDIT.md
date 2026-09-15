# AUDIT.md — read-only audit of nba-ai-ml before the template pass

Snapshot: branch `main` at commit `3fb19bc` (2026-09-15), audited on 2026-09-15 with the
`evidence-first-ml-pipeline` skill (`references/audit-checklist.md`) before any file on branch
`template` was changed. Every claim cites a path and line; "unclear" means the code could not
settle it. Section 14 adds the template boundary classification (Generic / Parameterizable /
Domain) that the second-use checklist asks for, and section 15 records the discrepancies the
owner had already found plus the ones this audit added.

Status vocabulary used throughout: **implemented** (code exists and has run in GitHub Actions),
**prototyped** (code and tests exist, has not run in Actions), **planned** (described only).

## 0. Snapshot

- 117 commits. Two bursts: 2025-08-04 … 2025-08-09 (the pruned prototype) and 2026-09-11 …
  2026-09-15 (the v2 rebuild, merged from `v2` via PR #4 on 2026-09-12). Nothing in between.
- Branches: `main` (default, renamed from `master` on 2026-09-15: commit `3fb19bc`), `v2`
  (merged). No unmerged branch; the live site is built from `main` (section 8).
- Tree (tracked, depth ≤ 3): `.github/workflows/` (7 files, 591 lines), `docs/` (2 files,
  464 lines), `frontend/` (21 files, 4,000 lines incl. `package-lock.json`), `nba/` (29 files,
  4,960 lines), `reports/` (5 JSON files, 1,760 lines), `tests/` (21 files, 3,471 lines plus
  6 recorded traces, 120 KB), root: `.gitignore`, `LICENSE`, `README.md`, `pyproject.toml`.
- No tracked file exceeds 1 MB. No `.dockerignore`, no Dockerfile, no notebooks.
- `.gitignore` rules that matter later: `*.json` and `*.parquet` are ignored globally
  (`.gitignore:75-76`) with re-includes only for `reports/metrics.json`,
  `reports/replay_*.json`, `reports/agent_*.json` (`.gitignore:129-132`); `*.db` is ignored
  (`.gitignore:114`); `/lib/` is root-scoped (`.gitignore:12`) so `frontend/lib/` is tracked;
  `/models/`, `/predictions/`, `/residuals/`, `/replay/`, `/daily_reports/`, `/brief/`, `data/`
  are ignored (`.gitignore:55,140-146`). Any new committed JSON (provenance, policy, drift
  calibration) needs its own re-include rule.

## 1. Canonical paths

There is no server. Every launch point is a `python -m nba.<module>` CLI invoked by a workflow:

| Launch point | Invoked by | Writes |
|---|---|---|
| `nba/nightly.py` (`main`, `nightly.py:180`) | `.github/workflows/nightly.yml:62-64`, cron 10:00 UTC | `data/daily_report.json`, `daily_reports/<date>.json`, `residuals/<date>.parquet`, `predictions/rolling_metrics.json`, `brief/*`, `predictions/<date>.parquet`, `predictions/latest.json`; pushes them to the HF dataset repo (`nightly.py:167-172`) |
| `nba/models/train.py` | `train.yml:47` (dispatch) | `models/lgbm_{pts,reb,ast}.txt`, `reports/metrics.json` (uploaded as an artifact; committed by hand) |
| `nba/models/publish.py` | `train.yml:61` when `publish=true` | HF model repo: model files, `README.md` (model card), `metrics.json` |
| `nba/predict/replay.py` | `replay.yml:53-57` (dispatch) | `reports/replay_<season>.json`, `replay/<season>/{replay,daily_mae,sample_<date>}.json`, `replay/<season>/residuals/<date>.parquet`; pushed to HF |
| `nba/agent/loop.py` | `agent.yml:67` (dispatch) and `nightly.py:138` | `brief/<date>.json`, `.trace.json`, `index.json`, `latest.json` |
| `nba/agent/pass_rates.py` | `agent-eval.yml:58-60` (dispatch) | `reports/agent_pass_rates.json` (artifact; committed by hand) |
| `nba/agent/evals.py` | tests and by hand | `reports/agent_evals.json` |
| `nba/ingest/kaggle_backfill.py` | by hand only (needs the extracted dump) | per-season parquet + dataset card push |
| `.github/workflows/probe-nba-api.yml` | dispatch | nothing (connectivity probe) |

Requirements: one `pyproject.toml` with exact pins (`pyproject.toml:12-29`). No conflicting
pins. `nba_api==1.11.4` and `pyyaml==6.0.3` are declared (`pyproject.toml:18-19`) but no module
under `nba/` imports either (grep `^(import|from) (nba_api|yaml)` → none); the probe workflow
installs its own `nba_api==1.4.1` (`probe-nba-api.yml:30`). `groq` is imported lazily inside
functions (`nba/agent/loop.py:161,175`), `lightgbm` lazily in `nba/predict/model.py:34` and at
module level in `nba/models/train.py:19`.

Import health: `tests/test_cli_isolation.py:14-25` proves `nba.ingest`, `nba.storage` import
without LightGBM. On this Mac `import lightgbm` fails (no `libomp`), so `pytest` errors at
collection on `tests/test_model.py` and `tests/test_train.py` (7 tests); the remaining 112 tests
pass. CI runs on `ubuntu-latest` where all collect (`ci.yml:31`; last runs green).

Verdict: the canonical runtime path is `nightly.yml → nba.nightly.run → {kaggle_daily.run_daily,
residuals.compute, agent_loop.run_and_write, slate.run_slate} → hf.push_products → Vercel pages
fetch HF resolve/main`. No file is redundant to that path except the two unused dependency pins.

## 2. Data sources and licensing

| Source | Where | Auth | Rate limit / cache | On canonical path |
|---|---|---|---|---|
| Kaggle dataset `eoinamoore/historical-nba-data-and-player-box-scores` (CC0) | `nba/config.py:59`, `nba/ingest/kaggle_daily.py:43,54-92` (single-file API, `PlayerStatistics.csv` ~400 MB, `TeamHistories.csv`, `LeagueScheduleYY_YY.csv`) | `KAGGLE_USERNAME`/`KAGGLE_KEY` (`config.py:126-127`) | full re-download nightly; no cache between runs; 900 s timeout (`kaggle_daily.py:44`) | yes |
| HF dataset `cbratkovics/nba-game-logs` | `config.py:53`, `nba/storage/hf.py` | `HF_TOKEN` for writes | `snapshot_download` per run | yes (system of record) |
| HF model `cbratkovics/nba-stat-predictor` @ `fb427de1…` | `config.py:54,69`, `nba/predict/model.py:47-64` | none for reads | `hf_hub_download` cache | yes |
| Groq (`openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b`) | `config.py:73-77`, `nba/agent/loop.py:151-218` | `GROQ_API_KEY` | free tier: 200k tokens/day, 8k/min, 1k req/day (`docs/agent.md:211-214`) | yes (brief step; never fails the job, `nightly.py:131-153`) |
| stats.nba.com / cdn.nba.com | `probe-nba-api.yml` only | none | unreachable from Actions runners (`docs/reconciliation.md:223-226`) | no |

No HTML scraping. Training data is reproducible from a tracked script given the Kaggle dump
(`nba/ingest/kaggle_backfill.py`), and the dump findings are recorded in
`docs/reconciliation.md:10-121`. Fixtures: `tests/conftest.py` generates a synthetic dump
(no data files tracked); `tests/traces/*.trace.json` (6 files, 120 KB) are recorded Groq
conversations read by `nba/agent/evals.py:161-178` and `tests/test_agent_evals.py`.

Licensing note the cards get wrong: both the dataset card and the model card on HF say daily
updates come from `nba_api` (dataset card "Provenance" bullet 2; `nba/models/publish.py:93-94`
"`nba_api` for updates"). The code has no nba_api path; the nightly source is the Kaggle dump
(`nba/nightly.py:80`, `nba/ingest/kaggle_daily.py`). See section 15, item 2.

## 3. Storage

- Game logs: one parquet per season, grain `(player_id, game_id)`, enforced by
  `nba/schema.py:126-129` on every read (`nba/storage/local.py:36-42`). 130,414 rows over five
  seasons (`docs/reconciliation.md:12-18`).
- Predictions: `predictions/<date>.parquet`, grain `(player_id, game_id)`; uniqueness is
  guaranteed by the roster rule (`nba/predict/slate.py:120-129`), not asserted at write time.
- Residuals: `residuals/<date>.parquet` (nightly) and `replay/2025-26/residuals/<date>.parquet`
  (164 files); same columns (`nba/predict/residuals.py:39-65`); no uniqueness assertion.
- Rolling metrics: `predictions/rolling_metrics.json` (JSON Lines, replace-by-date,
  `residuals.py:179-190`). Not present in the dataset repo yet (no slate has run in-season).
- Daily reports, briefs: JSON per date; `brief/index.json` and `brief/latest.json` rebuilt from
  the local `brief/` folder on every write (`nba/agent/loop.py:553-563`), which only lists
  dates present locally after `pull_products` (`nightly.py:112-113`).
- The app boots without every service: pages return `null` and a message on HTTP 404
  (`frontend/lib/data.ts:208-218`). Writers do not silently fail: `hf.push_products` raises
  without a token (`hf.py:38-42`).
- No database anywhere.

## 4. Features and leakage

One feature module, `nba/features/asof.py`, used by training (`nba/models/train.py:105`), the
slate (`nba/predict/slate.py:141` via `features_for_pending`) and the replay (through the slate).

| Feature | Formula | Window | Shift inside entity group |
|---|---|---|---|
| `{pts,reb,ast,minutes}_mean_last{5,10,20}` | rolling mean of prior games | 5 / 10 / 20 games | yes: `s.shift(1).rolling(w, min_periods=1)` inside `groupby("player_id")` (`asof.py:50-52,79`) |
| `{stat}_mean_season` | expanding mean of prior games in the same season | season to date | yes: `groupby(["player_id","season"])` + `shift(1).expanding()` (`asof.py:55-57,80`) |
| `games_played_season` | `cumcount()` within player-season | — | count of prior rows (`asof.py:82`) |
| `days_rest`, `back_to_back` | diff of `game_date` within player | — | `groupby("player_id")["game_date"].diff()` (`asof.py:84-86`) |
| `home` | schedule flag of the target game | — | not history; fixed before tip-off (`asof.py:87`) |
| `{pts,reb,ast}_mean_vs_opp` | expanding mean of prior games vs the same opponent | all prior | yes (`asof.py:89-90`) |

Sorting is `["player_id","game_date","game_id"]` before any groupby (`asof.py:69`); no
positional `.values` assignment after a flattened shift; no scaler; no target encoding; no
season-end aggregate. Pending rows are appended with zeroed stats and can only see earlier rows
(`asof.py:107-135`). Leakage tests exist on the synthetic fixture: `tests/test_asof.py` (7 tests,
including recomputation from earlier rows and a perturbation check). Not yet proven on real rows
(the template's checklist item C.2).

Serving vs training parity: the same function builds both; `nba/predict/model.py:67-76` refuses
a booster whose `feature_name()` differs from `asof.FEATURE_COLUMNS`.

One caveat: `get_upstream_freshness` (`nba/agent/tools.py:127-155`) reports `days_stale` from
the newest row in the *dump*, which includes playoff rows the pipeline never ingests; the
2026-09-14 brief therefore says "94 days stale" (dump newest 2026-06-13) next to "156 days"
(stored newest 2026-04-12). Not leakage, but a misleading number on the page.

## 5. Models and artifacts

| Artifact | Where | Size | Format / loader |
|---|---|---|---|
| `lgbm_pts.txt`, `lgbm_reb.txt`, `lgbm_ast.txt` | HF model repo @ `fb427de1…` (commit "Update model", 2026-09-12 02:00 UTC) | 1.73–1.74 MB each | LightGBM text; `lgb.Booster(model_file=…)` (`model.py:41`) |
| `metrics.json` | same repo, copy of `reports/metrics.json` | 3.0 KB | read by the site (`frontend/lib/data.ts:18,240`) |
| `README.md` | same repo | 3.1 KB | model card rendered by `publish.py:22-97` |

Training: `LGBMRegressor` with `config.LGBM_PARAMS` (`config.py:109-120`), one per target,
`n_features_in_` = 23 (`reports/metrics.json` `features`). No ensemble, no registry, no
intervals: predictions are point estimates (`slate.py:38-59`). Loaders: `load_from_hub`
(pinned revision) and `load_from_dir` (tests). Nothing loads at "startup" because there is no
server. The HF model repo has two commits (`49031b82` initial, `fb427de1` current).

## 6. Trace of the published numbers

Every figure on the site is fetched from a published file (`frontend/lib/data.ts:1-8`); every
figure in the README is typed by hand and cites a report. Classification per literal:

| Number | Where shown | Origin | Class |
|---|---|---|---|
| pts MAE 4.764 / 4.908 / 4.946, RMSE 6.160, R² 0.457; reb 1.942 / 2.009 / 2.008; ast 1.431 / 1.460 / 1.456; n 22,244; 88,257 train rows; 22,630 holdout rows | README:17,59-65; site `/` (from `metrics.json` at HF model revision `fb427de`) | `reports/metrics.json` `metrics.<target>.<predictor>.{mae,rmse,r2,n}`, `split.*`; generated by `train.yml` run 34666354516, git `50a3b2e` | generated artifact |
| replay diffs +0.0021 / +0.0008 / +0.0010, 164 dates, 22,075 restricted rows, 38,372 predicted, 26,031 with actuals, 617 / 367 unpredicted | README:18,67; `docs/reconciliation.md:203-218`; site `/` (from HF `replay/2025-26/replay.json`) | `reports/replay_2025-26.json` (`replay.yml` run 34673015904, pushed at HF `8a127c93`) | generated artifact |
| all-rows MAE 4.864 / 2.023 / 1.403 | site `/` "All replayed rows" column | `reports/replay_2025-26.json` `mae_unrestricted` | generated artifact |
| `/replay` "row-weighted season MAE" model 4.864 · baseline 4.679 (pts), 2.023 · 1.939 (reb), 1.403 · 1.374 (ast) | site `/replay` | computed in the page (`frontend/app/replay/page.tsx:20-25`) from HF `replay/2025-26/daily_mae.json` (26,031 rows); **not in any committed report** | published product, page-computed |
| grounding 6 of 6, golden 5 of 5 | README:19,73-74; `docs/agent.md:137-147` | `reports/agent_evals.json` (trace replay, no network) | generated artifact |
| pass rates 10 of 25, 9 grounded, 10 golden | README:75; `docs/agent.md:179-186` | `reports/agent_pass_rates.json` (`agent-eval.yml` run 34701742530) | generated artifact |
| brief latency, tool calls | site `/brief` | `brief/<date>.json` on HF | published product |

Arithmetic consistency: MAE and RMSE pairs are consistent (RMSE > MAE for every predictor); the
replay diffs equal `mae_restricted − reference.model_mae` to four decimals; `n_with_actuals +
n_missing_actuals = n_predicted` (26,031 + 12,341 = 38,372). Split code is temporal
(`nba/models/train.py:46-50`: train seasons vs holdout season; no shuffling). No notebooks.

Two provenance wrinkles found while tracing:

1. The site's `/` reads HF `replay/2025-26/replay.json` at `resolve/main`, whose current copy
   was produced by a *later* replay run (generated 2026-09-12 06:12:48, git `0f426d2`, HF
   commit `6cbc915b`) than the committed `reports/replay_2025-26.json` (05:15:10, git
   `e2e7a9c`, HF commit `8a127c93`). All metric fields and all 164 `per_date` rows are
   identical; only `generated_at`, `git_sha` and the recorded `dataset_revision` differ
   (`b20b5601` vs `8a127c93`, whose five parquet files have identical LFS SHA-256).
2. `reports/metrics.json` carries no file-level hash of its inputs: `dataset.version` is the
   HF commit `hf:b20b5601…` (repo revision), not a SHA-256 of the parquet files, and there is
   no hash of the model files or of the report itself (section 15, item 6).

## 7. API surface

None. There is no FastAPI, no Space, no route. `frontend/lib/data.ts` fetches nine public URLs
directly from Hugging Face (`data.ts:232-243`). No auth, payments or subscriptions anywhere.

## 8. Frontend

Next.js 14.2.25 (`frontend/package.json:16`), four routes, all server-rendered with
`revalidate = 3600` except `/brief` which is `force-dynamic` (`frontend/app/brief/page.tsx:6`).

| Page | Data source (URL) | Notes |
|---|---|---|
| `/` (`app/page.tsx`) | `metrics.json` at `https://huggingface.co/cbratkovics/nba-stat-predictor/resolve/fb427de…/metrics.json`; `replay/2025-26/replay.json` at dataset `resolve/main` | population named as "at least 10 minutes … same 22,244 holdout rows" (`page.tsx:66-68`); footnote labels the git commit `50a3b2e` as "model revision" (`page.tsx:103`) while `MODEL_REVISION` in `data.ts:13` is the HF commit `fb427de` |
| `/predictions` | `predictions/latest.json`, `predictions/rolling_metrics.json`, `replay/2025-26/daily_mae.json` | off-season copy shown (both files 404); table shows `lastLine.mae` (unrestricted) without naming the population (`predictions/page.tsx:63-77`) |
| `/replay` | `daily_mae.json`, `sample_2026-04-12.json` | population string from the file (`replay/page.tsx:36`); row-weighted means computed in the page; nowhere says the headline uses a different population or that the baseline wins here |
| `/brief` | `brief/index.json`, `brief/latest.json`, `brief/<date>.json` | heading is the brief date (`brief/page.tsx:67`), run date only in the footer (`page.tsx:141-143`); the date list mixes replay dates (2025-10-23 … 2026-04-12) and live off-season dates (2026-09-11 …) with no label |

Env vars expected: none (all URLs are constants). Vercel: root directory `frontend/`, no
`vercel.json`; a `VERCEL_TOKEN` repository secret dated 2025-08-04 exists but no workflow uses
it. Verified on 2026-09-15 that the live `https://nba-ai-ml.vercel.app/` renders the same 93
numeric tokens as a local `next build` of `main` (same multiset), i.e. the deployment matches
`main`. No "illustrative/demo" labels; every figure is presented as measured and is.
Unused packages: `framer-motion`, `class-variance-authority`, `tailwind-merge`, `clsx`
(`package.json:12-21`; no import under `app/`, `components/`, `lib/`). `components/charts/` is
an empty directory.

## 9. Tests and CI

- 118 tests across 19 files (`grep -c "def test_"`): backfill 25, daily ingest 10, agent loop
  11, tools 7, asof 7, schema 6, nightly 6, evals 6, replay 5, slate 5, residuals 4, dataset
  card 4, pass rates 4, model 4, train 3, hf push 3, schedule 3, cli isolation 3, hf pull 2.
- Coverage that matters for the template: leakage (`test_asof.py`), grain (`test_schema.py`),
  replay equivalence (`test_replay.py`), roster rule (`test_slate.py`), agent grounding
  (`test_agent_evals.py`, replayed traces). No test reads a real HF file; no test asserts that
  README or card numbers match `reports/*.json`.
- Local: 112 passed with `--ignore tests/test_model.py --ignore tests/test_train.py` (LightGBM
  cannot load here); `ruff check .` and `ruff format --check .` clean.
- CI (`ci.yml`): ruff lint + format + pytest on every push and PR. No frontend build in CI, no
  dbt, no docs build.
- Actions history: `nightly.yml` 5 runs (1 dispatch on 2026-09-12 and 4 scheduled 09-12 …
  09-15), all green, all off-season "no games"; `train.yml` 2 runs (2026-09-12); `replay.yml` 3
  runs (one failure 34676082455 then success 34676187318); `agent-eval.yml` 1 run
  (34701742530, rate-limited after 10 briefs).

## 10. Configuration, secrets, hygiene

- Env vars read: `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GROQ_API_KEY`, all in
  `nba/config.py:124-129` only. Documented in README:91. Repository secrets present:
  those four plus the stale `VERCEL_TOKEN`.
- Committed credentials: none (`git grep` for `supabase.co|postgresql://|sk-|gsk_[A-Za-z0-9]|hf_…|api_key=|password=`
  hits only the redaction code and its tests: `nba/agent/loop.py:524`, `tests/test_agent_loop.py:169-179`).
  `git ls-files | grep -i env` → nothing; a local `.env` exists and is ignored.
- Personal data: player names and ids (public box-score data). `reports/agent_pass_rates.json`
  embeds Groq error strings that include the owner's Groq organisation id
  (`org_01m…`) in 15 `error` fields; not a credential, but an identifier worth scrubbing.
- Dead code: none by import graph. `nba/ingest/kaggle_backfill.py` re-exports `kaggle_dump`
  names for tests (`kaggle_backfill.py:22-94`). No TODO/FIXME. No notebooks.

## 11. Claims vs code

| Claim | Where | Evidence | Status |
|---|---|---|---|
| Points MAE 4.764 vs 4.908 (last-10) on the full 2025-26 season | README:17 | `reports/metrics.json`; population = minutes ≥ 10, both baselines defined, 22,244 rows | Implemented (population stated in README:59, not in line 17) |
| Replay reproduces holdout metrics within +0.0021 | README:18,67 | `reports/replay_2025-26.json` `diff_vs_metrics_json`, `passed: true` | Implemented |
| Agent grounding 6 of 6 briefs | README:19,73 | `reports/agent_evals.json` (trace replay) | Implemented |
| Pass rates 10 of 25 (incomplete) | README:75 | `reports/agent_pass_rates.json` `overall.rate_limited = 15` | Implemented, incomplete |
| Nightly ingest at 10:00 UTC, $0 | README:50, `nightly.yml:16` | 4 scheduled runs green | Implemented |
| "Predict… evaluated on the full 2025-26 season against the player's last-10-game mean" | README:17 | true on the headline population; on all replayed rows (26,031) the last-10 mean has the lower MAE on all three targets (section 15, item 1) | Partially (population-dependent) |
| Daily updates from `nba_api` | HF dataset card "Provenance"; HF model card "Data note" | no nba_api code path; nightly reads Kaggle (`nightly.py:80`) | Contradicted |
| "they will be backfilled from nba_api" (seven missing games) | HF dataset card; `nba/storage/dataset_card.py:54` | no code; `docs/reconciliation.md:220-226` says it must run locally | Planned only |
| Model identity | README:91 ("model revision" in config), `docs/reconciliation.md:200` (`fb427de`), site `/` footnote (`50a3b2e` labelled "model revision"), `/replay` and `/predictions` (`fb427de`) | two ids for one model: git commit `50a3b2e` (training code) and HF commit `fb427de` (published files) | Implemented, inconsistently labelled |
| GitHub topics `mlflow`, `xgboost`, `fastapi` | repository settings (as reported by the owner) | `gh api` on 2026-09-15 returns `github-actions, huggingface, lightgbm, llm-agents, mlops, nextjs, sports-analytics` — the three were already gone | Resolved before this audit |
| Repository layout block | README:118-133 | matches the tree | Implemented |

## 12. Fitness for the target architecture

- Nothing blocks scoring end to end without a database: it already runs that way.
- Heavy deps: `lightgbm` (needs OpenMP) is imported only by training and scoring; ingest, storage,
  agent and the site never load it. `scikit-learn` is used for three metric functions only
  (`train.py:22`). The minimal runtime set for the nightly job is pandas, pyarrow, lightgbm,
  huggingface_hub, requests, python-dotenv, groq. Drop `nba_api`, `pyyaml`.
- No image is built; the job runs on `setup-python` (pip cache). Estimated install: ~1 minute.
- Keep: `nba/` (all), `tests/`, `frontend/` (four pages, `lib/data.ts`), `reports/*.json`, the
  five workflows plus the probe.
- Refactor (later phases): provenance and identity (Phase 0), warehouse over the HF files
  (Phase 1), population-aware pages and the decision policy (Phase 2), drift (Phase 3), agent
  tools over marts (Phase 4).
- Delete: `nba_api` and `pyyaml` pins; `framer-motion`, `class-variance-authority`,
  `tailwind-merge`, `clsx` from `package.json`; the empty `frontend/components/charts/`;
  the stale `VERCEL_TOKEN` secret (owner).
- Top risks, ordered:
  1. **The headline hides a population where the baseline wins.** Mitigation: name the
     population on every page, publish the all-rows comparison with its baseline next to it
     (Phase 0 fix 1), make the decision policy report both (Phase 2).
  2. **Cards contradict the code** (nba_api). Mitigation: render both cards from config
     (Phase 0 fix 2).
  3. **No file-level provenance**: a re-push of the same commit could not be detected.
     Mitigation: `reports/provenance_b20b5601.json` (Phase 0).
  4. **Silent replay-report drift**: the site reads whichever `replay.json` was pushed last;
     the committed report is a different run. Mitigation: the provenance file records both
     hashes; Phase 1 reconciles gold to the committed report.
  5. **Kaggle dump as the only automated source**: if the author stops publishing
     `LeagueSchedule26_27.csv`, the nightly job never produces a slate (exits 0 forever).
     Mitigation: the freshness tool already reports it; Phase 3 should turn a long
     no-schedule streak into a WARN with an issue.
  6. **Pass-rate measurement is incomplete** and one 25-brief run exhausts the daily cap
     when anything else runs. Mitigation: spread across days, merge per date (Phase 0).
  7. **Groq quota shared with the nightly brief**: the same key serves nightly briefs and
     evals. Mitigation: schedule evals away from the nightly window and cap per run.
  8. **`brief/index.json` rebuilt from the local folder** after `pull_products`; a partial
     pull would drop dates from the index. Mitigation: build the index from the HF listing
     or assert the pulled count.
  9. **Freshness reports playoff rows** as "newest game" (section 4). Mitigation: filter
     the dump by the regular-season rules before taking the max (Phase 4).
  10. **A 83-point box score** (Bam Adebayo, game `0022500938`, 2026-03-10, source
      `kaggle_v515`) is the season maximum and the golden-set outlier; whether it is real is
      unverifiable from the dump alone. Mitigation: a plausibility range in the silver
      contracts (warn, not fail) and a note on the golden set.

## 13. Open questions for the owner

1. Is the 83-point row above a real box score or an upstream error? (It drives the
   `2026-03-10` golden-set case.)
2. Should the stale `VERCEL_TOKEN` secret be deleted? Nothing reads it.
3. May the regenerated dataset and model cards be pushed to Hugging Face from this branch, or
   do you push them yourself after review? (Pushing the model card creates a new commit on the
   model repo; the pinned revision `fb427de` and the model files are untouched.)
4. Is the daily cron for the pass-rate spreader acceptable on `main`, or dispatch-only?

## 14. Template boundary (Generic / Parameterizable / Domain)

| Row | Path | Class | Interface / variables |
|---|---|---|---|
| config | `nba/config.py` | Parameterizable | seasons, targets, min minutes, repo ids, model revision, Groq model, thresholds |
| schema | `nba/schema.py` | Domain | `SourceLoader` (column contract) |
| loader (backfill + daily) | `nba/ingest/kaggle_dump.py`, `kaggle_backfill.py`, `kaggle_daily.py`, `schedule.py` | Domain | `SourceLoader`; the reconciliation logic in `kaggle_daily.classify/merge_rows` is Generic |
| target | none (targets are raw columns `pts`, `reb`, `ast`) | Domain | `TargetSpec` is trivial: `derive` = identity, `reconcile` = empty |
| features | `nba/features/asof.py` | Domain | `FeatureModule` (`FEATURE_VERSION` missing; 23 features) |
| trainer / evaluator | `nba/models/train.py`, `evaluate.py` | Parameterizable | targets, baselines (last-10, season mean), holdout season |
| publish / model card | `nba/models/publish.py` | Parameterizable | repo id, card text |
| storage | `nba/storage/{hf,local,dataset_card}.py` | Generic (hf, local) / Domain (card text) | prefixes from config |
| scoring | `nba/predict/{model,slate,residuals,replay}.py` | Parameterizable | roster rule (Domain: last-10 team games), targets, min minutes |
| scheduled job | `nba/nightly.py`, `.github/workflows/nightly.yml` | Parameterizable | cron, secrets, step names |
| agent | `nba/agent/*` | Parameterizable (loop, evals, pass rates) / Domain (tool bodies, prompt) | seven-tool contract |
| workflows | `ci.yml`, `train.yml`, `replay.yml`, `agent.yml`, `agent-eval.yml`, `probe-nba-api.yml` | Parameterizable | package name, secrets |
| frontend | `frontend/app/*`, `lib/data.ts` | Parameterizable (data layer) / Domain (copy) | repo ids, paths, labels |
| docs | `README.md`, `docs/reconciliation.md`, `docs/agent.md` | Domain | content files |
| dbt | none yet | — | Phase 1 |

Vocabulary decided for the template seams (the brief's fixed decisions override the
template's ordinal-period rule): entity `player` / `player_id`; period grain `game_date` +
`game_id` (not an ordinal); season `2021-22` … `2025-26` strings; cohort: none (one model per
target; `population` in {`min10`, `all`} plays the cohort role in marts); targets `pts`,
`reb`, `ast`; bands from residual quantiles (Phase 2); `periods_per_season` not used.

Hard-coded strings that would break on a second domain: team abbreviations (`SAN`),
`FULL_SEASON_GAMES = 1230`, game-id prefixes `002`/`006`, the Cup labels, `MIN_MINUTES`,
`ROSTER_LOOKBACK_GAMES`, the seven missing game ids, and the Groq model ids.

## 15. Discrepancies

Items 1–6 were reported by the owner; 7–9 were added by this audit.

1. **`/replay` population vs headline.** `/replay` computes row-weighted season MAE from
   `daily_mae.json` on 26,031 rows (all replayed rows with actuals and a last-10 baseline);
   there the last-10 mean beats the model on every target (pts 4.679 vs 4.864; reb 1.939 vs
   2.023; ast 1.374 vs 1.403). The overview headline uses minutes ≥ 10 with both baselines
   (22,244 rows) where the model wins. Neither page says the populations differ.
   *Fix in Phase 0:* name the population on every page that shows a metric; publish the
   all-rows comparison with its baseline column next to the headline.
2. **Cards say `nba_api` is the daily source; code uses Kaggle.** Dataset card "Provenance"
   bullet 2 and "Games missing" bullet; model card "Data note" (`publish.py:93-94`).
   *Fix:* both cards render their source lines from `nba/config.py`; regenerated copies are
   committed under `docs/` and a test checks them; pushing to HF is an owner step.
3. **Brief heading date ≠ run date; replay dates unlabelled.** `/brief` heads with
   `brief.date` (2026-09-14) while the run was 2026-09-15 (`brief/page.tsx:67`); the date
   selector lists replay-season dates and live dates alike. *Fix:* heading shows both dates;
   each date is labelled `replay` (on or before the last replayed date) or `live`.
4. **Two model identifiers.** `50a3b2e` (git commit of the training code, in
   `metrics.json.git_sha`) is labelled "model revision" on `/` while `fb427de` (HF commit
   holding the files) is the pinned `MODEL_REVISION`. *Fix:* one identity string everywhere,
   `commit 50a3b2e / HF fb427de`, with both ids mirrored in config and checked by a test.
5. **GitHub topics.** `mlflow`, `xgboost`, `fastapi` are already absent (section 11). Owner
   still to add `dbt`, `duckdb`.
6. **`reports/metrics.json` has no file-level hash.** Confirmed: only `dataset.version =
   hf:<commit>`. *Fix:* `reports/provenance_b20b5601.json` with SHA-256 of every parquet at
   `b20b5601…`, of the model files and `metrics.json` at `fb427de…`, and of the committed
   reports.
7. **Committed replay report vs the HF copy the site reads** are different runs with
   identical numbers (section 6). Recorded in the provenance file.
8. **`agent_pass_rates.json` embeds the Groq organisation id** in error strings.
9. **Freshness counts playoff rows** in the dump as "newest game" (section 4).
