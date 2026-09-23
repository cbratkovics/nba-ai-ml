# Citable facts, by phase

What a resume bullet or an interview answer may state about this repository, one or two
facts per phase, each with the artifact key or ADR it lives in. Facts only. Anything not on
this list is not a claim this repository backs.

| Phase | Fact | Where it lives |
|---|---|---|
| Audit | The earlier site's headline hid a population where the baseline wins; the audit found it, and every page now names its population. | `AUDIT.md` §15 item 1; ADR-0002 |
| Audit | The model has one identity everywhere, `commit 50a3b2e / HF fb427de`, mirrored in config, the site and the dbt vars and checked by a test. | ADR-0005; `tests/test_identity.py` |
| Data | 130,414 game-log rows over five seasons; seven 2024-25 games have no box scores (1,223 of 1,230) and are declared, not filled. | `docs/reconciliation.md` finding 7; `dbt/seeds/known_missing_games.csv`; ADR-0012 |
| Data | Five box scores above the plausibility thresholds were each verified against nba.com and seeded with the verification date. | `dbt/seeds/known_stat_exceptions.csv`; ADR-0013 |
| Features | One feature module (`asof_v1`); the warehouse's season-to-date mean equals it on all 130,414 rows with zero difference. | `nba/features/asof.py` `FEATURE_VERSION`; ADR-0015; `tests/test_dbt_gold.py` |
| Evaluation | Points MAE 4.764 vs 4.908 for the last-10 mean on 22,244 holdout rows of a full held-out season. | `reports/metrics.json` `metrics.pts.{model,baseline_last10}.mae` |
| Evaluation | The 164-date replay of the nightly path reproduces the holdout MAE within +0.0021 points on the same population. | `reports/replay_2025-26.json` `diff_vs_metrics_json.pts` |
| Evaluation | On all replayed rows (26,031) the last-10 mean is the better predictor on every target; published next to the headline. | `reports/replay_all_rows_2025-26.json`; `frontend/lib/all_rows_baseline.json`; ADR-0002 |
| Warehouse | The gold marts recompute the replay reports to 1e-15 and `metrics.json` within 0.0021; the latter is a population-restricted comparison because 169 holdout rows are never slated. | `dbt/tests/gold/assert_holdout_metrics_reconcile_*.sql`; ADR-0011 |
| Warehouse | `dim_player_asof` derives 2,003 team stints for 1,027 players (976 team changes) from the game logs alone, and the replay's roster rule reproduces from it. | ADR-0010; `assert_replay_roster_rule_reproduces_from_asof` |
| Warehouse | DuckDB parallel aggregation is not deterministic at 1e-15; a fixed thread count makes a build reproducible. | ADR-0011; `dbt/profiles.yml` `NBA_DUCKDB_THREADS` |
| Decisions | Line-free directional calls beat both causal baselines on the training population (points: 0.667 vs 0.592 for the season-mean sign, 6,680 resolved calls) and lose to the season-mean sign on all rows (0.547 vs 0.566). | `reports/policy_2025-26.json` `populations.<p>.targets.pts.{hit_rate,baselines.season_mean_sign.hit_rate,n_resolved}`; ADR-0015 |
| Decisions | Thresholds are the largest grid value that still calls a quarter of the rows, chosen in-sample; the whole coverage curve is published and reconciled point for point (230 points) by SQL and pandas. | ADR-0006, ADR-0015; `mart_policy_sweep`; `assert_policy_sweep_reconciles_to_policy_report` |
| Drift | A season-long reference fires on normal data at every season position (the season counter reached PSI 6.4 during the Cup); the reference is aligned by season day and built from seasons with an earlier season in the data. | ADR-0017 |
| Drift | Retrospectively calibrated on 164 replay dates (157 scoreable): PSI ≥ 0.15 on three features produced no calibration-set alerts (0.05 would have fired on 69 dates); three injected shifts HOLD, one WARNs. Prospective reliability is untested. | `reports/drift_calibration_2025-26.json` `chosen`, `false_positives_total`, `sensitivity`; ADR-0018 |
| Drift | HOLD opens a labelled issue and never blocks the slate; fourteen no-schedule nights open a WARN issue. | ADR-0018; `nba/drift/policy.py`; `tests/test_drift.py` |
| Agent | Seven read-only tools; every number in a brief must match its cited tool output within 0.01 or the finding is dropped. | `nba/agent/evals.py`; `docs/agent.md` |
| Agent | The tools read the exported gold marts where those are the better source, with file fallbacks that compute the same numbers; return shapes are versioned and the version-1 keys are frozen in a test. | ADR-0019; `tests/test_agent_tools.py` |
| Agent | Golden set version 2: five dates, three facts each (player, decision hit rate, drift status); the committed traces replay 6 of 6 grounded and 5 of 5 golden. | `reports/agent_evals.json` `golden_facts`; ADR-0020 |
| Agent | The traces the grounding claim rested on were untracked until 2026-09-16; they are committed now and CI replays them, and the test fails on an empty folder. | ADR-0021; `AUDIT.md` §15 item 10 |
| Operations | Nightly job: ingest, residuals, brief, slate, decisions, drift, push, then a gated warehouse build with a local fallback; off-season runs exit 0 with one line each. | `.github/workflows/nightly.yml`; ADR-0007 |
| Operations | The spreader completed 25 of 25 live briefs over five Actions runs under golden set version 2; every measured brief passed grounding and golden checks. | `reports/agent_pass_rates.json` `overall` and per-date `measured`; `agent-eval.yml` |
| Operations | MotherDuck, pushed gold exports, and nightly decision/drift publication remain unverified by committed run evidence; workflow capability is not deployment proof. | `README.md` deployment boundary; ADR-0007 |
