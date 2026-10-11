# Resume bullets and interview stories

Built only from the facts in [STORIES.md](STORIES.md); every number names the artifact it is read
from. Nothing here may be stated that STORIES does not back. Bullets are exactly two lines each.

## Bullets

Built a dbt warehouse (DuckDB locally, MotherDuck weekly) over a frozen LightGBM pipeline: 41 models, 93 data tests,
20 enforced contracts, 11 exposures; every mart reconciles to a committed evaluation artifact or the publish blocks.

Proved the model's 23 as-of features in SQL against the Python module row for row over 130,414 game logs (largest
difference 2.1e-14, tolerance set from it), and pre-registered an out-of-sample evaluation against market lines.

Sources: bullet 1 — README `dbt build` counts (rendered from `dbt/target/manifest.json` by
`scripts/render_dbt_counts.py`); reconciliation tolerances in `dbt/dbt_project.yml` (`tol_replay`,
`tol_metrics_json`, `tol_policy`) and `dbt/tests/gold/assert_*_reconcile_*.sql`; severity policy ADR-0026.
Bullet 2 — `reports/feature_parity_asof_v1.json` (`features.<name>.max_abs_diff`, `rows_joined`,
`tolerance`), `tests/test_feature_parity.py`, ADR-0032; `nba/ledger/rule.py` (`registered_on`,
`first_live_date`, hash pinned as `ledger_rule_hash` in `dbt/dbt_project.yml`), ADR-0030.

## Stories

### 1. The export that had been stale for three weeks

**Situation.** The README said the gold marts were exported to Hugging Face after every weekly
refresh. The read-only audit compared that claim with the run steps instead of the workflow file.
**What I found.** The publish step's condition, `inputs.publish != false`, is false on scheduled
runs, where `inputs` is empty and both sides coerce to 0; `gold/` had been pushed once, on
2026-09-16, and skipped for three weeks while four green runs said otherwise.
**What I did.** Fixed the condition (`github.event_name == 'schedule' || inputs.publish`), and
changed what "ran" is allowed to mean: every run claim names a run id in `reports/runs.json`,
`tests/test_runs.py` fails when a cited id is missing, and the README's dbt counts are rendered
from the manifest (the hand-typed ones said 24 models and 53 tests against 31 and 70).
**Evidence.** `reports/runs.json`; ADR-0022; `scripts/render_dbt_counts.py`; STORIES "Operations".
**What I would say if asked what it taught me.** A green workflow is not evidence that a step
ran; the artifact the step was supposed to produce is.

### 2. A tolerance that was measured, not guessed

**Situation.** The model's 23 features existed only in pandas. The warehouse needed them in SQL
for lineage, but a second implementation drifts unless something proves it does not; the first
audit of this repository had found four feature implementations that disagreed.
**What I did.** Kept the Python module as the single reference and the thing that serves. The
loader runs it over the same game logs the warehouse loads and writes its output next to them;
`int_player_game_features` recomputes the features with window functions; a singular test
full-outer-joins the two on every build and fails on a row on one side only, a null on one side
only, any integer difference, or a double difference above a tolerance.
**The number.** Over 130,414 rows the largest difference was 0.0 on 19 features and 2.13e-14 on
the four minutes means (summation order on fractional minutes). The tolerance was set from that,
1e-12, and a test holds the var to the report; at a tolerance of 0 the test fails on 42,828 rows,
and perturbing one value by 0.01 makes it fail on exactly that row.
**Evidence.** `reports/feature_parity_asof_v1.json`;
`dbt/tests/intermediate/assert_int_features_match_feature_module.sql`;
`tests/test_feature_parity.py`; ADR-0032; STORIES "Warehouse".
**What I would say if asked.** The audit had proposed 1e-9 before anything was measured. The
measured value was three orders of magnitude tighter, and the write-up says which.

### 3. Pre-registering the evaluation before the season

**Situation.** The owner wanted the frozen model evaluated against market lines over 2026-27,
out of sample, on a free odds API with 500 credits a month, with raw lines kept private and no
chance of the result being shaped after the fact.
**What I did.** Wrote the rule first and froze it: eligibility (one prior game this season, a
last-10 minutes mean of at least 15, inputs at most 10 days old) chosen on the 2025-26 replay
(proxy precision 0.957, recall 0.863), edge thresholds reused from the already-committed policy,
the whole rule hashed and the hash pinned in the dbt vars so every ledger row names the rule it
was scored under. Capture is budgeted and bias-guarded: a credit guard that fails closed at 425
of 500 with two independent readings and no retries, and a decision record written before any
line is fetched, so a captured night cannot be dropped later. Settlement is computed, never
stored, so a late box score settles pending entries on the next build. The reading rule was
written in advance too: "too early" under 100 settled entries, a verdict only when the interval
excludes break-even, and the arithmetic that says one season of points lines is not enough.
**Evidence.** `nba/ledger/rule.py`; `reports/ledger_rule_evidence_2025-26.json`;
`nba/lines/guard.py`; `nba/lines/capture.py`; `dbt/models/gold/fct_paper_bet.sql`;
ADR-0030, ADR-0031; STORIES "Ledger".
**What I would say if asked.** The honest result of this design may be "too early" at the end
of the season. The design makes that answer trustworthy, which is the point.
