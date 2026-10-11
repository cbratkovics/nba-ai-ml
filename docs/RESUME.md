# Resume bullets and interview stories

Built only from the facts in [STORIES.md](STORIES.md); every number names the artifact it is read
from. Nothing here may be stated that STORIES does not back. Bullets are two lines each, under
about 200 characters.

## Bullets

Built a 41-model dbt warehouse on DuckDB and MotherDuck with enforced contracts and exposures;
proved 23 SQL features match the Python feature module row for row across 130,414 player-games.

Pre-registered an out-of-sample evaluation of the frozen LightGBM model against market lines;
on the 2025–26 holdout, MAE improved 2–3% for players with 10+ minutes and the baseline won on all rows.

Sources: bullet 1 — model count, contracts and exposures from the README `dbt build` line
(rendered from `dbt/target/manifest.json` by `scripts/render_dbt_counts.py`; 41 models, 20 with
enforced contracts, 11 exposures in `dbt/models/exposures.yml`); the parity proof from
`reports/feature_parity_asof_v1.json` (`rows_joined` = 130,414, `features.<name>.max_abs_diff`,
`tolerance`), `dbt/tests/intermediate/assert_int_features_match_feature_module.sql`,
`tests/test_feature_parity.py`, ADR-0032. Bullet 2 — the rule and its registration from
`nba/ledger/rule.py` (`registered_on`, `first_live_date`; hash pinned as `ledger_rule_hash` in
`dbt/dbt_project.yml`), ADR-0030; the holdout numbers from `reports/metrics.json`
(`metrics.<target>.{model,baseline_last10}.mae`, `split.min_minutes` = 10, population n = 22,244);
the all-rows result from `reports/replay_all_rows_2025-26.json` (row-weighted `days[].model`
vs `days[].baseline_last10`), ADR-0002.

## Stories

### 1. The export that had been stale for three weeks

**Situation.** My README said the gold marts were exported to Hugging Face after every weekly
refresh. I commissioned a read-only audit of the repository before changing anything, and asked
that it compare claims with run evidence rather than with the workflow files.
**What the audit found.** The publish step's condition, `inputs.publish != false`, is false on
scheduled runs, where `inputs` is empty and both sides coerce to 0. `gold/` had been pushed once,
on 2026-09-16, and skipped for three weeks while four green runs said otherwise.
**What I did.** I had the condition fixed (`github.event_name == 'schedule' || inputs.publish`)
and changed what "ran" is allowed to mean in this repository: every run claim names a run id in
`reports/runs.json`, `tests/test_runs.py` fails when a cited id is missing, and the README's dbt
counts are rendered from the manifest (the hand-typed ones said 24 models and 53 tests against
31 and 70). I reviewed each change and dispatched the first corrected refresh myself.
**Evidence.** `reports/runs.json`; ADR-0022; `scripts/render_dbt_counts.py`; STORIES "Operations".
**What it taught me.** A green workflow is not evidence that a step ran; the artifact the step
was supposed to produce is.

### 2. A tolerance that was measured, not guessed

**Situation.** The model's 23 features existed only in pandas. I wanted them in the warehouse
for lineage, but a second implementation drifts unless something proves it does not, and I was
not willing to move the serving path off the module the published model was trained on.
**What I decided.** The Python module stays the single reference and the thing that serves. The
loader runs it over the same game logs the warehouse loads and writes its output next to them;
`int_player_game_features` recomputes the features with window functions; a singular test
full-outer-joins the two on every build and fails on a row on one side only, a null on one side
only, any integer difference, or a double difference above a tolerance. I asked for the largest
observed difference per feature to be reported and the tolerance set from it, rather than
accepting the audit's proposed 1e-9.
**The number.** Over 130,414 rows the largest difference was 0.0 on 19 features and 2.13e-14 on
the four minutes means (summation order on fractional minutes). The tolerance was set from that,
1e-12, and a test holds the dbt var to the report; at a tolerance of 0 the test fails on 42,828
rows, and perturbing one value by 0.01 makes it fail on exactly that row.
**Evidence.** `reports/feature_parity_asof_v1.json`;
`dbt/tests/intermediate/assert_int_features_match_feature_module.sql`;
`tests/test_feature_parity.py`; ADR-0032; STORIES "Warehouse".
**What it taught me.** The proposed tolerance was three orders of magnitude looser than the
measured one. Measure first, then set the bar, and write down which it was.

### 3. Pre-registering the evaluation before the season

**Situation.** I wanted the frozen model evaluated against market lines over 2026-27, out of
sample, on a free odds API with 500 credits a month, with raw lines kept private and no way for
the result to be shaped after the fact.
**What I did.** I had the rule written first and reviewed it before it was frozen: eligibility
(one prior game this season, a last-10 minutes mean of at least 15, inputs at most 10 days old)
chosen on the 2025-26 replay (proxy precision 0.957, recall 0.863), edge thresholds reused from
the already-committed policy, the whole rule hashed and the hash pinned in the dbt vars so every
ledger row names the rule it was scored under. I approved it with five amendments, among them
that settlement waits for a box score and voids only a player with no row in it, and that
results are reported by side because the model is a conditional mean. Capture is budgeted and
bias-guarded: a credit guard that fails closed at 425 of 500 with two independent readings and
no retries, and a decision record written before any line is fetched, so a captured night cannot
be dropped later. Settlement is computed, never stored, so a late box score settles pending
entries on the next build. The reading rule was written in advance too: "too early" under 100
settled entries, and a verdict only when the interval excludes break-even.
**Evidence.** `nba/ledger/rule.py`; `reports/ledger_rule_evidence_2025-26.json`;
`nba/lines/guard.py`; `nba/lines/capture.py`; `dbt/models/gold/fct_paper_bet.sql`;
ADR-0030, ADR-0031; STORIES "Ledger".
**What it taught me.** The honest result of this design may be "too early" at the end of the
season. The design is what makes that answer trustworthy.
