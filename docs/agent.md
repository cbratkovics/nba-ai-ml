# The analyst agent

A tool-using agent that reads the pipeline's published files after each nightly run
and writes a short, structured brief. It runs on Groq's free tier, is bounded in every
dimension, and is checked by deterministic evals before anything it says is published.

## What it can do

- Read, through seven read-only tools, the ingest report for a run date, how stale the
  upstream dump is, the largest residuals for a date, rolling accuracy versus the
  last-10-game baseline, a player's recent game lines, a team's recent games, and the
  known data gaps.
- Write a brief of at most 120 words plus at most five findings, each with a severity,
  a short kind label, the tool call it relies on (name, arguments, returned values), and
  one or two sentences of text.
- Fail quietly: on any provider error or limit breach it writes a brief with status
  `agent_unavailable` and the nightly job continues.

## What it cannot do

- It has no web access, no NBA API access, and no write access to anything but its own
  brief. It cannot fetch news, injury reports, lineups, or odds, and the system prompt
  forbids speculating about them.
- It cannot change the model, retrain, alter data, or trigger jobs.
- It cannot publish a number that its evidence does not contain: the grounding check
  drops any finding whose text cites a number absent from `evidence.values`
  (tolerance 0.01) and marks the brief `ungrounded`.
- It is not consulted for predictions. Predictions come from the LightGBM models; the
  agent only reports on their inputs and outcomes.

## Tools (`nba/agent/tools.py`)

| Tool | Reads | Returns |
|---|---|---|
| `get_daily_report(date)` (v2) | `daily_reports/<date>.json`; `gold/mart_drift.parquet` else `drift/<date>.json` else the calibration's per-date row; `gold/mart_restatement_lag.parquet` | ingest window, new/changed/unchanged counts, up to 5 changed examples, seasons written, push status, revisions; `drift` (latest verdict on or before the date: status, position, flagged features, largest PSI, no-schedule streak, source); `restatement` (observed lag against the lookback) |
| `get_upstream_freshness()` (v2) | stored game logs, the downloaded dump if present under the regular-season rules | newest stored game date, newest regular-season dump game date (and the raw maximum of any type, with the rows the rules excluded), days stale versus the run date |
| `get_residuals(date, top_n=5)` | `residuals/<date>.parquet`, else `replay/2025-26/residuals/<date>.parquet` | MAE per target, top-n absolute residuals per target with predicted/actual/minutes, counts of did-not-play and game-not-ingested |
| `get_rolling_metrics(days=30)` (v2) | `gold/mart_daily_metrics.parquet`, else residual files in the window, else `replay/2025-26/daily_mae.json`; `decisions` from `gold/fct_decision_policy.parquet`, else the residual files with `reports/policy_<season>.json` | MAE per target for the model and the last-10 baseline on the same rows, window bounds, count of dates covered; `decisions`: the training-population policy's called rows, resolved rows and hit rate to date per target |
| `get_player_recent(player_id, n=10)` | stored game logs | last n lines on or before the run date, plus means |
| `get_team_context(team, date)` | stored game logs | the team's last ten games before the date with player counts, distinct players, and whether it played on the date |
| `list_data_gaps()` | `KNOWN_MISSING_GAMES`, stored game logs | the missing-game count, the seven missing 2024-25 games, seasons below 1,230 games |

Every tool returns a JSON-serializable dict; bad arguments or failures come back as
`{"error": ...}` so a mistake never ends the loop. Each tool is unit-tested against
fixtures (`tests/test_agent_tools.py`).

### Marts and fallbacks (ADR-0019)

The marts arrive through `pull_products` (the `gold/` folder the nightly warehouse step
pushes after its build); until the first warehouse build has run in Actions the tools use
the fallbacks, which compute the same numbers (the decisions fallback was checked equal
to the mart on two golden dates, every count and rate). A tool whose return shape grew
carries `tool_version: 2` and keeps every version-1 key; the test suite freezes the
version-1 keys per tool, so a shape change is a versioned, additive change with a test.

| Tool | Version | Mart when exported | Fallback |
|---|---|---|---|
| `get_daily_report` | 2: adds `drift`, `restatement` | `mart_drift` (latest run on or before the date: status, flagged features by name, streak), `mart_restatement_lag` | `drift/<date>.json` products, then the calibration's per-date row for replay dates; restatement unavailable |
| `get_upstream_freshness` | 2: adds `dump_max_game_date_any_type`, `dump_rows_excluded_by_rules`, `rules_applied` | (stored logs) | the dump under the backfill's regular-season rules, so playoff rows never set the newest date |
| `get_rolling_metrics` | 2: adds `decisions` | `mart_daily_metrics` (row-weighted over the window, population `all`), `fct_decision_policy` (training-population calls and hit rate to date) | residual files, then the replay daily file; decisions from the residual files with `reports/policy_<season>.json` through the same rule |
| the other four | 1 | | unchanged |

## Model and limits (`nba/agent/loop.py`)

**Model: `openai/gpt-oss-120b` on Groq**, pinned in `nba/config.py`. Why this one: the
specification asked for the current production tool-calling Llama model, but the Groq
catalogue queried on 2026-09-12 for this key listed no Llama chat model at all (only
`allam-2-7b`, `groq/compound`, `groq/compound-mini`, `openai/gpt-oss-120b`,
`openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.6-27b`,
`qwen/qwen3.8-27b`, plus prompt-guard, TTS, and Whisper models). A one-tool probe
confirmed native tool calls (`finish_reason=tool_calls`, correct arguments) on
`openai/gpt-oss-120b` (0.53 s), `qwen/qwen3.8-27b` (0.24 s), and `openai/gpt-oss-20b`
(0.27 s). The 120b model was chosen as the largest with verified tool calling and a
131k context; `openai/gpt-oss-20b` is the documented fallback if the free tier
rate-limits it. When the catalogue changes, re-run the probe and update the pin and this
paragraph.

Before the model's first turn the loop runs the five calls every brief needs (daily
report for the run date, upstream freshness, residuals for the brief date, 30-day
rolling metrics, data gaps) and feeds them in as tool results. This is deterministic
code, not a model decision: the first live runs showed the reasoning model requesting one
tool per turn and spending 25 to 40 s of reasoning per turn, which blew the 60 s wall
clock. The prefetched calls count against the tool budget and are recorded under
`prefetch` in the trace. The model may spend the remaining three calls on
`get_player_recent` or `get_team_context`. Reasoning effort is pinned to `low`.

| Limit | Value |
|---|---|
| Tool calls per brief | 8 including the 5 prefetched (the ninth is refused and the model is told to answer) |
| Model turns after the last tool call | 3 (then `agent_unavailable`) |
| Wall clock | 60 s |
| Temperature | 0 |
| Findings | 5 |
| Summary | 120 words |
| Trace size | 100 KB (request snapshots are dropped first) |
| Provider errors | one retry on the fallback model after a 429; anything else ends the brief as `agent_unavailable` |

Rules in the system prompt: report only tool output; cite the tool call for every
number; say "no evidence" instead of guessing; never speculate about injuries or
lineups; at most five findings; keep `evidence.values` a small flat object of the cited
numbers; round to two decimals; name the player in a residual finding; always include one
`decision_policy` finding (hit rate and resolved calls to date) and one `drift` finding
(status word, flagged features, streak); answer as plain JSON content, never as a tool
call.

Two provider quirks are handled in code. gpt-oss models sometimes emit the final JSON
as a call to a tool named `json`; the provider rejects that with a 400 whose body
carries the generated text, and the loop parses the brief out of it. Free-tier quotas
are per model per day; a 429 on the pinned model retries once on the fallback and the
brief records which model answered.

## Output

`brief/<date>.json`:

```json
{"date": "...", "run_date": "...", "status": "ok|ungrounded|agent_unavailable",
 "summary": "...", "findings": [{"kind": "...", "severity": "info|warning|critical",
 "evidence": {"tool": "...", "args": {...}, "values": {...}}, "text": "..."}],
 "tool_calls_made": 0, "model_id": "...", "latency_ms": 0, "generated_at": "..."}
```

`brief/<date>.trace.json` holds every request message list, response, tool call and
tool result. No headers or keys are recorded; the writer redacts anything matching
`gsk_...` and the committed-trace test refuses any trace containing `gsk_`.
`brief/index.json` lists the available dates and `brief/latest.json` mirrors the newest.
All of these are pushed to the Hugging Face dataset repo with the other products and
rendered at `/brief` on the site.

## Evals (`nba/agent/evals.py`)

Two deterministic checks, no model involved:

1. **Grounding.** Every number in a finding's text must match a numeric leaf of that
   finding's `evidence.values`, or of `evidence.args`, within 0.01. Counting the cited
   call's arguments is a deliberate widening of "must appear in evidence.values": a
   finding that says "over 30 days" and cites `get_rolling_metrics(days=30)` is grounded
   by that call. Skipped as labels rather than measurements: ISO dates, ten-digit game
   ids, and numbers glued to a word by a hyphen ("30-day", "2024-25"); signs are ignored
   so "32 days ahead" matches `days_stale = -32`. Failing findings are dropped from the
   published brief, kept under `dropped_findings`, and the brief is marked `ungrounded`.
2. **Golden set (version 2).** Five 2025-26 replay dates, each with the player who had the
   single largest points residual that day, plus one decision fact (the policy's pts hit
   rate to date on the training population, cited from `get_rolling_metrics`) and one drift
   fact (the status word cited from `get_daily_report`; off-season dates carry the
   `insufficient` / streak line), all computed by the tools themselves
   (`python -m nba.agent.golden`, `reports/agent_golden.json`). A date passes only when a
   finding names the player, a finding carries the hit rate (within 0.01, or as a
   percentage) and a finding names the drift status. Results are in
   `reports/agent_evals.json` with each fact counted separately.

CI replays the committed traces (`tests/traces/<date>.trace.json`) through the loop
with the provider mocked, so the checks run without network or keys; the live run
happens on demand through `agent.yml`.

### Results

Committed traces (`tests/traces/`, re-recorded locally on 2026-09-16 under the version-2
golden set and the Phase 4 tools, run date = the day after each brief date), replayed
offline by `python -m nba.agent.evals`:

| Date | Status | Grounded | Golden player | Golden (player, decision, drift) | Live latency |
|---|---|---|---|---|---|
| 2025-10-23 | ok | 5/5 | Aaron Gordon | pass | 3.1 s |
| 2025-12-03 | ok | 3/3 | Giannis Antetokounmpo | pass | 1.6 s |
| 2026-01-14 | ok | 5/5 | Brice Sensabaugh | pass | 25.8 s |
| 2026-03-10 | ok | 3/3 | Bam Adebayo | pass | 7.9 s |
| 2026-04-03 | ok | 4/4 | Cooper Flagg | pass | 3.9 s |
| 2026-04-12 | ok | 3/3 | (not a golden date) |  | 5.8 s |

Every brief used `openai/gpt-oss-120b` with 5 tool calls (the prefetch) and a single
model turn; three to five findings each, the three required kinds (`residual_outlier`,
`decision_policy`, `drift`) first. The 2026-09-12 traces recorded under the version-1
golden set are superseded; their results (6 of 6 grounded, 5 of 5 golden by player only)
are in the git history.

What it took to get there, so the numbers above are not read as a first try:

- Dispatch 1 (before the prefetch existed): one-tool-per-turn behaviour, 65 to 70 s,
  `agent_unavailable` on the wall clock.
- Dispatch 2: 4 of 6 ok; 2 dates emitted the brief as a `json` pseudo tool call and
  ended `agent_unavailable` (now recovered in code).
- Dispatch 3: all 6 hit the 200k tokens-per-day cap on the pinned model after local
  smoke runs (now falls back to the second model).
- Dispatch 4: 2 ok, 3 `ungrounded` because the text named the 30-day window without
  copying it into values (now grounded through `evidence.args`), 1 `agent_unavailable`
  because the model copied whole residual lists into evidence, broke its own JSON, and
  the second turn hit the 8k tokens-per-minute cap (evidence is now required to be small
  and flat).
- Dispatch 5: 5 ok, 1 `ungrounded` (2025-12-03: the residual finding named five players
  but only two of their residuals were in values; the finding was dropped, so the golden
  check failed 4/5). A second run of that date passed, and that trace is the committed
  one. Read the golden result as "passes when the residual finding survives grounding",
  not as a guarantee: at temperature 0 the model's output still varies between runs.

## Reliability

`agent-eval.yml` runs every golden date N times against the provider with no fallback
model (`python -m nba.agent.pass_rates --no-fallback`), applies the nightly grounding
check, and writes `reports/agent_pass_rates.json`. A 429 on the pinned model is recorded
as `rate_limited` and fails both checks, so a capped run reads as incomplete rather than
as a lower pass rate.

The first attempt on 2026-09-12 (GitHub Actions run `34701742530`) completed 10 of
25 requested briefs before the provider's daily limit. It produced 9 grounding passes
and 10 golden passes on the completed briefs; the other 15 were recorded as
`rate_limited`, not silently answered by another model. That incomplete attempt is useful
failure history, but it is not the current repeated-run result.

**Current version-2 result.** The spreader subsequently measured one golden date per
Actions run. The committed report records the runner and run id on every date:

| Date | Player | Actions run | Completed | Grounding | Golden | Rate limited |
|---|---|---:|---:|---:|---:|---:|
| 2025-10-23 | Aaron Gordon | 35073930571 | 5 | 5/5 | 5/5 | 0 |
| 2025-12-03 | Giannis Antetokounmpo | 35200296421 | 5 | 5/5 | 5/5 | 0 |
| 2026-01-14 | Brice Sensabaugh | 35322849219 | 5 | 5/5 | 5/5 | 0 |
| 2026-03-10 | Bam Adebayo | 35430694570 | 5 | 5/5 | 5/5 | 0 |
| 2026-04-03 | Cooper Flagg | 35499665725 | 5 | 5/5 | 5/5 | 0 |
| **Overall** | | | **25/25** | **25/25** | **25/25** | **0** |

These are selected historical dates and model output can still vary; 25 passes do not
guarantee that a future nightly brief will pass.

**Spreading the measurement across days.** `agent-eval.yml` measures only golden dates that are
missing or rate-limited in the committed report (`--only-incomplete --max-dates 1` by
default, or an explicit `dates` input), merges the new per-date results into
`reports/agent_pass_rates.json` (each date records the Actions run id that measured it),
recomputes the overall block, rewrites the README row between its `pass-rates` markers
from the report, and commits both files back to the branch it ran on. A daily schedule
at 03:00 UTC (seven hours before the nightly brief) does one date per day until every
date is complete, then exits before installing anything. One date is 5 briefs, about
25k tokens. `overall.complete` is true only when all five dates have five unlimited runs
under the current golden set; the README row is rendered from that report and checked by
`tests/test_agent_pass_rates.py`.

**Status wording (the README row).** Every date entry records the golden-set version it
was measured under. A date measured under an older version is incomplete for the
spreader, which re-measures it, and its runs are left out of the overall block; the row
then reads "incomplete: N of 25 briefs completed under the current golden set (R
rate-limited; S dates measured under an older golden set, not counted)". The current
report is complete under golden set version 2, so the README instead reports 25 of 25.

**The single-run failure mode, plainly.** At temperature 0 the model's output still
varies between runs. The failure seen in every set of runs so far is the same one: a
residual finding names two or more players and their residuals, but `evidence.values`
carries only some of those numbers, so the grounding check drops the whole finding.
The brief is then marked `ungrounded`, and if the dropped finding was the one naming
the golden player, the golden check fails too. It happened on 1 of 10 briefs in this
run (2025-10-23, run 2, one finding dropped, golden still passed because a kept
finding named Aaron Gordon) and on 1 of 7 dispatch briefs earlier the same day
(2025-12-03, where the golden check failed). A single nightly brief can therefore
arrive with a finding missing, and the site shows that as status `ungrounded` with the
dropped finding listed. Nothing ungrounded is published as a finding.

## Cost

Groq free tier; no paid plan. Limits observed on 2026-09-12 for this key and
`openai/gpt-oss-120b`: 200,000 tokens per day, 8,000 tokens per minute, 1,000 requests
per day, each per model. One nightly brief is normally a single chat completion of about
3,200 prompt tokens and 600 to 800 completion tokens (roughly 4,000 to 5,000 tokens
including reasoning), and at most 4 completions if the model spends its three remaining
tool calls (1 + 3 tool turns, then up to 3 answer turns, but the wall clock ends it first).
That is about 2 percent of the daily token cap per night, so one nightly brief plus an
occasional manual dispatch stays free. A pass-rate run of 25 briefs is about half the
cap by itself and cannot share a day with other dispatches (see Reliability). The per-minute cap is the tight one: a second
model turn in the same minute can exceed it, which is why the loop aims for a single
turn and falls back to the second model on 429. Nothing else in the pipeline calls a
paid or metered API.
