# nba-ai-ml

A batch pipeline that predicts an NBA player's points, rebounds, and assists for a
game from that player's history before the game, scores the predictions against box
scores the next day, and writes a short tool-grounded brief about the results. It runs
on free tiers only (GitHub Actions, Hugging Face, Groq, Vercel), stores every product
as a file, and publishes numbers only from committed reports. Personal, non-commercial
portfolio project. Python 3.11, LightGBM, Parquet, Next.js.

## Pipeline

```
Kaggle dump (CC0 box scores)
   |  nightly ingest, 10:00 UTC (.github/workflows/nightly.yml -> nba/ingest/kaggle_daily.py)
   v
HF dataset  cbratkovics/nba-game-logs   game_logs/<season>.parquet
   |  train on GitHub Actions (train.yml -> nba/models/train.py, features: nba/features/asof.py)
   v
HF model    cbratkovics/nba-stat-predictor   lgbm_{pts,reb,ast}.txt, pinned revision fb427de
   |  nightly slate + residuals (nba/predict/slate.py, nba/predict/residuals.py)
   v
HF dataset  predictions/, residuals/, daily_reports/, replay/
   |  Groq agent brief (nba/agent/loop.py, 7 read-only tools, nba/agent/evals.py)
   v
HF dataset  brief/<date>.json, brief/latest.json
   |  fetch-only pages, revalidated hourly (frontend/)
   v
Next.js on Vercel   /  /predictions  /replay  /brief
```

Every stage reads and writes files; nothing needs a server. The nightly job exits 0
whether or not the agent produced a brief.

## Holdout results

Source: [reports/metrics.json](reports/metrics.json), written by
`python -m nba.models.train` on dataset revision `b20b560`.

Split: train on 2021-22 through 2024-25 (88,257 rows, 2021-10-19 to 2025-04-13);
holdout is all of 2025-26 (22,630 rows, 2025-10-21 to 2026-04-12). Rows are games where
the player logged at least 10 minutes. Nothing from the holdout is used for fitting or
for choosing settings. Each target is scored on the 22,244 holdout rows where both
baselines exist. Baselines are the player's last-10-game mean and season-to-date mean.

| Target | Model MAE | Last-10 MAE | Season MAE | Model RMSE | Model R² |
|---|---:|---:|---:|---:|---:|
| pts | 4.764 | 4.908 | 4.946 | 6.160 | 0.457 |
| reb | 1.942 | 2.009 | 2.008 | 2.527 | 0.402 |
| ast | 1.431 | 1.460 | 1.456 | 1.909 | 0.461 |

Replay equivalence: [reports/replay_2025-26.json](reports/replay_2025-26.json) replays
the nightly slate path over all 164 game dates of 2025-26 with the same model and
dataset revisions. On the same population its MAE differs from metrics.json by
+0.0021 pts, +0.0008 reb, +0.0010 ast, within the 0.05 tolerance. Over every row with
an actual (26,031 of 38,372 slated players) MAE is 4.864 pts, 2.023 reb, 1.403 ast.

## Agent evals

Source: [reports/agent_evals.json](reports/agent_evals.json), replayed from the
committed traces in `tests/traces/` with the provider mocked. Model
`openai/gpt-oss-120b` on Groq.

| Check | Result |
|---|---|
| Grounding: every number in a finding matches its cited tool output within 0.01 | 6 of 6 briefs |
| Golden set: the finding names the day's largest-points-residual player | 5 of 5 dates |
| Pass rates over repeated runs (`agent-eval.yml`) | pending |

## Links

- Live site: <https://nba-ai-ml.vercel.app>
- Dataset: <https://huggingface.co/datasets/cbratkovics/nba-game-logs>
- Model: <https://huggingface.co/cbratkovics/nba-stat-predictor>
- Data reconciliation notes: [docs/reconciliation.md](docs/reconciliation.md)
- Agent design, limits, evals, cost: [docs/agent.md](docs/agent.md)

## What this does not do

- **No live collection from nba.com.** The probe workflow
  (`.github/workflows/probe-nba-api.yml`) timed out on stats.nba.com and got HTTP 403
  from cdn.nba.com when run from a GitHub Actions runner, so the nightly job reads the
  Kaggle dump instead and lags it by however long the dump takes to update.
- **Seven 2024-25 games are missing.** The dump has box scores for 1,223 of that
  season's 1,230 games; the missing ids are listed in docs/reconciliation.md and
  reported by the agent's `list_data_gaps` tool. They are not filled in from anywhere.
- **Offseason behaviour.** When the dump has no schedule file for the next season, or
  the schedule lists no games for the date, the nightly job logs that and exits 0. No
  predictions, residuals, or brief are produced for that date.
- **Model limitations.** One LightGBM regressor per target on 23 rolling and
  situational features. It does not predict minutes or whether a player plays; a
  player with no game in their team's previous ten is not slated. There is no injury,
  lineup, or betting-market input.
- **The agent brief is not deterministic.** At temperature 0 the model's output still
  varies between runs. A single run can drop a finding as ungrounded or miss the golden
  player, and one of the six recorded runs did. Pass rates over repeated runs are the
  measure to read, and they are not published yet.

## Data-source terms

- Backfill and daily updates come from
  [Historical NBA Data and Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)
  by Eoin Moore on Kaggle, released under CC0. The code takes a local path or the
  Kaggle single-file download with your own credentials; none are stored here.
- The dump is derived from NBA.com box scores. This project uses it for non-commercial
  personal study, publishes derived per-season Parquet and prediction files on Hugging
  Face, and does not redistribute the raw CSVs.
- The agent sends Groq only the outputs of its tools, which are aggregates of the
  published files. No credentials or request headers are recorded in the traces.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

LightGBM needs OpenMP at runtime (`brew install libomp` on macOS). Secrets used by the
workflows: `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GROQ_API_KEY`. Repo ids, the
model revision, and thresholds live in `nba/config.py`.

## License

MIT. See [LICENSE](LICENSE).
