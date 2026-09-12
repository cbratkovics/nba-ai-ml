# NBA player-stat prediction pipeline

A small batch pipeline that predicts a player's points, rebounds, and assists for
a game from that player's history before the game. Personal, non-commercial
portfolio project. Python 3.11, LightGBM, Parquet on Hugging Face.

The `frontend/` directory is an earlier Next.js dashboard and is not part of
this pipeline. It is left untouched for now.

## What it does

1. **Backfill** game logs from a Kaggle CC0 dump into a canonical schema, one
   Parquet file per season (2021-22 through 2025-26).
2. **Store** those files in a Hugging Face dataset repo. Parquet is never
   committed to git.
3. **Build features** as of each game: rolling means over the previous 5, 10,
   and 20 games, season-to-date means, games played, days of rest, a
   back-to-back flag, a home flag, and prior means against the opponent. Every
   feature uses only games strictly before the target game; a unit test proves
   this by construction.
4. **Train** one LightGBM regressor per target on seasons through 2024-25 and
   score it on all of 2025-26 against two baselines: the player's last-10-game
   mean and the player's season-to-date mean.
5. **Report** MAE, RMSE, and R² for the model and both baselines to
   `reports/metrics.json`, together with the dataset version, split dates,
   feature list, and git commit.
6. **Publish** the model files and a model card to a Hugging Face model repo.

## Layout

```
nba/
  config.py                 seasons, windows, min minutes, HF repo ids, HF_TOKEN
  schema.py                 canonical game-log schema + validate()
  ingest/kaggle_backfill.py Kaggle dump (v515) -> schema -> Parquet per season, optional --push
  storage/local.py          per-season Parquet read/write, dataset fingerprint
  storage/hf.py             push/pull Parquet and model files to Hugging Face
  features/asof.py          point-in-time features
  models/train.py           LightGBM per target + baselines on the holdout season
  models/evaluate.py        reports/metrics.json and the metrics table
  models/publish.py         model card + upload to the HF model repo
tests/                      schema, leakage, backfill mapping, training smoke test
.github/workflows/ci.yml    ruff + pytest on push and pull request
.github/workflows/probe-nba-api.yml  manual check that stats.nba.com answers a runner
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

LightGBM needs OpenMP at runtime. On macOS with Homebrew: `brew install libomp`.

## Running the pipeline

```bash
# 1. Backfill from the Kaggle dump (download it yourself; no credentials are stored here)
python -m nba.ingest.kaggle_backfill --kaggle-dir /path/to/kaggle --out-dir data/game_logs

# 2. Push to / pull from the Hugging Face dataset repo (needs HF_TOKEN to push)
HF_TOKEN=... python -m nba.storage.hf push-dataset
python -m nba.storage.hf pull-dataset

# 3. Train, evaluate, and write reports/metrics.json
python -m nba.models.train

# 4. Publish models + model card
HF_TOKEN=... python -m nba.models.publish
```

Hugging Face repos: dataset `cbratkovics/nba-game-logs`, model
`cbratkovics/nba-stat-predictor` (both set in `nba/config.py`).

## Evaluation

The holdout is the whole 2025-26 season; nothing from it is used for fitting or
for choosing settings. Rows are limited to games where the player logged at
least 10 minutes (`MIN_MINUTES` in `nba/config.py`), so the model does not
predict minutes or DNPs. Metrics are published only from `reports/metrics.json`,
which is written by `python -m nba.models.train` on a real data pull. That file
does not exist yet, so no numbers are listed here. When it does, the table from
`nba.models.evaluate.metrics_table` will be pasted below verbatim.

## Data provenance

- **Backfill:** [Historical NBA Data and Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)
  by Eoin Moore on Kaggle (version 515), released under CC0. Two files are
  read: `PlayerStatistics.csv` (box scores, streamed and cut to October 2021
  onward) and `TeamHistories.csv` (team abbreviations by season). Download it
  manually; the code takes a local path and stores no Kaggle credentials.
- **Daily updates:** `nba_api`, an unofficial client for `stats.nba.com`
  (not yet wired in this phase; the probe workflow checks whether GitHub
  runners can reach it).
- Both sources are derived from NBA.com box scores. The data is used here for
  non-commercial personal study only, requests are rate-limited politely, and
  the repository does not redistribute the raw data.

## License

MIT. See [LICENSE](LICENSE).
