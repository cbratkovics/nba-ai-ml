---
license: mit
language: en
tags:
  - nba
  - basketball
  - lightgbm
  - tabular-regression
datasets:
  - cbratkovics/nba-game-logs
---

# NBA player-stat predictor

One LightGBM regressor per target (points, rebounds, assists) that predicts a
player's box-score line for a game from that player's history before the game.
Personal, non-commercial portfolio project.

## Dataset

- Repo: `cbratkovics/nba-game-logs`, version `hf:b20b5601de213fa8e704ebffaafd18f182ea68c3`
- Rows are per (player, game); features use only games strictly before the target game.
- Training rows are limited to games where the player logged at least
  10.0 minutes.

## Split

| | Seasons | Dates | Rows |
|---|---|---|---:|
| Train | 2021-22, 2022-23, 2023-24, 2024-25 | 2021-10-19 to 2025-04-13 | 88257 |
| Holdout | 2025-26 | 2025-10-21 to 2026-04-12 | 22630 |

The holdout season is never used for fitting or for choosing settings.

## Metrics on the holdout season

`baseline_last10` is the player's mean over the previous 10 games;
`baseline_season` is the player's season-to-date mean. All three predictors are
scored on the same eligible rows: players who logged at least
10.0 minutes and for whom both baselines are defined. This is the
training-population cohort, not every player-game in the replay.

Holdout season: 2025-26 (2025-10-21 to 2026-04-12)

| Target | Predictor | MAE | RMSE | R² | n |
|---|---|---:|---:|---:|---:|
| pts | model | 4.764 | 6.160 | 0.457 | 22244 |
| pts | baseline_last10 | 4.908 | 6.393 | 0.415 | 22244 |
| pts | baseline_season | 4.946 | 6.467 | 0.401 | 22244 |
| reb | model | 1.942 | 2.527 | 0.402 | 22244 |
| reb | baseline_last10 | 2.009 | 2.644 | 0.346 | 22244 |
| reb | baseline_season | 2.008 | 2.659 | 0.338 | 22244 |
| ast | model | 1.431 | 1.909 | 0.461 | 22244 |
| ast | baseline_last10 | 1.460 | 1.967 | 0.427 | 22244 |
| ast | baseline_season | 1.456 | 1.973 | 0.424 | 22244 |

On this eligible holdout cohort, the model's MAE is about 2–3% lower than the
last-10 baseline (2.9% for points, 3.3% for rebounds, and 2.0% for assists). The
separate nightly-path replay also reports an all-rows population that includes games
under 10 minutes; on that wider population, the last-10 baseline has lower MAE on
all three targets. The model's advantage must not be generalized beyond the eligible
cohort in the table above.

## Features

- `pts_mean_last5`
- `pts_mean_last10`
- `pts_mean_last20`
- `reb_mean_last5`
- `reb_mean_last10`
- `reb_mean_last20`
- `ast_mean_last5`
- `ast_mean_last10`
- `ast_mean_last20`
- `minutes_mean_last5`
- `minutes_mean_last10`
- `minutes_mean_last20`
- `pts_mean_season`
- `reb_mean_season`
- `ast_mean_season`
- `minutes_mean_season`
- `games_played_season`
- `days_rest`
- `back_to_back`
- `home`
- `pts_mean_vs_opp`
- `reb_mean_vs_opp`
- `ast_mean_vs_opp`

## Files

- `lgbm_pts.txt` (LightGBM text model for `pts`)
- `lgbm_reb.txt` (LightGBM text model for `reb`)
- `lgbm_ast.txt` (LightGBM text model for `ast`)
- `metrics.json` (the report these numbers come from)

## Known limitations

- Predicts only for players who play; it does not predict minutes or DNPs. The model
  metrics above use only games with at least 10.0 minutes played
  and both baselines available; the wider replay does not show a model advantage.
- No injury, lineup, betting-line, or opponent-strength inputs.
- Regular-season games only.
- Game-to-game variance in box-score stats is high; compare against the
  baselines above rather than reading the absolute error alone.

## Data sources

- **Historical backfill:** Eoin Moore, *Historical NBA Data and Player Box Scores*, Kaggle (`eoinamoore/historical-nba-data-and-player-box-scores`), version 515, CC0 1.0. Only `PlayerStatistics.csv` and `TeamHistories.csv` are used; rows carry `source = kaggle_v515`.
- **Daily updates:** the same Kaggle dataset, re-downloaded by the nightly GitHub Actions job. Rows within 7 days of the newest stored game are reconciled against the stored rows; new or changed rows carry `source = kaggle_daily`. There is no live collection from NBA.com: stats.nba.com is not reachable from GitHub Actions runners.

Both are derived from NBA.com box scores. NBA data is used here for non-commercial
personal study only.

## Identity

Trained at git commit `50a3b2e33b443d1db19274cea27467072ebfb3f8` on 2026-09-12T02:00:06+00:00. Pinned by the pipeline as `commit 50a3b2e / HF fb427de` (`nba/config.py`).
