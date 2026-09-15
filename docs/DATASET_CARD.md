---
license: cc0-1.0
pretty_name: NBA Player Game Logs (2021-22 to 2025-26)
task_categories:
  - tabular-regression
tags:
  - nba
  - basketball
  - sports
  - time-series
size_categories:
  - 100K<n<1M
---

# NBA Player Game Logs

One row per player per regular-season game, 2021-22 through 2025-26, in a
canonical schema used by [cbratkovics/nba-ai-ml](https://github.com/cbratkovics/nba-ai-ml)
for point-in-time feature engineering and batch prediction of points, rebounds, and assists.

## Provenance

- **Historical backfill:** Eoin Moore, *Historical NBA Data and Player Box Scores*, Kaggle (`eoinamoore/historical-nba-data-and-player-box-scores`), version 515, CC0 1.0. Only `PlayerStatistics.csv` and `TeamHistories.csv` are used; rows carry `source = kaggle_v515`.
- **Daily updates:** the same Kaggle dataset, re-downloaded by the nightly GitHub Actions job. Rows within 7 days of the newest stored game are reconciled against the stored rows; new or changed rows carry `source = kaggle_daily`. There is no live collection from NBA.com: stats.nba.com is not reachable from GitHub Actions runners.
- The `source` column records which pipeline produced each row.

Both sources are derived from NBA.com. This dataset is published for
**non-commercial, personal portfolio use**; the CC0 label covers the upstream
Kaggle packaging, not the NBA's underlying terms.

## Schema

One row per `(player_id, game_id)`. Columns: `game_id`, `game_date`, `season`,
`player_id`, `player_name`, `team`, `opponent`, `home`, `minutes`, `pts`, `reb`, `ast`,
`fgm`, `fga`, `fg3m`, `fg3a`, `ftm`, `fta`, `oreb`, `dreb`, `stl`, `blk`, `tov`, `pf`,
`plus_minus`, `source`. Team columns are abbreviations. Regular season only.

## Files

`game_logs/game_logs_YYYY-YY.parquet`, one per season.

| Season | Rows | Games | First game | Last game |
|---|---:|---:|---|---|
| 2021-22 | 25,826 | 1,230 | 2021-10-19 | 2022-04-10 |
| 2022-23 | 25,687 | 1,230 | 2022-10-18 | 2023-04-09 |
| 2023-24 | 26,095 | 1,230 | 2023-10-24 | 2024-04-14 |
| 2024-25 | 26,158 | 1,223 | 2024-10-22 | 2025-04-13 |
| 2025-26 | 26,648 | 1,230 | 2025-10-21 | 2026-04-12 |

## Known limitations

- Playoffs, play-in, preseason, and All-Star games are excluded. NBA Cup (in-season tournament) group and knockout games are included and the Cup final is excluded, matching official regular-season accounting.
- Rows for players who did not play (DNP) are dropped: 29,585 rows (2021-22: 5,495; 2022-23: 5,855; 2023-24: 6,290; 2024-25: 6,334; 2025-26: 5,611). A row counts as DNP when minutes are missing or 0, or when the box-score comment is populated.
- Games missing from the dump: 7 regular-season games were postponed and never re-captured upstream, so their box scores are absent here and are not filled in from any other source. 2024-25: 0022400524 (2025-01-09, LAL vs CHA), 0022400532 (2025-01-11, ATL vs HOU), 0022400537 (2025-01-11, LAL vs SAN), 0022400538 (2025-01-11, LAC vs CHA), 0022400617 (2025-01-22, NOP vs MIL), 0022400627 (2025-01-23, UTA vs WAS), 0022400988 (2025-03-17, SAN vs ORL).
- Team abbreviations follow `TeamHistories.csv` for the season in which the game was played.