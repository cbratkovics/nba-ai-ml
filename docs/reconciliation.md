# Data reconciliation notes

What the Kaggle backfill found in the source dump and how the loader handles it.
This file is extended when `nba_api` daily rows are reconciled against the backfill.

Source: Eoin Moore, *Historical NBA Data and Player Box Scores*, Kaggle, version 515
(files dated 2025-06-15). Loader: `nba/ingest/kaggle_backfill.py`. Only
`PlayerStatistics.csv` and `TeamHistories.csv` are read.

## Backfill result (seasons 2021-22 to 2025-26, regular season incl. NBA Cup, final excluded)

| Season | Rows | Games | First game | Last game | DNP dropped | Team by name | Cup games | Cup final dropped |
|---|---:|---:|---|---|---:|---:|---:|---:|
| 2021-22 | 25,826 | 1,230 | 2021-10-19 | 2022-04-10 | 5,495 | 25,805 | 0 | 0 |
| 2022-23 | 25,687 | 1,230 | 2022-10-18 | 2023-04-09 | 5,855 | 0 | 0 | 0 |
| 2023-24 | 26,095 | 1,230 | 2023-10-24 | 2024-04-14 | 6,290 | 0 | 66 | 1 |
| 2024-25 | 26,158 | 1,223 | 2024-10-22 | 2025-04-13 | 6,334 | 0 | 66 | 1 |
| 2025-26 | 26,648 | 1,230 | 2025-10-21 | 2026-04-12 | 5,611 | 0 | 66 | 1 |

A full regular season is 1,230 games (30 teams × 82 / 2). Rows are one per
(player_id, game_id) with no duplicates; 13 to 30 players per game.

## Findings

### 1. Empty team ids for all of 2021-22 (name fallback)

`playerteamId` and `opponentteamId` are empty for every 2021-22 row (31,297 rows,
1,229 games; 21 rows have ids) and for a few hundred later rows, which are all
0-minute rows. The city and name columns are always populated.

Handling: each row is resolved by id when present, then by `(playerteamCity,
playerteamName)` against the NBA rows of `TeamHistories.csv` active in that season
(`seasonFounded <= start year <= seasonActiveTill`), then by name alone. The
name-only step exists because the Clippers appear as both `LA` and `Los Angeles`
while the history table only has `Los Angeles`. Where both an id and a name are
present they agree on all 121 distinct (id, city, name, season) combinations.
The `team_by_name` column above counts kept rows resolved without an id.

### 2. Minutes in two formats

`numMinutes` is a decimal for most rows (`39.166666`) but `MM:SS` for 2,097 raw rows
(`23:21`). `parse_minutes` accepts both; empty values become NaN and count as DNP.

### 3. Padded abbreviations and non-NBA rows in TeamHistories.csv

`teamAbbrev` values carry trailing spaces (`ATL  `) and are stripped. The file also
lists EuroLeague and other non-NBA teams and All-Star squads; only rows with
`league == NBA` are used. Active franchises have `seasonActiveTill = 2100`.

### 4. `SAN` vs nba_api `SAS`

TeamHistories abbreviates San Antonio as `SAN`; `nba_api` (stats.nba.com) uses `SAS`.
All other 29 abbreviations match nba_api's static team table. When daily `nba_api`
rows are reconciled, `SAS` must be mapped to `SAN` (or the backfill re-mapped) before
comparing `team`/`opponent`.

### 5. Empty `gameType` rows

50 rows have an empty `gameType`, all from one game (`0022500696`, 2026-01-30) and
all with 0 minutes and no team ids. They are excluded by the game-type filter and
would be dropped as DNP anyway.

### 6. NBA Cup (in-season tournament) labeling differs by season

Official accounting counts Cup group and knockout games as regular-season games and
does not count the final. The dump labels them:

| Season | Group + knockout (66 games) | Final |
|---|---|---|
| 2023-24 | `gameType = NBA Emirates Cup`, `gameLabel = Emirates NBA Cup`, `gameSubLabel` = group / quarterfinal / semifinal | `gameType = NBA Cup`, game id `0062300001`, `gameSubLabel` empty |
| 2024-25 | `gameType = Regular Season`, `gameLabel = Emirates NBA Cup` | `gameType = Regular Season`, `gameSubLabel = Championship`, game id `0062400001` |
| 2025-26 | `gameType = Regular Season`, `gameLabel = Emirates NBA Cup` | 35 rows split across `gameType = Emirates NBA Cup` (30) and `in-season-knockout` (5), `gameSubLabel = Championship`, game id `0062500001` |

Handling: `GAME_TYPES` keeps `Regular Season` and every Cup label; the loader then
drops any row whose `gameSubLabel` is `Championship` or whose game id starts with
`006` (the season-type code of the final). `gameLabel = Emirates NBA Cup` identifies
Cup games for the `cup_games` count. Before this rule 2023-24 had only 1,164 games.

### 7. Seven 2024-25 games have no box-score rows

`LeagueSchedule24_25.csv` lists 1,230 regular-season game ids (prefix `002`); seven
of them have no rows in `PlayerStatistics.csv`, leaving 2024-25 at 1,223 games
(abbreviations as stored in the dataset, so San Antonio is `SAN`):

| Game id | Scheduled | Home | Away |
|---|---|---|---|
| 0022400524 | 2025-01-09 | LAL | CHA |
| 0022400532 | 2025-01-11 | ATL | HOU |
| 0022400537 | 2025-01-11 | LAL | SAN |
| 0022400538 | 2025-01-11 | LAC | CHA |
| 0022400617 | 2025-01-22 | NOP | MIL |
| 0022400627 | 2025-01-23 | UTA | WAS |
| 0022400988 | 2025-03-17 | SAN | ORL |

`Games.csv` carries two of them (`0022400627` on 2025-03-19, `0022400988` on
2025-04-01) with 0–0 scores, consistent with postponed games whose rescheduled
box scores were never captured in the dump. These games cannot be recovered from
the dump; they are a candidate for backfilling from `nba_api` once daily ingestion
exists.

### 8. Did-not-play rows

A row is DNP when `numMinutes` is null or 0, or when `comment` is populated
(`DNP - Coach's Decision`, injury notes, and similar). DNP rows are dropped and
counted per season (table above); 29,585 in total across the five seasons.

### 9. An 83-point box score is genuine

`PlayerStatistics.csv` records Bam Adebayo (player id `1628389`, MIA vs WAS, game
`0022500938`, 2026-03-10) with 41.54 minutes, 83 points, 9 rebounds and 3 assists: the
season maximum, the largest points residual of the replay (predicted 20.47) and the
`2026-03-10` golden-set case for the analyst agent. Verified 2026-09-15 from a local machine
(GitHub runners cannot reach nba.com) with `nba_api` endpoint `boxscoretraditionalv3` for
game `0022500938`: Adebayo MIA, 41:54, 83 pts, 9 reb, 3 ast. The row stands. The warehouse
carries a plausibility warning for box scores above 70 points, 30 rebounds or 25 assists
with this game listed as a known exception, so the warning cites the verification instead
of reopening it.

### 10. Other observations

- `gameDate` is `YYYY-MM-DD HH:MM:SS` and parsed with `format="ISO8601"`; it is empty
  on some non-NBA rows, which fall before the 2021-10-01 cutoff and are skipped.
- Player ids and team ids are written as integers, but the empty team-id columns make
  pandas emit `1610612747.0` when re-saved, so the loader reads id columns as
  nullable floats and casts later.
- Game ids in the dump have no leading zeros (`22300001`); the loader zero-pads to
  10 characters to match nba_api (`0022300001`).
- The dump also ships `PlayerStatisticsExtended.csv`, `TeamStatistics*.csv`,
  `Players.csv`, `Games.csv`, `LeagueSchedule24_25.csv`, `LeagueSchedule25_26.csv`
  (whose home/away column names differ: `hometeamId` vs `homeTeamId`), and
  `PlayByPlay.parquet` (933 MB). None are used by the backfill.
- Rows per team in 2025-26 range from 813 to 935, i.e. roughly 10 to 11 players per
  game after DNP removal.

## Daily-vs-backfill reconciliation (design)

The nightly job (`nba/nightly.py`, `.github/workflows/nightly.yml`, 10:00 UTC) keeps the
Hugging Face dataset in step with the Kaggle dump without ever re-running the backfill:

1. **Same rules, one implementation.** Every parsing and mapping rule lives in
   `nba/ingest/kaggle_dump.py` and is used by both the one-off backfill and the daily
   ingest, so a row produced on a given day is byte-for-byte what the backfill would
   have produced for it (except `source`).
2. **Window.** The daily ingest downloads only `PlayerStatistics.csv`,
   `TeamHistories.csv`, and the season's `LeagueScheduleYY_YY.csv` with the Kaggle
   single-file API, streams the box scores, and keeps rows with
   `game_date > max(stored game_date) - 7 days`. Seven days covers late corrections
   the dump author applies to recent games; an empty window (off-season) is a normal
   zero-count run.
3. **Classification.** Each incoming `(player_id, game_id)` is compared with the stored
   row on every schema column except `source`: **new** (not stored), **unchanged**
   (all columns equal; floats to 4 decimals), or **changed**. New and changed rows are
   written with `source = kaggle_daily`; unchanged rows keep `kaggle_v515`.
4. **Write and push.** Only the affected season Parquet files are rewritten and pushed,
   together with a refreshed dataset card whose season table (rows, games, dates) is
   recomputed and whose DNP bullet is left as the backfill wrote it. Nothing is pushed
   on a day with zero new or changed rows.
5. **Report.** `data/daily_report.json` records the stored revision, the window, row
   counts at each stage, new/changed/unchanged counts, up to 20 changed examples with
   the differing fields, the seasons written, and the revision after the push. The
   nightly workflow uploads it (with `data/nightly_summary.json` and the log) as an
   artifact, and `predictions/rolling_metrics.json` in the dataset repo accumulates
   the residual lines.
6. **No-schedule and no-games days.** Until the dump author publishes
   `LeagueSchedule26_27.csv`, every run from 2026-10-01 logs
   `SLATE <date>: no schedule file for season 2026-27 (...)`; before then (season
   2025-26 by the October rule) it logs `SLATE <date>: no games on this date (...)`.
   Both exit 0 and are distinguishable in the job summary; schema errors fail the job.

## First nightly run (workflow_dispatch, 2026-09-12)

GitHub Actions run `34668092730` performed the first real Kaggle single-file download
(`PlayerStatistics.csv`, `TeamHistories.csv`, `LeagueSchedule25_26.csv`) and produced
this `data/daily_report.json`:

```json
{
  "date": "2026-09-12",
  "dataset_revision_before": "b20b5601de213fa8e704ebffaafd18f182ea68c3",
  "stored_rows": 130414,
  "stored_max_game_date": "2026-04-12",
  "window_start_exclusive": "2026-04-05",
  "window_rows_in_dump": 4139,
  "window_rows_after_rules": 1234,
  "dnp_dropped_in_window": 327,
  "counts": {
    "new": 0,
    "changed": 0,
    "unchanged": 1234
  },
  "changed_examples": [],
  "seasons_written": [],
  "schedule_file": "LeagueSchedule25_26.csv",
  "pushed": false,
  "dataset_revision_after": "b20b5601de213fa8e704ebffaafd18f182ea68c3"
}
```

Reading: the stored data ends 2026-04-12, so the window began 2026-04-06; the dump had
4,139 rows there, 1,234 survived the game-type, Cup-final, and DNP rules (327 DNP rows
dropped in the window), and every one of them matched the stored row exactly. Nothing
was written or pushed. The same run logged `RESIDUALS 2026-09-11: no predictions file`
and `SLATE 2026-09-12: no games on this date (LeagueSchedule25_26.csv lists 1230
regular-season games, none on 2026-09-12)`, which is the expected off-season outcome
until 2026-10-01, after which the message becomes `no schedule file for season 2026-27`
until the dump author publishes `LeagueSchedule26_27.csv`.

## Season replay 2025-26 (backtest of the nightly path)

GitHub Actions run `34668262142` (`.github/workflows/replay.yml`) replayed the slate for
all 164 game dates of 2025-26, truncating the game logs to games before each
date, scoring with model `commit 50a3b2e / HF fb427de` on dataset `b20b5601`, and
joining actuals. Output committed as `reports/replay_2025-26.json` (the run pushed at
dataset-repo revision `8a127c93`; a later identical run, `6cbc915b`, is what the site
reads; `reports/provenance_b20b5601.json` records both hashes).

| Target | metrics.json MAE | replay MAE (same population) | diff | replay MAE (all rows with actuals) |
|---|---:|---:|---:|---:|
| pts | 4.7644 | 4.7665 | +0.0021 | 4.8639 |
| reb | 1.9421 | 1.9429 | +0.0008 | 2.0228 |
| ast | 1.4310 | 1.4320 | +0.0010 | 1.4029 |

The first three columns are the training population (minutes >= 10, both baselines
defined); the last is every replayed row with a box score, where the last-10 baseline is
the better predictor on all three targets (README, "All rows").

Tolerance 0.05 per target: **passed**. The restricted population (minutes >= 10, both
baselines present) has 22,075 replayed rows against 22,244 in
`metrics.json`; the small gap is the 367 rows
(of 617 actual rows in total) the roster rule never
predicted, chiefly post-trade debuts and players with no appearance in their team's
previous ten games. Of 38,372 slated players, 12,341 did not
play (no box-score row), which is what the nightly residuals report as
"player did not play". One replay bug was found and fixed on the way: a player traded
inside the ten-game window was slated for both teams; the roster rule now keeps a
player only on the team of their most recent game.

## What remains manual

- The seven 2024-25 games missing from the dump (finding 7) still need box scores; the
  dump cannot supply them and nba.com is unreachable from GitHub runners (the probe
  timed out on stats.nba.com and got 403 from cdn.nba.com), so they must be fetched
  from `nba_api` on a machine that can reach it, mapped with the same rules, and pushed.
- `nba_api` checks in general (the `SAS`/`SAN` alias, and the one-week overlap
  reconciliation below) can only run locally for the same reason.
- LightGBM does not load on the development Mac without `libomp`; training, scoring,
  and the replay run in Actions.
- The 2026-27 schedule file appears only when the dump author publishes it; until then
  every nightly run ends at the no-schedule line.

## Open items for the nba_api path (Phase 1b acceptance checklist)

- (a) Fetch the seven missing 2024-25 games (finding 7) from nba_api, map them into the
  canonical schema with `source` set to the nba_api pipeline, and confirm 2024-25 reaches
  1,230 distinct games.
- (b) Add a `SAS` -> `SAN` alias (or re-map the backfill to `SAS`) so nba_api rows and
  backfill rows agree on `team`/`opponent` (finding 4).
- (c) Before trusting daily rows, fetch one week of 2025-26 games that the dump already
  covers and reconcile them against the backfill row by row on (player_id, game_id):
  same row set, same minutes to two decimals, same counting stats. Record the result here.
