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
of them have no rows in `PlayerStatistics.csv`, leaving 2024-25 at 1,223 games:

| Game id | Scheduled | Home | Away |
|---|---|---|---|
| 0022400524 | 2025-01-09 | LAL | CHA |
| 0022400532 | 2025-01-11 | ATL | HOU |
| 0022400537 | 2025-01-11 | LAL | SAS |
| 0022400538 | 2025-01-11 | LAC | CHA |
| 0022400617 | 2025-01-22 | NOP | MIL |
| 0022400627 | 2025-01-23 | UTA | WAS |
| 0022400988 | 2025-03-17 | SAS | ORL |

`Games.csv` carries two of them (`0022400627` on 2025-03-19, `0022400988` on
2025-04-01) with 0–0 scores, consistent with postponed games whose rescheduled
box scores were never captured in the dump. These games cannot be recovered from
the dump; they are a candidate for backfilling from `nba_api` once daily ingestion
exists.

### 8. Did-not-play rows

A row is DNP when `numMinutes` is null or 0, or when `comment` is populated
(`DNP - Coach's Decision`, injury notes, and similar). DNP rows are dropped and
counted per season (table above); 29,585 in total across the five seasons.

### 9. Other observations

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

## Open items for nba_api reconciliation

- Map `SAS` to `SAN` (finding 4).
- Decide whether to fetch the seven missing 2024-25 games from nba_api (finding 7).
- Confirm nba_api box scores use the same `minutes` convention (decimal minutes).
