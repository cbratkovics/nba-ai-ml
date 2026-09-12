"""Deterministic fixtures in the layout of the Kaggle dump, version 515.

Files: PlayerStatistics.csv (box scores, with gameDate, gameType, team ids and
team city/name on every row) and TeamHistories.csv (teamId, teamCity, teamName,
teamAbbrev, seasonFounded, seasonActiveTill, league).

Like the real dump, one whole season (2023-24 here; 2021-22 in the dump) has empty
team id columns and must be resolved by team city/name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba import config
from nba.ingest import kaggle_backfill

# (teamId, city, name). Team 6 changes city and abbreviation after 2023-24.
TEAMS: list[tuple[int, str, str]] = [
    (1610612747, "Los Angeles", "Lakers"),
    (1610612738, "Boston", "Celtics"),
    (1610612744, "Golden State", "Warriors"),
    (1610612748, "Miami", "Heat"),
    (1610612743, "Denver", "Nuggets"),
    (6, "Oldtown", "Renamed"),
]
TEAM_HISTORIES: list[dict] = [
    dict(
        teamId=1610612747,
        teamCity="Los Angeles",
        teamName="Lakers",
        teamAbbrev="LAL  ",
        seasonFounded=1948,
        seasonActiveTill=2100,
        league="NBA",
    ),
    dict(
        teamId=1610612738,
        teamCity="Boston",
        teamName="Celtics",
        teamAbbrev="BOS",
        seasonFounded=1946,
        seasonActiveTill=np.nan,
        league="NBA",
    ),
    dict(
        teamId=1610612744,
        teamCity="Golden State",
        teamName="Warriors",
        teamAbbrev="GSW",
        seasonFounded=1971,
        seasonActiveTill=2100,
        league="NBA",
    ),
    dict(
        teamId=1610612748,
        teamCity="Miami",
        teamName="Heat",
        teamAbbrev="MIA",
        seasonFounded=1988,
        seasonActiveTill=2100,
        league="NBA",
    ),
    dict(
        teamId=1610612743,
        teamCity="Denver",
        teamName="Nuggets",
        teamAbbrev="DEN",
        seasonFounded=1976,
        seasonActiveTill=2100,
        league="NBA",
    ),
    # City and abbreviation change after the 2023-24 season.
    dict(
        teamId=6,
        teamCity="Oldtown",
        teamName="Renamed",
        teamAbbrev="OLD",
        seasonFounded=2000,
        seasonActiveTill=2023,
        league="NBA",
    ),
    dict(
        teamId=6,
        teamCity="Newtown",
        teamName="Renamed",
        teamAbbrev="NEW",
        seasonFounded=2024,
        seasonActiveTill=2100,
        league="NBA",
    ),
    # A defunct team that must never match.
    dict(
        teamId=7,
        teamCity="Gone",
        teamName="Gone",
        teamAbbrev="GONE",
        seasonFounded=1990,
        seasonActiveTill=2001,
        league="NBA",
    ),
    # A non-NBA team sharing a name; must be ignored by the league filter.
    dict(
        teamId=9064,
        teamCity="Madrid",
        teamName="Lakers",
        teamAbbrev="MAD",
        seasonFounded=1944,
        seasonActiveTill=2100,
        league="EuroLeague",
    ),
]
SEASON_STARTS = {"2023-24": "2023-10-24", "2024-25": "2024-10-22", "2025-26": "2025-10-21"}
# The season whose rows carry no team ids (resolved by city/name).
NO_ID_SEASON = "2023-24"
GAMES_PER_SEASON = 12
PLAYERS_PER_TEAM = 2
# NBA Cup games added per season: group/knockout games that must be kept, plus one
# final that must be dropped. The second season labels its final "Regular Season"
# with gameSubLabel "Championship" (as the 2024-25 dump does); the others label it
# "NBA Cup" with a 006-prefixed game id (as 2023-24 does).
CUP_GAMES_PER_SEASON = 2
CUP_FINAL_SEASON_LABELLED_REGULAR = "2024-25"
# Regular-season games kept per fixture season: regular games plus Cup group games.
FIXTURE_GAMES_PER_SEASON = (GAMES_PER_SEASON - 2) * 3 + CUP_GAMES_PER_SEASON
# A pre-cutoff row that the streamed reader must drop.
OLD_GAME_DATE = "2019-11-05 19:30:00"
# Player ids used only for DNP rows (never in the output).
DNP_PLAYER_IDS = (9990, 9991, 9992)
# One played row for this player has numMinutes written as "MM:SS".
CLOCK_PLAYER_ID = 1000
CLOCK_MINUTES_TEXT = "12:30"
CLOCK_MINUTES = 12.5
# One 2024-25 row has no team id and the city spelled differently ("LA"), so it
# can only be resolved by name.
ALT_CITY_PLAYER_ID = 1001
ALT_CITY = "LA"


def make_kaggle_fixture(directory: Path, seed: int = 0) -> tuple[Path, Path]:
    """Write a tiny PlayerStatistics.csv and TeamHistories.csv and return their paths."""
    rng = np.random.default_rng(seed)
    box: list[dict] = []
    game_no = 0
    players = {
        tid: [(1000 + 10 * i + j, f"P{i}{j}") for j in range(PLAYERS_PER_TEAM)]
        for i, (tid, _, _) in enumerate(TEAMS)
    }
    city_of = {tid: city for tid, city, _ in TEAMS}
    name_of = {tid: name for tid, _, name in TEAMS}

    def box_row(
        pid: int, pname: str, tid: int, opp: int, is_home: int, gid: int, when: str, game_type: str
    ) -> dict:
        minutes = float(rng.uniform(4, 38))
        pts = int(rng.poisson(minutes * 0.6))
        fga = pts // 2 + int(rng.integers(0, 6))
        fgm = min(fga, pts // 3)
        fg3a = int(rng.integers(0, 8))
        fg3m = int(rng.integers(0, fg3a + 1))
        fta = int(rng.integers(0, 8))
        ftm = int(rng.integers(0, fta + 1))
        oreb = int(rng.integers(0, 4))
        dreb = int(rng.integers(0, 9))
        season = kaggle_backfill.season_from_date(pd.Timestamp(when))
        no_ids = season == NO_ID_SEASON
        return {
            "firstName": pname,
            "lastName": name_of[tid],
            "personId": pid,
            "gameId": gid,
            "gameDate": when,
            "gameLabel": np.nan,
            "gameSubLabel": np.nan,
            "playerteamCity": city_of[tid],
            "playerteamName": name_of[tid],
            "opponentteamCity": city_of[opp],
            "opponentteamName": name_of[opp],
            "playerteamId": np.nan if no_ids else tid,
            "opponentteamId": np.nan if no_ids else opp,
            "gameType": game_type,
            "home": is_home,
            "numMinutes": minutes,
            "comment": "",
            "points": pts,
            "assists": int(rng.poisson(minutes * 0.12)),
            "blocks": int(rng.integers(0, 3)),
            "steals": int(rng.integers(0, 3)),
            "fieldGoalsAttempted": fga,
            "fieldGoalsMade": fgm,
            "threePointersAttempted": fg3a,
            "threePointersMade": fg3m,
            "freeThrowsAttempted": fta,
            "freeThrowsMade": ftm,
            "reboundsOffensive": oreb,
            "reboundsDefensive": dreb,
            "reboundsTotal": oreb + dreb,
            "foulsPersonal": int(rng.integers(0, 6)),
            "turnovers": int(rng.integers(0, 5)),
            "plusMinusPoints": int(rng.integers(-15, 16)),
        }

    # One old game that precedes BACKFILL_START.
    game_no += 1
    for tid, opp, is_home in ((TEAMS[0][0], TEAMS[1][0], 1), (TEAMS[1][0], TEAMS[0][0], 0)):
        for pid, pname in players[tid]:
            box.append(
                box_row(
                    pid,
                    pname,
                    tid,
                    opp,
                    is_home,
                    21900000 + game_no,
                    OLD_GAME_DATE,
                    "Regular Season",
                )
            )

    for start in SEASON_STARTS.values():
        day = pd.Timestamp(start)
        for k in range(GAMES_PER_SEASON):
            # Mostly 2-day gaps with an occasional back-to-back.
            day = day + pd.Timedelta(days=1 if k % 5 == 4 else 2)
            order = rng.permutation(len(TEAMS))
            game_type = "Regular Season"
            if k == 0:
                game_type = "Preseason"  # must be filtered out
            if k == GAMES_PER_SEASON - 1:
                game_type = "Playoffs"  # must be filtered out
            for h in range(0, len(TEAMS), 2):
                game_no += 1
                gid = 22300000 + game_no  # int like the Kaggle dump (no leading zeros)
                home_tid, away_tid = TEAMS[order[h]][0], TEAMS[order[h + 1]][0]
                when = f"{day.date()} 19:30:00"
                for tid, opp, is_home in ((home_tid, away_tid, 1), (away_tid, home_tid, 0)):
                    for pid, pname in players[tid]:
                        box.append(box_row(pid, pname, tid, opp, is_home, gid, when, game_type))

    # NBA Cup games: group games (kept) and one final per season (dropped).
    for season_no, (season, start) in enumerate(SEASON_STARTS.items()):
        day = pd.Timestamp(start) + pd.Timedelta(days=40)
        for c in range(CUP_GAMES_PER_SEASON + 1):
            game_no += 1
            when = f"{(day + pd.Timedelta(days=c)).date()} 19:30:00"
            home_tid, away_tid = TEAMS[c % len(TEAMS)][0], TEAMS[(c + 1) % len(TEAMS)][0]
            is_final = c == CUP_GAMES_PER_SEASON
            if not is_final:
                gid, game_type, sub = 22300000 + game_no, "NBA Emirates Cup", "East Group A"
            elif season == CUP_FINAL_SEASON_LABELLED_REGULAR:
                gid, game_type, sub = 22300000 + game_no, "Regular Season", "Championship"
            else:
                gid, game_type, sub = 62300001 + season_no, "NBA Cup", np.nan
            for tid, opp, is_home in ((home_tid, away_tid, 1), (away_tid, home_tid, 0)):
                for pid, pname in players[tid]:
                    row = box_row(pid, pname, tid, opp, is_home, gid, when, game_type)
                    row["gameLabel"] = "Emirates NBA Cup"
                    row["gameSubLabel"] = sub
                    box.append(row)

    regular = [
        r
        for r in box
        if r["gameType"] == "Regular Season"
        and r["gameDate"] != OLD_GAME_DATE
        and pd.isna(r["gameSubLabel"])
    ]

    # Did-not-play rows inside regular-season games, all of which must be dropped:
    # per season one row with no minutes and a comment; in the second season also a
    # row with 0 minutes and no comment, and a row with minutes but a comment.
    by_season: dict[str, dict] = {}
    for r in regular:
        by_season.setdefault(kaggle_backfill.season_from_date(pd.Timestamp(r["gameDate"])), r)
    for i, (_season, template) in enumerate(sorted(by_season.items())):
        dnp = dict(template)
        dnp["personId"] = DNP_PLAYER_IDS[0]
        dnp["firstName"], dnp["lastName"] = "Did", "NotPlay"
        dnp["comment"] = "DNP - Coach's Decision"
        for col in ("numMinutes", "points", "assists", "reboundsTotal"):
            dnp[col] = np.nan
        box.append(dnp)
        if i == 1:
            zero = dict(template)
            zero["personId"] = DNP_PLAYER_IDS[1]
            zero["firstName"], zero["lastName"] = "Zero", "Minutes"
            zero["numMinutes"] = 0.0
            box.append(zero)
            noted = dict(template)
            noted["personId"] = DNP_PLAYER_IDS[2]
            noted["firstName"], noted["lastName"] = "Has", "Comment"
            noted["comment"] = "NWT - Injury/Illness"
            box.append(noted)

    # One played row with minutes written as MM:SS, as the dump sometimes does.
    clock = next(r for r in regular if r["personId"] == CLOCK_PLAYER_ID)
    clock["numMinutes"] = CLOCK_MINUTES_TEXT

    # One 2024-25 played row with no team ids and an alternate city spelling.
    alt = next(
        r
        for r in regular
        if r["personId"] == ALT_CITY_PLAYER_ID
        and kaggle_backfill.season_from_date(pd.Timestamp(r["gameDate"])) == "2024-25"
    )
    alt["playerteamId"] = np.nan
    alt["opponentteamId"] = np.nan
    alt["playerteamCity"] = ALT_CITY

    directory.mkdir(parents=True, exist_ok=True)
    box_path = directory / kaggle_backfill.BOX_SCORE_FILE
    hist_path = directory / kaggle_backfill.TEAM_HISTORY_FILE
    pd.DataFrame(box).to_csv(box_path, index=False)
    pd.DataFrame(TEAM_HISTORIES).to_csv(hist_path, index=False)
    return box_path, hist_path


@pytest.fixture(autouse=True)
def relaxed_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture has ~30 games per season; keep the real thresholds out of the way."""
    monkeypatch.setattr(config, "MIN_ROWS_PER_SEASON", 0)
    monkeypatch.setattr(config, "MIN_GAMES_PER_SEASON", 0)
    monkeypatch.setattr(config, "FULL_SEASON_GAMES", FIXTURE_GAMES_PER_SEASON)


@pytest.fixture
def kaggle_dir(tmp_path: Path) -> Path:
    make_kaggle_fixture(tmp_path / "kaggle")
    return tmp_path / "kaggle"


@pytest.fixture
def game_logs(kaggle_dir: Path) -> pd.DataFrame:
    return kaggle_backfill.map_to_schema(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_team_histories(kaggle_dir)
    )
