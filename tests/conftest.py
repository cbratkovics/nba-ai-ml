"""Deterministic fixtures in the layout of the Kaggle dump, version 515.

Files: PlayerStatistics.csv (box scores, with gameDate and gameType on every row)
and TeamHistories.csv (teamId, teamAbbrev, seasonFounded, seasonActiveTill).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba.ingest import kaggle_backfill

# (teamId, lastName used for its fixture players). Team 6 is renamed between seasons.
TEAMS: list[tuple[int, str]] = [
    (1610612747, "Lakers"),
    (1610612738, "Celtics"),
    (1610612744, "Warriors"),
    (1610612748, "Heat"),
    (1610612743, "Nuggets"),
    (6, "Renamed"),
]
TEAM_HISTORIES: list[dict] = [
    {"teamId": 1610612747, "teamAbbrev": "LAL", "seasonFounded": 1948, "seasonActiveTill": np.nan},
    {"teamId": 1610612738, "teamAbbrev": "BOS", "seasonFounded": 1946, "seasonActiveTill": np.nan},
    {"teamId": 1610612744, "teamAbbrev": "GSW", "seasonFounded": 1971, "seasonActiveTill": np.nan},
    {"teamId": 1610612748, "teamAbbrev": "MIA", "seasonFounded": 1988, "seasonActiveTill": np.nan},
    {"teamId": 1610612743, "teamAbbrev": "DEN", "seasonFounded": 1976, "seasonActiveTill": np.nan},
    # Abbreviation changes after the 2023-24 season.
    {"teamId": 6, "teamAbbrev": "OLD", "seasonFounded": 2000, "seasonActiveTill": 2023},
    {"teamId": 6, "teamAbbrev": "NEW", "seasonFounded": 2024, "seasonActiveTill": np.nan},
    # A defunct team that must never match.
    {"teamId": 7, "teamAbbrev": "GONE", "seasonFounded": 1990, "seasonActiveTill": 2001},
]
SEASON_STARTS = {"2023-24": "2023-10-24", "2024-25": "2024-10-22", "2025-26": "2025-10-21"}
GAMES_PER_SEASON = 12
PLAYERS_PER_TEAM = 2
# A pre-cutoff row that the streamed reader must drop.
OLD_GAME_DATE = "2019-11-05 19:30:00"
# Player ids used only for DNP rows (never in the output).
DNP_PLAYER_IDS = (9990, 9991, 9992)
# One played row for this player has numMinutes written as "MM:SS".
CLOCK_PLAYER_ID = 1000
CLOCK_MINUTES_TEXT = "12:30"
CLOCK_MINUTES = 12.5


def make_kaggle_fixture(directory: Path, seed: int = 0) -> tuple[Path, Path]:
    """Write a tiny PlayerStatistics.csv and TeamHistories.csv and return their paths."""
    rng = np.random.default_rng(seed)
    box: list[dict] = []
    game_no = 0
    players = {
        tid: [(1000 + 10 * i + j, f"P{i}{j}") for j in range(PLAYERS_PER_TEAM)]
        for i, (tid, _) in enumerate(TEAMS)
    }
    last_names = dict(TEAMS)

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
        return {
            "firstName": pname,
            "lastName": last_names[tid],
            "personId": pid,
            "gameId": gid,
            "gameDate": when,
            "playerteamId": tid,
            "opponentteamId": opp,
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

    # Did-not-play rows inside regular-season games, all of which must be dropped:
    # per season one row with no minutes and a comment; in the second season also a
    # row with 0 minutes and no comment, and a row with minutes but a comment.
    regular = [
        r for r in box if r["gameType"] == "Regular Season" and r["gameDate"] != OLD_GAME_DATE
    ]
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

    directory.mkdir(parents=True, exist_ok=True)
    box_path = directory / kaggle_backfill.BOX_SCORE_FILE
    hist_path = directory / kaggle_backfill.TEAM_HISTORY_FILE
    pd.DataFrame(box).to_csv(box_path, index=False)
    pd.DataFrame(TEAM_HISTORIES).to_csv(hist_path, index=False)
    return box_path, hist_path


@pytest.fixture
def kaggle_dir(tmp_path: Path) -> Path:
    make_kaggle_fixture(tmp_path / "kaggle")
    return tmp_path / "kaggle"


@pytest.fixture
def game_logs(kaggle_dir: Path) -> pd.DataFrame:
    return kaggle_backfill.map_to_schema(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_team_histories(kaggle_dir)
    )
