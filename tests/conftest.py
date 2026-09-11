"""Deterministic fixtures in the layout of the Kaggle dump (PlayerStatistics.csv, Games.csv)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba.ingest import kaggle_backfill

# Real nicknames so the nickname -> abbreviation map resolves.
NICKNAMES = ["Lakers", "Celtics", "Warriors", "Heat", "Nuggets", "Bucks"]
SEASON_STARTS = {"2023-24": "2023-10-24", "2024-25": "2024-10-22", "2025-26": "2025-10-21"}
GAMES_PER_SEASON = 12
PLAYERS_PER_TEAM = 2


def make_kaggle_fixture(directory: Path, seed: int = 0) -> tuple[Path, Path]:
    """Write a tiny PlayerStatistics.csv and Games.csv and return their paths."""
    rng = np.random.default_rng(seed)
    games: list[dict] = []
    box: list[dict] = []
    game_no = 0
    players = {
        nick: [(1000 + 10 * i + j, f"P{i}{j}") for j in range(PLAYERS_PER_TEAM)]
        for i, nick in enumerate(NICKNAMES)
    }

    for _season, start in SEASON_STARTS.items():
        day = pd.Timestamp(start)
        for k in range(GAMES_PER_SEASON):
            # Mostly 2-day gaps with an occasional back-to-back.
            day = day + pd.Timedelta(days=1 if k % 5 == 4 else 2)
            order = rng.permutation(len(NICKNAMES))
            game_type = "Regular Season"
            if k == 0:
                game_type = "Preseason"  # must be filtered out
            if k == GAMES_PER_SEASON - 1:
                game_type = "Playoffs"  # must be filtered out
            for h in range(0, len(NICKNAMES), 2):
                game_no += 1
                gid = 22300000 + game_no  # int like the Kaggle dump (no leading zeros)
                home_nick, away_nick = NICKNAMES[order[h]], NICKNAMES[order[h + 1]]
                games.append(
                    {
                        "gameId": gid,
                        "gameDate": f"{day.date()} 19:30:00",
                        "hometeamName": home_nick,
                        "awayteamName": away_nick,
                        "gameType": game_type,
                    }
                )
                for nick, opp, is_home in ((home_nick, away_nick, 1), (away_nick, home_nick, 0)):
                    for pid, pname in players[nick]:
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
                        box.append(
                            {
                                "firstName": pname,
                                "lastName": nick,
                                "personId": pid,
                                "gameId": gid,
                                "playerteamName": nick,
                                "opponentteamName": opp,
                                "home": is_home,
                                "numMinutes": minutes,
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
                        )
    # One did-not-play row (no minutes) that must be dropped.
    dnp = dict(box[-1])
    dnp["personId"] = 9999
    dnp["firstName"], dnp["lastName"] = "Did", "NotPlay"
    for col in ("numMinutes", "points", "assists", "reboundsTotal"):
        dnp[col] = np.nan
    box.append(dnp)

    directory.mkdir(parents=True, exist_ok=True)
    box_path = directory / kaggle_backfill.BOX_SCORE_FILE
    sched_path = directory / kaggle_backfill.SCHEDULE_FILE
    pd.DataFrame(box).to_csv(box_path, index=False)
    pd.DataFrame(games).to_csv(sched_path, index=False)
    return box_path, sched_path


@pytest.fixture
def kaggle_dir(tmp_path: Path) -> Path:
    make_kaggle_fixture(tmp_path / "kaggle")
    return tmp_path / "kaggle"


@pytest.fixture
def game_logs(kaggle_dir: Path) -> pd.DataFrame:
    return kaggle_backfill.map_to_schema(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_schedule(kaggle_dir)
    )
