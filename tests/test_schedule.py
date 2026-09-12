from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nba.ingest import kaggle_dump, schedule
from tests.conftest import TEAMS


def _write_schedule(path: Path, id_style: str) -> None:
    home, away = (
        ("hometeamId", "awayteamId") if id_style == "24_25" else ("homeTeamId", "awayTeamId")
    )
    rows = [
        # regular season, two games on one night, one the next
        {
            "gameId": 22500001,
            "gameDateTimeEst": "2025-10-21 19:30:00",
            home: TEAMS[0][0],
            away: TEAMS[1][0],
            "gameLabel": None,
            "gameSubLabel": None,
        },
        {
            "gameId": 22500002,
            "gameDateTimeEst": "2025-10-21 22:00:00",
            home: TEAMS[2][0],
            away: TEAMS[3][0],
            "gameLabel": None,
            "gameSubLabel": None,
        },
        {
            "gameId": 22500003,
            "gameDateTimeEst": "2025-10-22 19:00:00",
            home: TEAMS[4][0],
            away: TEAMS[5][0],
            "gameLabel": None,
            "gameSubLabel": None,
        },
        # preseason and a Cup final: excluded
        {
            "gameId": 12500001,
            "gameDateTimeEst": "2025-10-05 19:00:00",
            home: TEAMS[0][0],
            away: TEAMS[1][0],
            "gameLabel": "Preseason",
            "gameSubLabel": None,
        },
        {
            "gameId": 62500001,
            "gameDateTimeEst": "2025-12-16 20:30:00",
            home: TEAMS[0][0],
            away: TEAMS[1][0],
            "gameLabel": "Emirates NBA Cup",
            "gameSubLabel": "Championship",
        },
        # Cup group game labelled as regular season id: kept
        {
            "gameId": 22500010,
            "gameDateTimeEst": "2025-11-04 19:00:00",
            home: TEAMS[2][0],
            away: TEAMS[1][0],
            "gameLabel": "Emirates NBA Cup",
            "gameSubLabel": "East Group A",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize("id_style", ["24_25", "25_26"])
def test_load_schedule_handles_both_column_styles(
    kaggle_dir: Path, tmp_path: Path, id_style: str
) -> None:
    path = tmp_path / schedule.schedule_file_name("2025-26")
    _write_schedule(path, id_style)
    hist = kaggle_dump.load_team_histories(kaggle_dir)
    sched = schedule.load_schedule(path, hist)
    assert list(sched["game_id"]) == [
        "0022500001",
        "0022500002",
        "0022500010",
        "0022500003",
    ] or set(sched["game_id"]) == {"0022500001", "0022500002", "0022500003", "0022500010"}
    assert set(sched["season"]) == {"2025-26"}
    first = sched[sched["game_id"] == "0022500001"].iloc[0]
    assert (first["home"], first["away"]) == ("LAL", "BOS")
    assert first["game_date"] == pd.Timestamp("2025-10-21")


def test_games_on_and_file_naming(kaggle_dir: Path, tmp_path: Path) -> None:
    assert schedule.schedule_file_name("2025-26") == "LeagueSchedule25_26.csv"
    assert schedule.schedule_file_name("2026-27") == "LeagueSchedule26_27.csv"
    # October starts a season: September still belongs to the previous one.
    assert schedule.season_for_date(date(2026, 9, 12)) == "2025-26"
    assert schedule.season_for_date(date(2026, 10, 12)) == "2026-27"
    assert schedule.season_for_date(date(2026, 6, 1)) == "2025-26"
    path = tmp_path / schedule.schedule_file_name("2025-26")
    _write_schedule(path, "25_26")
    sched = schedule.load_schedule(path, kaggle_dump.load_team_histories(kaggle_dir))
    assert len(schedule.games_on(sched, date(2025, 10, 21))) == 2
    assert len(schedule.games_on(sched, date(2025, 10, 22))) == 1
    assert schedule.games_on(sched, date(2025, 10, 23)).empty
    # Only the date's own season file is consulted.
    assert schedule.schedule_path(tmp_path, date(2025, 10, 21)) == path
    assert schedule.schedule_path(tmp_path, date(2026, 10, 21)) is None


def test_schedule_missing_column_is_an_error(tmp_path: Path, kaggle_dir: Path) -> None:
    path = tmp_path / "LeagueSchedule25_26.csv"
    pd.DataFrame([{"gameId": 22500001, "gameDateTimeEst": "2025-10-21 19:30:00"}]).to_csv(
        path, index=False
    )
    with pytest.raises(KeyError, match="home_team_id"):
        schedule.load_schedule(path, kaggle_dump.load_team_histories(kaggle_dir))
