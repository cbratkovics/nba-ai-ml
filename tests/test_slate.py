import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nba import config
from nba.features import asof
from nba.ingest import kaggle_dump, schedule
from nba.predict import model, slate
from nba.predict.slate import SlateStatus
from tests.conftest import TEAMS


class FakeBooster:
    """Predicts the player's last-10 mean of a target, so outputs are checkable."""

    def __init__(self, target: str):
        self.target = target

    def feature_name(self):
        return list(asof.FEATURE_COLUMNS)

    def predict(self, X):
        return X[f"{self.target}_mean_last10"].fillna(0).to_numpy()


def fake_models() -> model.Models:
    return model.Models(revision="fake-rev", boosters={t: FakeBooster(t) for t in config.TARGETS})


def _schedule_dir(kaggle_dir: Path, tmp_path: Path, games: list[dict]) -> Path:
    """A dump-like directory with TeamHistories.csv and a 2025-26 schedule file."""
    d = tmp_path / "dump"
    d.mkdir()
    (d / kaggle_dump.TEAM_HISTORY_FILE).write_bytes(
        (kaggle_dir / kaggle_dump.TEAM_HISTORY_FILE).read_bytes()
    )
    pd.DataFrame(games).to_csv(d / schedule.schedule_file_name("2025-26"), index=False)
    return d


def _game(gid: int, when: str, home: int, away: int) -> dict:
    return {"gameId": gid, "gameDateTimeEst": when, "homeTeamId": home, "awayTeamId": away}


def test_no_schedule_and_no_games_are_distinct_and_never_load_models(
    game_logs: pd.DataFrame, kaggle_dir: Path, tmp_path: Path
) -> None:
    def must_not_load() -> model.Models:
        raise AssertionError("models must not be loaded without games")

    empty_dir = tmp_path / "nothing"
    empty_dir.mkdir()
    r = slate.run_slate(date(2025, 10, 30), game_logs, empty_dir, must_not_load, "rev")
    assert r.status is SlateStatus.NO_SCHEDULE
    assert "no schedule file for season 2025-26" in r.message
    assert "LeagueSchedule25_26.csv" in r.message

    d = _schedule_dir(
        kaggle_dir, tmp_path, [_game(22500001, "2025-11-01 19:30:00", TEAMS[0][0], TEAMS[1][0])]
    )
    r = slate.run_slate(date(2025, 10, 30), game_logs, d, must_not_load, "rev")
    assert r.status is SlateStatus.NO_GAMES
    assert "no games on this date" in r.message and "1 regular-season games" in r.message
    assert (
        r.message
        != slate.run_slate(date(2025, 10, 30), game_logs, empty_dir, must_not_load, "rev").message
    )


def test_slate_scores_rostered_players_and_writes_outputs(
    game_logs: pd.DataFrame, kaggle_dir: Path, tmp_path: Path
) -> None:
    # A date after every stored 2025-26 game, so all history is as-of.
    d = date(2026, 1, 15)
    games = [
        _game(22500900, "2026-01-15 19:30:00", TEAMS[0][0], TEAMS[1][0]),
        _game(22500901, "2026-01-15 22:00:00", TEAMS[2][0], TEAMS[3][0]),
    ]
    sched_dir = _schedule_dir(kaggle_dir, tmp_path, games)
    r = slate.run_slate(d, game_logs, sched_dir, fake_models, "data-rev")
    assert r.status is SlateStatus.OK and r.n_games == 2
    p = r.predictions
    assert list(p.columns) == list(slate.OUTPUT_COLUMNS)
    assert set(p["game_id"]) == {"0022500900", "0022500901"}
    assert set(p["team"]) == {"LAL", "BOS", "GSW", "MIA"}
    # Roster: everyone who appeared in the team's last 10 games (fixture: 2 players per team).
    assert r.n_players == 8
    assert (p["model_revision"] == "fake-rev").all() and (p["dataset_revision"] == "data-rev").all()
    assert (p["date"] == "2026-01-15").all()
    # Home/opponent come from the schedule.
    lal = p[p["team"] == "LAL"].iloc[0]
    assert bool(lal["home"]) is True and lal["opponent"] == "BOS"
    bos = p[p["team"] == "BOS"].iloc[0]
    assert bool(bos["home"]) is False and bos["opponent"] == "LAL"
    # The fake model returns the last-10 mean, which the output also carries.
    for t in config.TARGETS:
        assert (p[f"pred_{t}"].round(6) == p[f"{t}_mean_last10"].fillna(0).round(6)).all()

    parquet, latest = slate.write_outputs(r, tmp_path / "predictions")
    assert parquet.name == "2026-01-15.parquet" and latest.name == "latest.json"
    on_disk = pd.read_parquet(parquet)
    assert len(on_disk) == 8
    payload = json.loads(latest.read_text())
    assert payload["date"] == "2026-01-15" and payload["model_revision"] == "fake-rev"
    assert payload["n_games"] == 2 and payload["n_players"] == 8
    assert set(payload["predictions"][0]) == {
        "player_id",
        "player_name",
        "team",
        "opponent",
        "home",
        "pred_pts",
        "pred_reb",
        "pred_ast",
    }


def test_roster_uses_only_games_before_the_date(game_logs: pd.DataFrame) -> None:
    hist = game_logs[game_logs["team"] == "LAL"].sort_values("game_date")
    first_date = hist["game_date"].min()
    assert slate.roster(game_logs, "LAL", first_date).empty
    later = slate.roster(game_logs, "LAL", first_date + pd.Timedelta(days=1))
    assert set(later["player_id"]) == set(hist[hist["game_date"] == first_date]["player_id"])
    # A player who last appeared more than 10 games ago is off the roster.
    logs = game_logs.copy()
    victim = int(hist["player_id"].iloc[0])
    keep_first_only = (logs["player_id"] != victim) | (logs["game_date"] == first_date)
    logs = logs[keep_first_only]
    assert victim not in set(slate.roster(logs, "LAL", pd.Timestamp("2027-01-01"))["player_id"])


def test_cli_prints_outcome_lines(
    game_logs: pd.DataFrame,
    kaggle_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from nba.storage import local

    data_dir = tmp_path / "game_logs"
    local.write_per_season(game_logs, data_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        slate.main(
            [
                "--date",
                "2025-10-30",
                "--schedule-dir",
                str(empty),
                "--data-dir",
                str(data_dir),
                "--no-pull",
            ]
        )
        == 0
    )
    assert "SLATE 2025-10-30: no schedule file" in capsys.readouterr().out

    sched_dir = _schedule_dir(
        kaggle_dir, tmp_path, [_game(22500900, "2026-01-15 19:30:00", TEAMS[0][0], TEAMS[1][0])]
    )
    assert (
        slate.main(
            [
                "--date",
                "2025-10-30",
                "--schedule-dir",
                str(sched_dir),
                "--data-dir",
                str(data_dir),
                "--no-pull",
            ]
        )
        == 0
    )
    assert "SLATE 2025-10-30: no games on this date" in capsys.readouterr().out

    monkeypatch.setattr(model, "load_from_hub", fake_models)
    out_dir = tmp_path / "preds"
    assert (
        slate.main(
            [
                "--date",
                "2026-01-15",
                "--schedule-dir",
                str(sched_dir),
                "--data-dir",
                str(data_dir),
                "--predictions-dir",
                str(out_dir),
                "--no-pull",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "SLATE 2026-01-15: 1 games, 4 player predictions" in out
    assert (out_dir / "2026-01-15.parquet").exists() and (out_dir / "latest.json").exists()


def test_traded_player_is_slated_once_on_current_team(
    game_logs: pd.DataFrame, kaggle_dir: Path, tmp_path: Path
) -> None:
    # Move one Lakers player to the Celtics for his last two games (a trade).
    logs = game_logs.copy()
    lal = logs[(logs["team"] == "LAL") & (logs["season"] == "2025-26")].sort_values("game_date")
    pid = int(lal["player_id"].iloc[0])
    last_two = lal[lal["player_id"] == pid].tail(2).index
    logs.loc[last_two, "team"] = "BOS"
    logs.loc[last_two, "opponent"] = "LAL"
    # Give those rows fresh game ids so they do not collide with the Celtics' own rows.
    logs.loc[last_two, "game_id"] = ["0022577701", "0022577702"]
    from nba import schema

    logs = schema.validate(schema.coerce(logs))
    d = date(2026, 1, 15)
    sched_dir = _schedule_dir(
        kaggle_dir, tmp_path, [_game(22500900, "2026-01-15 19:30:00", TEAMS[0][0], TEAMS[1][0])]
    )
    r = slate.run_slate(d, logs, sched_dir, fake_models, "rev")
    p = r.predictions
    assert p[["player_id", "game_id"]].duplicated().sum() == 0
    mine = p[p["player_id"] == pid]
    assert len(mine) == 1 and mine.iloc[0]["team"] == "BOS"
