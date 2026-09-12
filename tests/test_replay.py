import json
from pathlib import Path

import pandas as pd
import pytest

from nba import config
from nba.ingest import kaggle_dump, schedule
from nba.predict import replay
from nba.storage import local
from tests.conftest import TEAMS
from tests.test_slate import fake_models


def _schedule_from_logs(game_logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """A LeagueSchedule-shaped frame for the fixture's own games."""
    abbrev_to_id = {}
    for tid, _, name in TEAMS:
        abbrev_to_id[name] = tid
    games = (
        game_logs[game_logs["season"] == season]
        .groupby("game_id")
        .agg(
            game_date=("game_date", "first"),
            home=("team", lambda s: s[game_logs.loc[s.index, "home"]].iloc[0]),
            away=("team", lambda s: s[~game_logs.loc[s.index, "home"]].iloc[0]),
        )
        .reset_index()
    )
    by_abbrev = (
        game_logs.drop_duplicates("team").set_index("team")["player_name"].str.split().str[-1]
    )
    rows = []
    for g in games.itertuples(index=False):
        rows.append(
            {
                "gameId": int(g.game_id),
                "gameDateTimeEst": f"{g.game_date.date()} 19:30:00",
                "homeTeamId": abbrev_to_id[by_abbrev[g.home]],
                "awayTeamId": abbrev_to_id[by_abbrev[g.away]],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sched(game_logs: pd.DataFrame, kaggle_dir: Path, tmp_path: Path) -> pd.DataFrame:
    frame = _schedule_from_logs(game_logs, "2025-26")
    path = tmp_path / schedule.schedule_file_name("2025-26")
    frame.to_csv(path, index=False)
    return schedule.load_schedule(path, kaggle_dump.load_team_histories(kaggle_dir))


def test_replay_reproduces_holdout_population_and_checks_tolerance(
    game_logs: pd.DataFrame, sched: pd.DataFrame
) -> None:
    models = fake_models()
    combined, per_date = replay.replay_season("2025-26", game_logs, sched, models, "rev-d")
    n_dates = game_logs.loc[game_logs["season"] == "2025-26", "game_date"].nunique()
    assert len(per_date["dates"]) == n_dates
    # Every replayed prediction has an actual row: the fixture's rosters never change.
    assert combined["has_actual"].all()
    # Predictions use only prior games: the fake model returns the last-10 mean, and the
    # residual equals actual minus that mean computed from strictly earlier games.
    row = combined.iloc[-1]
    hist = game_logs[
        (game_logs["player_id"] == row["player_id"])
        & (game_logs["game_date"] < pd.Timestamp(row["date"]))
    ]
    assert row["pred_pts"] == pytest.approx(hist.sort_values("game_date").tail(10)["pts"].mean())

    # Reference metrics equal to the replayed restricted MAE pass; shifted ones fail.
    restricted = combined[combined["in_metrics_population"]]
    mae = replay._mae(restricted)
    metrics = {
        "git_sha": "abc",
        "dataset": {"version": "x"},
        "metrics": {t: {"model": {"mae": mae[t], "n": len(restricted)}} for t in config.TARGETS},
    }
    report = replay.summarize("2025-26", combined, per_date, game_logs, metrics, models, "rev-d")
    assert report["passed"] is True
    assert report["n_restricted"] == len(restricted)
    assert report["mae_unrestricted"]["pts"] == pytest.approx(combined["resid_pts"].abs().mean())
    assert report["unpredicted_actual_rows"]["all"] == 0
    shifted = {
        **metrics,
        "metrics": {t: {"model": {"mae": mae[t] + 0.5, "n": 1}} for t in config.TARGETS},
    }
    assert (
        replay.summarize("2025-26", combined, per_date, game_logs, shifted, models, "rev-d")[
            "passed"
        ]
        is False
    )


def test_replay_counts_unpredicted_debuts(game_logs: pd.DataFrame, sched: pd.DataFrame) -> None:
    # A player who appears for the first time in the season's last game is never rostered.
    last_game = game_logs[game_logs["season"] == "2025-26"].sort_values("game_date").iloc[-1]
    debut = last_game.copy()
    debut["player_id"] = 777777
    debut["player_name"] = "New Guy"
    debut["minutes"] = 30.0
    logs = pd.concat([game_logs, debut.to_frame().T], ignore_index=True)
    from nba import schema

    logs = schema.validate(schema.coerce(logs))
    combined, per_date = replay.replay_season("2025-26", logs, sched, fake_models(), "rev-d")
    counts = replay.unpredicted_actual_rows(logs, per_date["season_game_ids"], combined)
    assert counts == {"all": 1, "minutes_ge_min": 1}


def test_cli_writes_report_and_exit_code(
    game_logs: pd.DataFrame,
    sched: pd.DataFrame,
    kaggle_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from nba.predict import model as model_module

    data_dir = tmp_path / "game_logs"
    local.write_per_season(game_logs, data_dir)
    sched_dir = tmp_path / "dump"
    sched_dir.mkdir()
    (sched_dir / kaggle_dump.TEAM_HISTORY_FILE).write_bytes(
        (kaggle_dir / kaggle_dump.TEAM_HISTORY_FILE).read_bytes()
    )
    _schedule_from_logs(game_logs, "2025-26").to_csv(
        sched_dir / schedule.schedule_file_name("2025-26"), index=False
    )
    monkeypatch.setattr(model_module, "load_from_hub", fake_models)
    combined, _ = replay.replay_season("2025-26", game_logs, sched, fake_models(), "x")
    mae = replay._mae(combined[combined["in_metrics_population"]])
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "git_sha": "abc",
                "dataset": {},
                "metrics": {t: {"model": {"mae": mae[t], "n": 1}} for t in config.TARGETS},
            }
        )
    )
    out = tmp_path / "replay.json"
    rc = replay.main(
        [
            "--season",
            "2025-26",
            "--schedule-dir",
            str(sched_dir),
            "--data-dir",
            str(data_dir),
            "--metrics",
            str(metrics_path),
            "--out",
            str(out),
            "--no-pull",
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["passed"] is True and report["season"] == "2025-26"
    assert "REPLAY PASSED" in capsys.readouterr().out
    rc = replay.main(
        [
            "--season",
            "2025-26",
            "--schedule-dir",
            str(sched_dir),
            "--data-dir",
            str(data_dir),
            "--metrics",
            str(metrics_path),
            "--out",
            str(out),
            "--tolerance",
            "0",
            "--no-pull",
        ]
    )
    assert rc == 0  # exact equality still passes at tolerance 0
