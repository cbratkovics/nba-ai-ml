import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nba import config, nightly
from nba.ingest import kaggle_dump, schedule
from nba.storage import hf, local
from tests.conftest import TEAMS
from tests.test_slate import fake_models


@pytest.fixture
def root(
    kaggle_dir: Path, tmp_path: Path, game_logs: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A working directory with stored game logs and no HF access."""
    r = tmp_path / "root"
    local.write_per_season(game_logs, r / config.DATA_DIR)
    for name in (
        "pull_dataset",
        "pull_products",
        "push_dataset",
        "push_products",
        "fetch_dataset_card",
    ):
        monkeypatch.setattr(
            hf, name, lambda *a, _name=name, **k: pytest.fail(f"{_name} must not be called")
        )
    return r


def _dump_with_schedule(kaggle_dir: Path, tmp_path: Path, games: list[dict] | None) -> Path:
    """Copy of the fixture dump; games=None means no schedule file at all."""
    d = tmp_path / "dump"
    d.mkdir()
    for name in (kaggle_dump.BOX_SCORE_FILE, kaggle_dump.TEAM_HISTORY_FILE):
        (d / name).write_bytes((kaggle_dir / name).read_bytes())
    if games is not None:
        pd.DataFrame(games).to_csv(d / schedule.schedule_file_name("2025-26"), index=False)
    return d


def test_no_schedule_path_is_explicit(root: Path, kaggle_dir: Path, tmp_path: Path, capsys) -> None:
    dump = _dump_with_schedule(kaggle_dir, tmp_path, games=None)
    s = nightly.run(
        date(2026, 1, 15),
        push=False,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r.json",
        load_models=fake_models,
    )
    out = capsys.readouterr().out
    assert s.slate["status"] == "no_schedule"
    assert "SLATE 2026-01-15: no schedule file for season 2025-26" in out
    assert "RESIDUALS 2026-01-14: no predictions file" in out
    assert s.products_pushed == [] and "no product files written" in out


def test_no_games_path_is_explicit_and_distinct(
    root: Path, kaggle_dir: Path, tmp_path: Path, capsys
) -> None:
    games = [
        {
            "gameId": 22500900,
            "gameDateTimeEst": "2026-01-20 19:30:00",
            "homeTeamId": TEAMS[0][0],
            "awayTeamId": TEAMS[1][0],
        }
    ]
    dump = _dump_with_schedule(kaggle_dir, tmp_path, games)
    s = nightly.run(
        date(2026, 1, 15),
        push=False,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r.json",
        load_models=fake_models,
    )
    out = capsys.readouterr().out
    assert s.slate["status"] == "no_games"
    assert "SLATE 2026-01-15: no games on this date" in out
    assert "no schedule file" not in out


def test_full_night_writes_slate_then_residuals_next_day(
    root: Path, kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    games = [
        {
            "gameId": 22500900,
            "gameDateTimeEst": "2026-01-15 19:30:00",
            "homeTeamId": TEAMS[0][0],
            "awayTeamId": TEAMS[1][0],
        }
    ]
    dump = _dump_with_schedule(kaggle_dir, tmp_path, games)
    s1 = nightly.run(
        date(2026, 1, 15),
        push=False,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r.json",
        load_models=fake_models,
    )
    assert s1.slate["status"] == "ok" and s1.slate["n_players"] == 4
    assert (root / "predictions" / "2026-01-15.parquet").exists()
    assert s1.residuals == {"status": "no_predictions"}

    # Next night: the game has "happened" (add actual rows for it), so residuals exist.
    logs = local.read_game_logs(root / config.DATA_DIR)
    played = (
        logs[logs["team"].isin(["LAL", "BOS"])]
        .sort_values("game_date")
        .groupby("player_id")
        .tail(1)
        .copy()
    )
    played["game_id"] = "0022500900"
    played["game_date"] = pd.Timestamp("2026-01-15")
    played["season"] = "2025-26"
    merged = pd.concat([logs, played], ignore_index=True)
    local.write_per_season(merged, root / config.DATA_DIR)
    pushed: dict = {}
    monkeypatch.setattr(hf, "pull_dataset", lambda data_dir: "rev-before")
    monkeypatch.setattr(hf, "pull_products", lambda r: "prod-before")
    monkeypatch.setattr(
        hf,
        "push_products",
        lambda r, files=None, message="": (
            pushed.update(files=[p.name for p in files], message=message) or "prod-after"
        ),
    )
    s2 = nightly.run(
        date(2026, 1, 16),
        push=True,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r2.json",
        load_models=fake_models,
    )
    assert s2.residuals["n_predicted"] == 4 and s2.residuals["n_with_actuals"] == 4
    assert s2.slate["status"] == "no_games"
    assert s2.products_pushed == [
        "residuals/2026-01-15.parquet",
        "predictions/rolling_metrics.json",
    ]
    assert (
        pushed["files"] == ["2026-01-15.parquet", "rolling_metrics.json"]
        and pushed["message"] == "Nightly 2026-01-16"
    )
    assert s2.products_revision == "prod-after"


def test_cli_writes_summary(
    root: Path, kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nba.predict import model as model_module

    monkeypatch.setattr(model_module, "load_from_hub", fake_models)
    dump = _dump_with_schedule(kaggle_dir, tmp_path, games=None)
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(config, "DAILY_REPORT_PATH", tmp_path / "daily_report.json")
    rc = nightly.main(
        [
            "--date",
            "2026-01-15",
            "--local-dump",
            str(dump),
            "--root",
            str(root),
            "--summary",
            str(summary),
        ]
    )
    assert rc == 0
    payload = json.loads(summary.read_text())
    assert payload["date"] == "2026-01-15" and payload["slate"]["status"] == "no_schedule"
    assert any("no schedule file" in ln for ln in payload["lines"])
