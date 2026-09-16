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
    # The daily report and the (unavailable) brief are always written, never pushed here.
    assert s.products_pushed == [] and "product files (push disabled)" in out
    assert (root / "daily_reports" / "2026-01-15.json").exists()


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
    assert {"residuals/2026-01-15.parquet", "predictions/rolling_metrics.json"} <= set(
        s2.products_pushed
    )
    assert "daily_reports/2026-01-16.json" in s2.products_pushed
    assert {"2026-01-15.parquet", "rolling_metrics.json"} <= set(pushed["files"])
    assert pushed["message"] == "Nightly 2026-01-16"
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


def test_agent_step_never_fails_the_night(
    root: Path, kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Without a provider key the brief is agent_unavailable; the night still completes."""
    from nba.agent import loop as agent_loop

    monkeypatch.setattr(config, "GROQ_API_KEY", None)
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
    assert s.agent["status"] == "agent_unavailable"
    assert "AGENT 2026-01-14: status=agent_unavailable" in out
    assert (root / "brief" / "2026-01-14.json").exists()
    assert (root / "brief" / "latest.json").exists()
    assert (root / "daily_reports" / "2026-01-15.json").exists()
    assert s.slate["status"] == "no_schedule"

    # An unexpected exception inside the agent is caught and logged, not raised.
    def boom(*a, **k):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(agent_loop, "run_and_write", boom)
    s = nightly.run(
        date(2026, 1, 16),
        push=False,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r2.json",
        load_models=fake_models,
    )
    assert s.agent["status"] == "agent_unavailable" and "agent exploded" in s.agent["error"]
    assert (
        "AGENT 2026-01-15: agent_unavailable (RuntimeError: agent exploded)"
        in capsys.readouterr().out
    )


def test_products_pushed_include_daily_report_and_brief(
    root: Path, kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    dump = _dump_with_schedule(kaggle_dir, tmp_path, games=None)
    pushed: dict = {}
    monkeypatch.setattr(hf, "pull_dataset", lambda data_dir: "rev")
    monkeypatch.setattr(hf, "pull_products", lambda r: "prod")
    monkeypatch.setattr(
        hf,
        "push_products",
        lambda r, files=None, message="": (
            pushed.update(files=[str(p.relative_to(r)) for p in files]) or "sha"
        ),
    )
    s = nightly.run(
        date(2026, 1, 15),
        push=True,
        root=root,
        local_dump=dump,
        report_path=tmp_path / "r.json",
        load_models=fake_models,
    )
    assert "daily_reports/2026-01-15.json" in pushed["files"]
    assert "brief/2026-01-14.json" in pushed["files"] and "brief/latest.json" in pushed["files"]
    assert s.products_revision == "sha"
    # The drift report is a product too (ADR-0018); the fixture window is too thin for PSI.
    assert "drift/2026-01-15.json" in pushed["files"]
    assert s.drift["status"] == "insufficient" and s.drift["blocks_slate"] is False
    assert s.drift["no_schedule_streak"] == 1
