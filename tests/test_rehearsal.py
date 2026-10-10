"""The rehearsal writes a synthetic game night, runs the nightly without pushing, stages its
products for the warehouse, and cannot write to Hugging Face (ADR-0028)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nba import config, rehearsal
from nba.ingest import kaggle_dump, schedule
from nba.storage import hf, local
from tests.test_slate import fake_models


def test_synthetic_schedule_redates_the_last_real_night(
    game_logs: pd.DataFrame, kaggle_dir: Path
) -> None:
    histories = kaggle_dump.load_team_histories(kaggle_dir)
    d = date(2026, 10, 20)
    sched = rehearsal.synthetic_schedule(game_logs, d, histories)
    last = game_logs[game_logs["game_date"] == game_logs["game_date"].max()]
    assert len(sched) == last["game_id"].nunique()
    assert list(sched.columns) == list(rehearsal.SCHEDULE_COLUMNS)
    assert (sched["gameDateTimeEst"] == "2026-10-20 19:30:00").all()
    assert sched["gameId"].astype(str).str.startswith("226").all()  # 0022600001 once zero-padded
    assert sched["homeTeamId"].ne(sched["awayTeamId"]).all()
    # The loader reads it back as 2026-27 regular-season games on that date.
    path = kaggle_dir / schedule.schedule_file_name("2026-27")
    sched.to_csv(path, index=False)
    loaded = schedule.load_schedule(path, histories)
    assert set(loaded["season"]) == {"2026-27"}
    assert len(schedule.games_on(loaded, d)) == len(sched)


def test_write_dump_gives_the_ingest_an_empty_window(
    game_logs: pd.DataFrame, kaggle_dir: Path, tmp_path: Path
) -> None:
    dump = tmp_path / "dump"
    info = rehearsal.write_dump(
        dump, game_logs, date(2026, 10, 20), kaggle_dir / kaggle_dump.TEAM_HISTORY_FILE
    )
    assert info["schedule_file"] == "LeagueSchedule26_27.csv" and info["n_games"] >= 1
    assert (dump / kaggle_dump.TEAM_HISTORY_FILE).exists()
    box = kaggle_dump.load_box_scores(dump, start=pd.Timestamp("2026-10-01"), allow_empty=True)
    assert box.empty and "gameDate" in box.columns
    assert schedule.schedule_path(dump, date(2026, 10, 20)) == dump / "LeagueSchedule26_27.csv"
    # The workflow downloads TeamHistories.csv into the dump directory itself (run 38008812416
    # failed on copying it onto itself).
    again = rehearsal.write_dump(
        dump, game_logs, date(2026, 10, 20), dump / kaggle_dump.TEAM_HISTORY_FILE
    )
    assert again["n_games"] == info["n_games"]


def test_run_writes_a_slate_without_pushing_and_stages_it(
    game_logs: pd.DataFrame, kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    for name in ("pull_dataset", "pull_products", "push_dataset", "push_products", "push_gold"):
        monkeypatch.setattr(
            hf, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} must not be called")
        )
    monkeypatch.setattr(config, "DAILY_REPORT_PATH", tmp_path / "daily_report.json")
    monkeypatch.setattr(config, "GROQ_API_KEY", "would-be-a-key")
    # run() sets the guard itself; set it here first so monkeypatch restores the environment
    # for the tests that follow (a leaked NBA_REHEARSAL would make every push test fail).
    monkeypatch.setenv(rehearsal.REHEARSAL_ENV, "1")
    summary = rehearsal.run(
        date(2026, 10, 20),
        root,
        tmp_path / "dump",
        kaggle_dir / kaggle_dump.TEAM_HISTORY_FILE,
        pull=False,
        load_models=fake_models,
        summary_path=tmp_path / "summary.json",
    )
    assert summary.slate["status"] == "ok" and summary.slate["n_players"] > 0
    assert summary.products_pushed == [] and summary.push is False
    assert summary.agent["status"] == "agent_unavailable"  # the rehearsal never calls Groq
    assert summary.drift["status"] == "insufficient" and summary.drift["no_schedule_streak"] == 0
    assert summary.lines[0].startswith("REHEARSAL 2026-10-20: no pushes")
    assert (root / "predictions" / "2026-10-20.parquet").exists()
    assert (
        root / "decisions" / "2026-10-20.json"
    ).exists()  # the committed policy artifact applies
    assert (tmp_path / "summary.json").exists()
    # The warehouse gate would build on this summary.
    from nba.warehouse import gate

    assert gate.decide(__import__("json").loads((tmp_path / "summary.json").read_text()))[0]

    wh = tmp_path / "warehouse"
    staged = rehearsal.stage_products(root, wh)
    names = sorted(p.relative_to(wh).as_posix() for p in staged)
    assert "predictions/2026-10-20.parquet" in names
    assert "daily_reports/2026-10-20.json" in names and "drift/2026-10-20.json" in names
    assert not any(n.startswith("decisions/") for n in names)  # not a warehouse source family


def test_scratch_database_helpers_refuse_the_real_warehouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create/drop only ever touch nba_rehearsal*; the real `nba` database is unreachable here."""
    executed: list[str] = []

    class Con:
        def execute(self, sql):
            executed.append(sql)

        def close(self):
            pass

    monkeypatch.setattr(rehearsal, "_motherduck", lambda token: Con())
    with pytest.raises(ValueError, match="refusing"):
        rehearsal.drop_scratch("nba", token="t")
    with pytest.raises(ValueError, match="refusing"):
        rehearsal.create_scratch("nba_prod", token="t")
    rehearsal.create_scratch("nba_rehearsal", token="t")
    rehearsal.drop_scratch("nba_rehearsal", token="t")
    assert executed == [
        "create database if not exists nba_rehearsal",
        "drop database if exists nba_rehearsal cascade",
    ]


def test_hugging_face_writes_are_refused_during_a_rehearsal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(rehearsal.REHEARSAL_ENV, "1")
    monkeypatch.setattr(config, "HF_TOKEN", "hf_fake")
    with pytest.raises(RuntimeError, match="writes are disabled"):
        hf.push_gold([Path("x.parquet")])
    with pytest.raises(RuntimeError, match="writes are disabled"):
        hf._api(require_token=True)
    hf._api(require_token=False)  # reads stay possible
