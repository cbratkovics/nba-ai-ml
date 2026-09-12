import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nba import config
from nba.agent import tools
from nba.storage import local
from tests.test_residuals import _predictions_from_actuals


@pytest.fixture
def agent_root(tmp_path: Path, game_logs: pd.DataFrame) -> tuple[tools.ToolContext, dict]:
    """Products laid out as pull_products leaves them, built from the fixture game logs."""
    from nba.predict import residuals

    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    # Dates with enough rows for the prediction helper (the Cup fixture games have four).
    counts = game_logs.groupby("game_date").size()
    dates = sorted(pd.Timestamp(d) for d in counts[counts >= 8].index)
    # Nightly residual file for the last game date, replay residual files for two earlier ones.
    last = dates[-1]
    r_last = residuals.compute(
        last.date(),
        _predictions_from_actuals(game_logs, last),
        game_logs,
        root / config.RESIDUALS_DIR,
    )
    replay_dir = root / config.REPLAY_DIR / config.HOLDOUT_SEASON / "residuals"
    for d in dates[-3:-1]:
        residuals.compute(d.date(), _predictions_from_actuals(game_logs, d), game_logs, replay_dir)
    # Replay daily MAE for the whole fixture season, used as the rolling fallback.
    daily = {
        "season": config.HOLDOUT_SEASON,
        "population": "fixture",
        "days": [
            {
                "date": d.date().isoformat(),
                "n": 4,
                "model": {"pts": 2.0, "reb": 1.0, "ast": 0.5},
                "baseline_last10": {"pts": 3.0, "reb": 1.5, "ast": 0.7},
            }
            for d in dates
        ],
    }
    (root / config.REPLAY_DIR / config.HOLDOUT_SEASON / "daily_mae.json").write_text(
        json.dumps(daily)
    )
    # A daily report for the run date.
    run_date = last.date()
    report = {
        "date": run_date.isoformat(),
        "stored_rows": len(game_logs),
        "stored_max_game_date": last.date().isoformat(),
        "window_start_exclusive": (last - pd.Timedelta(days=7)).date().isoformat(),
        "window_rows_in_dump": 12,
        "window_rows_after_rules": 10,
        "dnp_dropped_in_window": 2,
        "counts": {"new": 4, "changed": 1, "unchanged": 5},
        "changed_examples": [
            {
                "player_id": 1000,
                "game_id": "x",
                "game_date": "2026-01-01",
                "player_name": "P",
                "fields": {"pts": {"stored": 10, "incoming": 12}},
            }
        ],
        "seasons_written": ["2025-26"],
        "pushed": True,
        "dataset_revision_before": "a",
        "dataset_revision_after": "b",
        "schedule_file": "LeagueSchedule25_26.csv",
    }
    (root / config.DAILY_REPORTS_DIR).mkdir()
    (root / config.DAILY_REPORTS_DIR / f"{run_date.isoformat()}.json").write_text(
        json.dumps(report)
    )
    ctx = tools.ToolContext(root=root, data_dir=root / config.DATA_DIR, run_date=run_date)
    return ctx, {"dates": dates, "last_line": r_last.line, "report": report}


def _roundtrip(value):
    return json.loads(json.dumps(value))


def test_daily_report(agent_root) -> None:
    ctx, info = agent_root
    out = _roundtrip(tools.get_daily_report(ctx, ctx.run_date.isoformat()))
    assert out["available"] is True and out["counts"] == {"new": 4, "changed": 1, "unchanged": 5}
    assert out["pushed"] is True and out["changed_examples"][0]["fields"]["pts"]["incoming"] == 12
    assert tools.get_daily_report(ctx, "2020-01-01") == {
        "available": False,
        "date": "2020-01-01",
        "reason": "no daily report for this date",
    }


def test_upstream_freshness(agent_root, kaggle_dir: Path, game_logs: pd.DataFrame) -> None:
    ctx, info = agent_root
    stored_max = game_logs["game_date"].max().date()
    out = _roundtrip(tools.get_upstream_freshness(ctx))
    assert out["stored_max_game_date"] == stored_max.isoformat()
    assert out["days_stale"] == (ctx.run_date - stored_max).days
    assert out["dump_present"] is False and out["dump_max_game_date"] is None
    later = tools.ToolContext(
        root=ctx.root,
        data_dir=ctx.data_dir,
        run_date=stored_max.replace(year=stored_max.year + 1),
        dump_dir=kaggle_dir,
    )
    out = _roundtrip(tools.get_upstream_freshness(later))
    assert out["days_stale_stored"] == 365 and out["dump_present"] is True
    # The dump's newest game may be later than the stored max (e.g. an excluded Cup final).
    raw = pd.read_csv(kaggle_dir / "PlayerStatistics.csv")
    dump_max = pd.to_datetime(raw["gameDate"]).max().date()
    assert out["dump_max_game_date"] == dump_max.isoformat()
    assert out["newest_game_date"] == max(stored_max, dump_max).isoformat()
    assert out["days_stale"] == (later.run_date - max(stored_max, dump_max)).days


def test_residuals_top_n_and_counts(agent_root) -> None:
    ctx, info = agent_root
    d = ctx.run_date.isoformat()
    out = _roundtrip(tools.get_residuals(ctx, d, top_n=3))
    assert out["available"] and out["source"] == "nightly"
    assert out["n_predicted"] == info["last_line"]["n_predicted"]
    assert out["did_not_play"] == 0 and out["missing_actual_game_not_ingested"] == 0
    assert out["mae"]["pts"] == pytest.approx(2.0)
    top = out["largest_residuals"]["pts"]
    assert len(top) == 3 and all(abs(r["residual"]) == pytest.approx(2.0) for r in top)
    assert {
        "player_id",
        "player_name",
        "team",
        "opponent",
        "home",
        "minutes",
        "predicted",
        "actual",
        "residual",
    } <= set(top[0])
    # Replay fallback for an earlier date, and a clean miss for an unknown date.
    earlier = info["dates"][-2].date().isoformat()
    assert tools.get_residuals(ctx, earlier)["source"] == "replay"
    assert tools.get_residuals(ctx, "2020-01-01")["available"] is False
    assert (
        len(tools.get_residuals(ctx, d, top_n=999)["largest_residuals"]["reb"]) <= tools.MAX_TOP_N
    )


def test_rolling_metrics_from_residual_files_then_daily_mae(agent_root) -> None:
    ctx, info = agent_root
    out = _roundtrip(tools.get_rolling_metrics(ctx, days=30))
    assert out["available"] and out["source"] == "residual files"
    assert len(out["dates_covered"]) == 3
    assert out["model"]["pts"] == pytest.approx(2.0) and out["baseline_last10"]["pts"] is not None
    # A window before any residual file falls back to the replay daily MAE.
    early = tools.ToolContext(
        root=ctx.root, data_dir=ctx.data_dir, run_date=info["dates"][2].date()
    )
    out = _roundtrip(tools.get_rolling_metrics(early, days=3))
    assert out["source"] == "replay daily_mae.json" and out["model"]["pts"] == pytest.approx(2.0)
    assert out["baseline_last10"]["pts"] == pytest.approx(3.0)
    nothing = tools.ToolContext(root=ctx.root, data_dir=ctx.data_dir, run_date=date(2019, 1, 1))
    assert tools.get_rolling_metrics(nothing)["available"] is False


def test_player_recent_and_team_context(agent_root, game_logs: pd.DataFrame) -> None:
    ctx, info = agent_root
    pid = int(game_logs["player_id"].iloc[0])
    out = _roundtrip(tools.get_player_recent(ctx, pid, n=5))
    assert out["available"] and out["n"] == 5 and out["player_name"]
    assert [g["game_date"] for g in out["games"]] == sorted(g["game_date"] for g in out["games"])
    assert out["games"][-1]["game_date"] <= ctx.run_date.isoformat()
    assert tools.get_player_recent(ctx, 424242)["available"] is False

    team = str(game_logs["team"].iloc[0])
    out = _roundtrip(tools.get_team_context(ctx, team.lower(), ctx.run_date.isoformat()))
    assert out["available"] and out["team"] == team
    assert 1 <= len(out["window_games"]) <= config.ROSTER_LOOKBACK_GAMES
    assert all(g["game_date"] < ctx.run_date.isoformat() for g in out["window_games"])
    assert out["game_on_date"]["played"] is True and out["game_on_date"]["n_players"] > 0
    assert out["distinct_players_in_window"] >= 2
    assert tools.get_team_context(ctx, "ZZZ", ctx.run_date.isoformat())["available"] is False


def test_data_gaps(agent_root, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = agent_root
    monkeypatch.setattr(config, "FULL_SEASON_GAMES", 1230)
    out = _roundtrip(tools.list_data_gaps(ctx))
    assert len(out["known_missing_games"]) == 7
    assert all(s["games"] < 1230 for s in out["seasons_below_full"])
    assert {s["season"] for s in out["seasons_below_full"]} == {"2023-24", "2024-25", "2025-26"}


def test_run_tool_dispatch_and_errors(agent_root) -> None:
    ctx, _ = agent_root
    full = tools.run_tool(ctx, "list_data_gaps", None)["full_season_games"]
    assert full == config.FULL_SEASON_GAMES
    assert "unknown tool" in tools.run_tool(ctx, "nope", {})["error"]
    assert "bad arguments" in tools.run_tool(ctx, "get_residuals", {"nope": 1})["error"]
    assert "error" in tools.run_tool(ctx, "get_residuals", {"date": "not-a-date"})
    names = [t["function"]["name"] for t in tools.TOOL_SCHEMAS]
    assert names == list(tools.TOOLS)
    for schema in tools.TOOL_SCHEMAS:
        json.dumps(schema)
