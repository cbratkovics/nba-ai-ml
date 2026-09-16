import json
from datetime import date, timedelta
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
    missing = tools.get_daily_report(ctx, "2020-01-01")
    assert {k: missing[k] for k in ("available", "date", "reason")} == {
        "available": False,
        "date": "2020-01-01",
        "reason": "no daily report for this date",
    }
    # Version 2 blocks are always present, even when the report is not.
    assert missing["tool_version"] == 2 and missing["drift"]["available"] is False
    assert missing["restatement"]["available"] is False


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
    # The regular-season rules apply before the newest date is taken (AUDIT.md risk 9): the
    # fixture's playoff rows are excluded, so the newest regular-season date is the stored
    # max, while dump_max_game_date_any_type still shows the raw maximum.
    raw = pd.read_csv(kaggle_dir / "PlayerStatistics.csv")
    raw["gameDate"] = pd.to_datetime(raw["gameDate"])
    raw_max = raw["gameDate"].max().date()
    regular = raw[raw["gameType"].isin(config.GAME_TYPES)]
    regular_max = regular["gameDate"].max().date()
    assert raw_max >= regular_max
    assert out["dump_max_game_date_any_type"] == raw_max.isoformat()
    assert out["dump_max_game_date"] == regular_max.isoformat()
    window_start = pd.Timestamp(stored_max) - pd.Timedelta(days=config.DAILY_LOOKBACK_DAYS)
    in_window = raw[raw["gameDate"] >= window_start]
    assert out["dump_rows_excluded_by_rules"] == int(
        (~in_window["gameType"].isin(config.GAME_TYPES)).sum()
    )
    assert out["newest_game_date"] == max(stored_max, regular_max).isoformat()
    assert out["days_stale"] == (later.run_date - max(stored_max, regular_max)).days
    assert out["tool_version"] == 2 and "Cup final" in out["rules_applied"]


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
    assert out["n_dates_covered"] == 3
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


# ---------- version 2: marts, fallbacks, and the return-shape contract ----------

V1_KEYS: dict[str, set[str]] = {
    "get_daily_report": {"available", "date", "counts", "changed_examples", "pushed"},
    "get_upstream_freshness": {
        "run_date",
        "stored_max_game_date",
        "stored_rows",
        "stored_seasons",
        "days_stale_stored",
        "dump_present",
        "dump_max_game_date",
        "newest_game_date",
        "days_stale",
        "note",
    },
    "get_residuals": {"available", "date", "source", "n_predicted", "mae", "largest_residuals"},
    "get_rolling_metrics": {
        "available",
        "source",
        "days_requested",
        "window",
        "n_dates_covered",
        "n",
        "population",
        "model",
        "baseline_last10",
    },
    "get_player_recent": {"available", "player_id", "player_name", "n", "games", "means"},
    "get_team_context": {"available", "team", "date", "window_games", "game_on_date"},
    "list_data_gaps": {"known_missing_count", "known_missing_games", "seasons_below_full"},
}


def test_every_tool_keeps_its_version_1_keys(agent_root, game_logs) -> None:
    """Changing a tool's return shape is a versioned, additive change (ADR-0019)."""
    ctx, info = agent_root
    d = info["dates"][-1].date().isoformat()
    pid = int(game_logs["player_id"].iloc[0])
    team = str(game_logs["team"].iloc[0])
    calls = {
        "get_daily_report": {"date": ctx.run_date.isoformat()},
        "get_upstream_freshness": {},
        "get_residuals": {"date": d},
        "get_rolling_metrics": {"days": 30},
        "get_player_recent": {"player_id": pid},
        "get_team_context": {"team": team, "date": d},
        "list_data_gaps": {},
    }
    assert set(calls) == set(tools.TOOLS) == set(tools.TOOL_RETURN_VERSIONS)
    for name, args in calls.items():
        out = tools.run_tool(ctx, name, args)
        assert "error" not in out, (name, out)
        missing = V1_KEYS[name] - set(out)
        assert not missing, f"{name} lost version-1 keys {missing}"
        if tools.TOOL_RETURN_VERSIONS[name] > 1:
            assert out["tool_version"] == tools.TOOL_RETURN_VERSIONS[name]
        else:
            assert "tool_version" not in out


def _gold(ctx: tools.ToolContext, name: str, frame: pd.DataFrame) -> Path:
    folder = ctx.root / config.HF_GOLD_PREFIX
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_daily_report_drift_and_restatement_prefer_marts(agent_root) -> None:
    ctx, info = agent_root
    run = ctx.run_date
    rows = []
    for offset, status, feats in ((3, "ok", []), (1, "warn", ["days_rest"])):
        d = run - timedelta(days=offset)
        for f, psi, flag in (
            ("days_rest", 0.21, "days_rest" in feats),
            ("pts_mean_last5", 0.02, False),
        ):
            rows.append(
                {
                    "run_date": d,
                    "season": "2025-26",
                    "model_revision": "m",
                    "feature_version": "asof_v1",
                    "feature": f,
                    "psi": psi,
                    "flagged": flag,
                    "status": status,
                    "position": "regular",
                    "reference_mode": "day_050",
                    "n_rows": 900,
                    "psi_threshold": 0.15,
                    "min_features": 3,
                    "calibrated": True,
                    "n_flagged": len(feats),
                    "slate_status": "ok",
                    "no_schedule_streak": 0,
                    "no_schedule_warn": False,
                    "source_file": "x",
                }
            )
    _gold(ctx, "mart_drift", pd.DataFrame(rows))
    _gold(
        ctx,
        "mart_restatement_lag",
        pd.DataFrame(
            [
                {
                    "report_date": run - timedelta(days=2),
                    "n_changed": 3,
                    "n_new": 1,
                    "n_examples": 3,
                    "max_lag_days": 4,
                    "p50_lag_days": 2.0,
                    "p90_lag_days": 4.0,
                    "lookback_days": 14,
                    "within_lookback": True,
                    "stored_max_game_date": run.isoformat(),
                    "window_start_exclusive": run.isoformat(),
                }
            ]
        ),
    )
    out = _roundtrip(tools.get_daily_report(ctx, run.isoformat()))
    drift = out["drift"]
    assert drift["source"].startswith("mart_drift") and drift["available"]
    assert drift["as_of"] == (run - timedelta(days=1)).isoformat()  # latest on or before
    assert drift["status"] == "warn" and drift["flagged"] == ["days_rest"]
    assert drift["largest_feature"] == "days_rest" and drift["largest_psi"] == 0.21
    assert drift["n_flagged"] == 1 and drift["no_schedule_streak"] == 0
    rest = out["restatement"]
    assert rest["available"] and rest["max_lag_days"] == 4 and rest["within_lookback"] is True
    # Before both rows: no drift row on or before the date falls through to the products.
    early = _roundtrip(tools.get_daily_report(ctx, (run - timedelta(days=10)).isoformat()))
    assert (
        early["drift"]["available"] is False
        or early["drift"]["source"] != "mart_drift (gold export)"
    )


def test_daily_report_drift_falls_back_to_reports_then_calibration(agent_root) -> None:
    ctx, info = agent_root
    run = ctx.run_date
    drift_dir = ctx.root / config.DRIFT_DIR
    drift_dir.mkdir()
    report = {
        "date": (run - timedelta(days=1)).isoformat(),
        "position": "off_season",
        "reference_mode": "all",
        "window": {"n_rows": 0},
        "thresholds": {"psi": 0.15, "calibrated": True},
        "status": "insufficient",
        "n_flagged": 0,
        "flagged": [],
        "slate_status": "no_schedule",
        "no_schedule_streak": 6,
        "no_schedule_warn": False,
        "features": [],
    }
    (drift_dir / f"{report['date']}.json").write_text(json.dumps(report))
    out = _roundtrip(tools.get_daily_report(ctx, run.isoformat()))["drift"]
    assert out["source"].startswith("drift report") and out["status"] == "insufficient"
    assert out["no_schedule_streak"] == 6 and out["largest_feature"] is None
    # A replay date with no product falls back to the committed calibration's per-date row.
    cal = ctx.root.parent / "no_products"
    cal.mkdir()
    replay_ctx = tools.ToolContext(root=cal, data_dir=ctx.data_dir, run_date=date(2026, 3, 11))
    out = _roundtrip(tools.get_daily_report(replay_ctx, "2026-03-11"))["drift"]
    if out["available"]:
        assert out["source"].startswith("drift calibration") and out["calibrated"] is True
        assert out["status"] in ("ok", "warn", "hold", "insufficient")


def test_rolling_metrics_prefer_the_daily_mart_and_add_decisions(agent_root) -> None:
    ctx, info = agent_root
    d1, d2 = info["dates"][-2].date(), info["dates"][-1].date()
    rows = []
    for d, n, mae in ((d1, 10, 2.0), (d2, 30, 4.0)):
        for t in config.TARGETS:
            for pop in ("all", "min10"):
                rows.append(
                    {
                        "season": "2025-26",
                        "run_kind": "replay",
                        "model_revision": "m",
                        "feature_version": "asof_v1",
                        "game_date": d,
                        "population": pop,
                        "target": t,
                        "n": n if pop == "all" else n - 2,
                        "model_mae": mae,
                        "baseline_last10_mae": mae + 1,
                    }
                )
    _gold(ctx, "mart_daily_metrics", pd.DataFrame(rows))
    out = _roundtrip(tools.get_rolling_metrics(ctx, days=30))
    assert out["source"] == "mart_daily_metrics (gold export)" and out["tool_version"] == 2
    assert out["n"] == 40 and out["n_dates_covered"] == 2
    assert out["model"]["pts"] == pytest.approx((10 * 2.0 + 30 * 4.0) / 40)
    assert out["baseline_last10"]["pts"] == pytest.approx((10 * 3.0 + 30 * 5.0) / 40)
    # Decisions from the mart: two called rows, one hit, one miss, one push.
    dec_rows = []
    for i, (decision, outcome) in enumerate(
        (("over", "hit"), ("under", "miss"), ("over", "push"), ("no_call", None))
    ):
        for t in config.TARGETS:
            dec_rows.append(
                {
                    "player_id": i,
                    "game_id": f"g{i}",
                    "model_revision": "m",
                    "season": "2025-26",
                    "game_date": d2,
                    "population": "min10",
                    "target": t,
                    "threshold": 1.75,
                    "decision": decision,
                    "has_actual": True,
                    "outcome": outcome,
                }
            )
    _gold(ctx, "fct_decision_policy", pd.DataFrame(dec_rows))
    dec = _roundtrip(tools.get_rolling_metrics(ctx, days=30))["decisions"]
    assert dec["available"] and dec["source"] == "fct_decision_policy (gold export)"
    assert dec["population"] == "min10" and dec["season"] == "2025-26"
    assert dec["pts"] == {
        "threshold": 1.75,
        "n_called": 3,
        "n_resolved": 2,
        "n_push": 1,
        "n_hit": 1,
        "hit_rate": 0.5,
    }


def test_decisions_fall_back_to_residual_files_with_the_policy_artifact(agent_root) -> None:
    ctx, info = agent_root
    artifact = config.REPO_ROOT / "reports" / f"policy_{config.HOLDOUT_SEASON}.json"
    if not artifact.exists():
        pytest.skip("policy artifact not written yet")
    dec = _roundtrip(tools.get_rolling_metrics(ctx, days=30))["decisions"]
    assert dec["available"] and dec["source"].startswith("residual files + reports/policy_")
    for t in config.TARGETS:
        assert dec[t]["n_called"] == dec[t]["n_resolved"] + dec[t]["n_push"]
        assert dec[t]["hit_rate"] is None or 0 <= dec[t]["hit_rate"] <= 1
    nothing = tools.ToolContext(root=ctx.root, data_dir=ctx.data_dir, run_date=date(2019, 1, 1))
    assert tools.get_rolling_metrics(nothing)["decisions"]["available"] is False
