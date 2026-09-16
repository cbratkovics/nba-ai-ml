"""Read-only tools for the analyst agent.

Every tool is a plain function over a `ToolContext` that points at local copies of the
published products (pulled from the Hugging Face dataset repo) and the stored game
logs. Tools never call the network and never write. Each returns a JSON-serializable
dict; errors are returned as {"error": ...} rather than raised, so a bad argument
never ends the agent loop.

`TOOL_SCHEMAS` holds the JSON-schema definitions handed to the model; `run_tool`
dispatches a call by name.

Sources (ADR-0019): where an exported gold mart is the better source the tool reads it
from <root>/gold/<mart>.parquet (the nightly job pushes the exports there and
pull_products brings them back) and says so in `source`; without the export it falls back
to the product files, computing the same numbers (the marts reconcile to those files by
test). The seven-tool contract is unchanged; a tool whose return shape grew carries
`tool_version` 2 and every version-1 key is still returned (tests/test_agent_tools.py).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.decisions import evaluate as policy_evaluate
from nba.decisions import policy as decision_policy
from nba.drift import calibrate as drift_calibrate
from nba.drift import check as drift_check
from nba.ingest import kaggle_dump
from nba.predict import residuals as residuals_module
from nba.storage import local

MAX_TOP_N = 25
MAX_RECENT = 25
MAX_DAYS = 120
ROLLING_DEFAULT_DAYS = 30
# Return-shape versions: 2 = keys were added (never removed) in Phase 4 (ADR-0019).
TOOL_RETURN_VERSIONS: dict[str, int] = {
    "get_daily_report": 2,
    "get_upstream_freshness": 2,
    "get_residuals": 1,
    "get_rolling_metrics": 2,
    "get_player_recent": 1,
    "get_team_context": 1,
    "list_data_gaps": 1,
}


@dataclass(frozen=True)
class ToolContext:
    """Where the tools read from. All paths are local; `run_date` is the day the agent runs."""

    root: Path  # holds predictions/, residuals/, replay/, daily_reports/, brief/
    data_dir: Path  # per-season game-log Parquet (pulled from HF)
    run_date: date
    dump_dir: Path | None = None  # extracted Kaggle dump, when the nightly downloaded one
    season: str = config.HOLDOUT_SEASON  # replay season used as fallback for residuals


# ---------- helpers ----------


def _json(value: Any) -> Any:
    """Convert pandas/numpy scalars and NaN to plain JSON values."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _parse_date(text: str) -> date:
    return date.fromisoformat(str(text)[:10])


def _game_logs(ctx: ToolContext) -> pd.DataFrame:
    return local.read_game_logs(ctx.data_dir)


def _mart(ctx: ToolContext, name: str) -> pd.DataFrame | None:
    """An exported gold mart pulled to <root>/gold/<name>.parquet, or None."""
    path = ctx.root / config.HF_GOLD_PREFIX / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _season_of(d: date) -> str:
    return drift_check.season_of(d)


def _residual_path(ctx: ToolContext, d: date) -> Path | None:
    nightly = ctx.root / config.RESIDUALS_DIR / f"{d.isoformat()}.parquet"
    if nightly.exists():
        return nightly
    replay = ctx.root / config.REPLAY_DIR / ctx.season / "residuals" / f"{d.isoformat()}.parquet"
    return replay if replay.exists() else None


def _residual_frames(ctx: ToolContext, start: date, end: date) -> list[tuple[date, pd.DataFrame]]:
    frames: list[tuple[date, pd.DataFrame]] = []
    d = start
    while d <= end:
        path = _residual_path(ctx, d)
        if path is not None:
            frames.append((d, pd.read_parquet(path)))
        d += timedelta(days=1)
    return frames


# ---------- tools ----------


def _drift_for(ctx: ToolContext, d: date) -> dict[str, Any]:
    """The latest drift verdict on or before d: mart_drift, else the drift/<date>.json
    products, else (replay dates) the committed calibration's per-date row."""
    mart = _mart(ctx, "mart_drift")
    if mart is not None and len(mart):
        mart = mart.copy()
        mart["run_date"] = pd.to_datetime(mart["run_date"]).dt.date
        rows = mart[mart["run_date"] <= d]
        if len(rows):
            latest = rows[rows["run_date"] == rows["run_date"].max()]
            run = latest.iloc[0]
            flagged = latest[latest["flagged"].fillna(False)].sort_values("psi", ascending=False)
            top = latest.dropna(subset=["psi"]).sort_values("psi", ascending=False).head(1)
            return _json(
                {
                    "available": True,
                    "as_of": run["run_date"].isoformat(),
                    "source": "mart_drift (gold export)",
                    "status": run["status"],
                    "position": run["position"],
                    "reference_mode": run["reference_mode"],
                    "n_rows": run["n_rows"],
                    "psi_threshold": run["psi_threshold"],
                    "calibrated": run["calibrated"],
                    "n_flagged": run["n_flagged"],
                    "flagged": flagged["feature"].tolist(),
                    "largest_feature": top["feature"].iloc[0] if len(top) else None,
                    "largest_psi": round(float(top["psi"].iloc[0]), 4) if len(top) else None,
                    "slate_status": run["slate_status"],
                    "no_schedule_streak": run["no_schedule_streak"],
                    "no_schedule_warn": run["no_schedule_warn"],
                }
            )
    drift_dir = ctx.root / config.DRIFT_DIR
    if drift_dir.is_dir():
        dated = []
        for path in drift_dir.glob("*.json"):
            try:
                if date.fromisoformat(path.stem) <= d:
                    dated.append((path.stem, path))
            except ValueError:
                continue
        if dated:
            _, path = max(dated)
            r = json.loads(path.read_text())
            feats = sorted(r.get("features", []), key=lambda f: -f["psi"])
            return _json(
                {
                    "available": True,
                    "as_of": r["date"],
                    "source": "drift report (drift/<date>.json)",
                    "status": r["status"],
                    "position": r["position"],
                    "reference_mode": r["reference_mode"],
                    "n_rows": r["window"]["n_rows"],
                    "psi_threshold": r["thresholds"]["psi"],
                    "calibrated": r["thresholds"]["calibrated"],
                    "n_flagged": r["n_flagged"],
                    "flagged": r["flagged"],
                    "largest_feature": feats[0]["feature"] if feats else None,
                    "largest_psi": round(feats[0]["psi"], 4) if feats else None,
                    "slate_status": r["slate_status"],
                    "no_schedule_streak": r["no_schedule_streak"],
                    "no_schedule_warn": r["no_schedule_warn"],
                }
            )
    cal = drift_calibrate.load_calibration()
    if cal is not None:
        row = next((p for p in cal["per_date"] if p["date"] == d.isoformat()), None)
        if row is not None:
            chosen = str(cal["chosen"]["threshold"])
            n_flagged = row["n_flagged"].get(chosen, 0)
            if row["max_psi"] is None:
                status = "insufficient"
            elif n_flagged >= cal["chosen"]["min_features"]:
                status = "hold"
            elif n_flagged:
                status = "warn"
            else:
                status = "ok"
            return _json(
                {
                    "available": True,
                    "as_of": row["date"],
                    "source": "drift calibration per-date row (replay season)",
                    "status": status,
                    "position": row["position"],
                    "reference_mode": row["reference_mode"],
                    "n_rows": row["n_rows"],
                    "psi_threshold": cal["chosen"]["threshold"],
                    "calibrated": True,
                    "n_flagged": n_flagged,
                    "flagged": [],
                    "largest_feature": row["max_feature"],
                    "largest_psi": round(row["max_psi"], 4) if row["max_psi"] is not None else None,
                    "slate_status": "ok",
                    "no_schedule_streak": 0,
                    "no_schedule_warn": False,
                }
            )
    return {"available": False, "as_of": None, "reason": "no drift report on or before this date"}


def _restatement_for(ctx: ToolContext, d: date) -> dict[str, Any]:
    """The latest restatement-lag row on or before d from mart_restatement_lag."""
    mart = _mart(ctx, "mart_restatement_lag")
    if mart is None or not len(mart):
        return {"available": False, "reason": "mart_restatement_lag is not exported"}
    mart = mart.copy()
    mart["report_date"] = pd.to_datetime(mart["report_date"]).dt.date
    rows = mart[mart["report_date"] <= d]
    if not len(rows):
        return {"available": False, "reason": "no restatement row on or before this date"}
    r = rows.sort_values("report_date").iloc[-1]
    return _json(
        {
            "available": True,
            "source": "mart_restatement_lag (gold export)",
            "report_date": r["report_date"].isoformat(),
            "n_changed": r["n_changed"],
            "n_new": r["n_new"],
            "max_lag_days": r["max_lag_days"],
            "p50_lag_days": r["p50_lag_days"],
            "p90_lag_days": r["p90_lag_days"],
            "lookback_days": r["lookback_days"],
            "within_lookback": r["within_lookback"],
        }
    )


def get_daily_report(ctx: ToolContext, date: str) -> dict[str, Any]:
    """Ingest counts, changed rows and push status for one nightly run date, plus the
    latest drift verdict and restatement lag on or before it (version 2)."""
    d = _parse_date(date)
    base = _daily_report_base(ctx, d)
    return {
        **base,
        "drift": _drift_for(ctx, d),
        "restatement": _restatement_for(ctx, d),
        "tool_version": TOOL_RETURN_VERSIONS["get_daily_report"],
    }


def _daily_report_base(ctx: ToolContext, d: date) -> dict[str, Any]:
    candidates = [
        ctx.root / config.DAILY_REPORTS_DIR / f"{d.isoformat()}.json",
        config.DAILY_REPORT_PATH if ctx.root == Path(".") else ctx.root / config.DAILY_REPORT_PATH,
    ]
    for path in candidates:
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("date") != d.isoformat():
                continue
            return {
                "available": True,
                "date": report["date"],
                "stored_rows": report.get("stored_rows"),
                "stored_max_game_date": report.get("stored_max_game_date"),
                "window_start_exclusive": report.get("window_start_exclusive"),
                "window_rows_in_dump": report.get("window_rows_in_dump"),
                "window_rows_after_rules": report.get("window_rows_after_rules"),
                "dnp_dropped_in_window": report.get("dnp_dropped_in_window"),
                "counts": report.get("counts"),
                "changed_examples": report.get("changed_examples", [])[:5],
                "seasons_written": report.get("seasons_written"),
                "pushed": report.get("pushed"),
                "dataset_revision_before": report.get("dataset_revision_before"),
                "dataset_revision_after": report.get("dataset_revision_after"),
                "schedule_file": report.get("schedule_file"),
            }
    return {"available": False, "date": d.isoformat(), "reason": "no daily report for this date"}


def get_upstream_freshness(ctx: ToolContext) -> dict[str, Any]:
    """Newest game date stored (and in the dump, if present) versus the run date."""
    logs = _game_logs(ctx)
    stored_max = logs["game_date"].max().date()
    out: dict[str, Any] = {
        "run_date": ctx.run_date.isoformat(),
        "stored_max_game_date": stored_max.isoformat(),
        "stored_rows": int(len(logs)),
        "stored_seasons": sorted(logs["season"].unique().tolist()),
        "days_stale_stored": (ctx.run_date - stored_max).days,
    }
    dump_max = None
    dump_max_any = None
    excluded = 0
    if ctx.dump_dir is not None and (ctx.dump_dir / kaggle_dump.BOX_SCORE_FILE).exists():
        window_start = pd.Timestamp(stored_max) - pd.Timedelta(days=config.DAILY_LOOKBACK_DAYS)
        box = kaggle_dump.load_box_scores(ctx.dump_dir, start=window_start, allow_empty=True)
        if not box.empty:
            dump_max_any = box["gameDate"].max().date()
            # The regular-season rules the backfill applies (AUDIT.md risk 9): playoff and
            # play-in rows and the Cup final never count as the newest game.
            regular = box[box["gameType"].astype("string").str.strip().isin(config.GAME_TYPES)]
            regular = regular[~kaggle_dump.is_cup_final(regular["gameId"], regular["gameSubLabel"])]
            excluded = int(len(box) - len(regular))
            if not regular.empty:
                dump_max = regular["gameDate"].max().date()
    out["dump_present"] = ctx.dump_dir is not None
    out["dump_max_game_date"] = dump_max.isoformat() if dump_max else None
    out["dump_max_game_date_any_type"] = dump_max_any.isoformat() if dump_max_any else None
    out["dump_rows_excluded_by_rules"] = excluded
    out["rules_applied"] = (
        f"gameType in {list(config.GAME_TYPES)} and not the Cup final, as the backfill does"
    )
    out["tool_version"] = TOOL_RETURN_VERSIONS["get_upstream_freshness"]
    newest = max(stored_max, dump_max) if dump_max else stored_max
    out["newest_game_date"] = newest.isoformat()
    days = (ctx.run_date - newest).days
    out["days_stale"] = days
    out["note"] = (
        f"newest game is {days} days before the run date"
        if days >= 0
        else f"stored data extends {-days} days past the run date (replay or backfilled data)"
    )
    return out


def get_residuals(ctx: ToolContext, date: str, top_n: int = 5) -> dict[str, Any]:
    """Largest absolute residuals per target for one date, plus missing-actual counts."""
    d = _parse_date(date)
    top_n = max(1, min(int(top_n), MAX_TOP_N))
    path = _residual_path(ctx, d)
    if path is None:
        return {
            "available": False,
            "date": d.isoformat(),
            "reason": "no residual file for this date",
        }
    df = pd.read_parquet(path)
    with_actuals = df[df["has_actual"]]
    missing = df[~df["has_actual"]]
    out: dict[str, Any] = {
        "available": True,
        "date": d.isoformat(),
        "source": "nightly" if path.parent == ctx.root / config.RESIDUALS_DIR else "replay",
        "n_predicted": int(len(df)),
        "n_with_actuals": int(len(with_actuals)),
        "did_not_play": int(missing["game_ingested"].sum()),
        "missing_actual_game_not_ingested": int((~missing["game_ingested"]).sum()),
        "mae": residuals_module.mae(with_actuals),
        "largest_residuals": {},
    }
    for t in config.TARGETS:
        cols = ["player_id", "player_name", "team", "opponent", "home", "minutes"]
        ordered = with_actuals.assign(abs_resid=with_actuals[f"resid_{t}"].abs()).sort_values(
            "abs_resid", ascending=False
        )
        rows = []
        for r in ordered.head(top_n).itertuples(index=False):
            rows.append(
                {
                    **{c: getattr(r, c) for c in cols},
                    "predicted": round(float(getattr(r, f"pred_{t}")), 2),
                    "actual": getattr(r, f"actual_{t}"),
                    "residual": round(float(getattr(r, f"resid_{t}")), 2),
                }
            )
        out["largest_residuals"][t] = rows
    return _json(out)


def _decisions_to_date(ctx: ToolContext, through: date) -> dict[str, Any]:
    """The decision policy's called rows and hit rate to date (training-population policy,
    the one the model beats both baselines on): fct_decision_policy when exported, else
    the residual files with the committed policy artifact (the same rule, ADR-0015)."""
    season = _season_of(through)
    mart = _mart(ctx, "fct_decision_policy")
    if mart is not None and len(mart):
        m = mart[(mart["population"] == "min10") & mart["has_actual"]].copy()
        m["game_date"] = pd.to_datetime(m["game_date"]).dt.date
        m = m[m["game_date"] <= through]
        if season in set(m["season"]):
            m = m[m["season"] == season]
        elif ctx.season in set(m["season"]):
            m, season = m[m["season"] == ctx.season], ctx.season
        if len(m) and m["decision"].notna().any():
            per: dict[str, Any] = {}
            for t in config.TARGETS:
                rows = m[m["target"] == t]
                called = rows[rows["decision"].isin(["over", "under"])]
                hit = int((called["outcome"] == "hit").sum())
                miss = int((called["outcome"] == "miss").sum())
                per[t] = {
                    "threshold": float(rows["threshold"].dropna().iloc[0]) if len(rows) else None,
                    "n_called": int(len(called)),
                    "n_resolved": hit + miss,
                    "n_push": int((called["outcome"] == "push").sum()),
                    "n_hit": hit,
                    "hit_rate": round(hit / (hit + miss), 4) if hit + miss else None,
                }
            return _json(
                {
                    "available": True,
                    "source": "fct_decision_policy (gold export)",
                    "population": "min10",
                    "season": season,
                    "through": through.isoformat(),
                    "first_date": min(m["game_date"]).isoformat(),
                    **per,
                }
            )
    artifact_path = config.REPO_ROOT / policy_evaluate.report_path(ctx.season)
    if not artifact_path.exists():
        return {"available": False, "reason": "no decisions mart and no policy artifact"}
    artifact = json.loads(artifact_path.read_text())
    season_start = date(int(season[:4]), 10, 1)
    frames = _residual_frames(ctx, season_start, through)
    if not frames:
        return {"available": False, "reason": "no residual files in the season through the date"}
    allres = pd.concat([f for _, f in frames], ignore_index=True)
    rows = allres[allres["has_actual"] & allres["in_metrics_population"]]
    per = {}
    for t in config.TARGETS:
        tb = artifact["populations"]["min10"]["targets"][t]
        edge = rows[f"pred_{t}"] - rows[f"{t}_mean_last10"]
        calls = decision_policy.calls(edge, tb["threshold"])
        outcomes = decision_policy.outcomes(calls, rows[f"actual_{t}"], rows[f"{t}_mean_last10"])
        hit, miss = int((outcomes == "hit").sum()), int((outcomes == "miss").sum())
        per[t] = {
            "threshold": tb["threshold"],
            "n_called": int(calls.isin(["over", "under"]).sum()),
            "n_resolved": hit + miss,
            "n_push": int((outcomes == "push").sum()),
            "n_hit": hit,
            "hit_rate": round(hit / (hit + miss), 4) if hit + miss else None,
        }
    return _json(
        {
            "available": True,
            "source": f"residual files + {policy_evaluate.report_path(ctx.season).as_posix()}",
            "population": "min10",
            "season": season,
            "through": through.isoformat(),
            "first_date": frames[0][0].isoformat(),
            **per,
        }
    )


def _rolling_from_mart(
    ctx: ToolContext, start: date, end: date, days: int
) -> dict[str, Any] | None:
    mart = _mart(ctx, "mart_daily_metrics")
    if mart is None or not len(mart):
        return None
    m = mart[mart["population"] == "all"].copy()
    m["game_date"] = pd.to_datetime(m["game_date"]).dt.date
    m = m[(m["game_date"] >= start) & (m["game_date"] <= end)]
    if not len(m):
        return None
    # A date scored both nightly and in the replay keeps the nightly row.
    m = m.sort_values("run_kind", ascending=False).drop_duplicates(["game_date", "target"])
    n_by_target = m.groupby("target")["n"].sum()
    n = int(n_by_target.min())
    model = {
        t: float((g["model_mae"] * g["n"]).sum() / g["n"].sum()) for t, g in m.groupby("target")
    }
    baseline = {
        t: float((g["baseline_last10_mae"] * g["n"]).sum() / g["n"].sum())
        for t, g in m.groupby("target")
    }
    return _json(
        {
            "available": True,
            "source": "mart_daily_metrics (gold export)",
            "days_requested": days,
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "n_dates_covered": int(m["game_date"].nunique()),
            "n": n,
            "population": "every row with a box score and a last-10 baseline (all)",
            "model": {t: model.get(t) for t in config.TARGETS},
            "baseline_last10": {t: baseline.get(t) for t in config.TARGETS},
        }
    )


def get_rolling_metrics(ctx: ToolContext, days: int = ROLLING_DEFAULT_DAYS) -> dict[str, Any]:
    """MAE per target for the model and the last-10 baseline over the last `days` days,
    plus the decision policy's hit rate to date (version 2)."""
    out = _rolling_metrics_base(ctx, days)
    out["decisions"] = _decisions_to_date(ctx, ctx.run_date)
    out["tool_version"] = TOOL_RETURN_VERSIONS["get_rolling_metrics"]
    return out


def _rolling_metrics_base(ctx: ToolContext, days: int) -> dict[str, Any]:
    days = max(1, min(int(days), MAX_DAYS))
    end = ctx.run_date
    start = end - timedelta(days=days - 1)
    from_mart = _rolling_from_mart(ctx, start, end, days)
    if from_mart is not None:
        return from_mart
    frames = _residual_frames(ctx, start, end)
    if frames:
        allres = pd.concat([f for _, f in frames], ignore_index=True)
        rows = allres[allres["has_actual"]]
        base_cols = [f"{t}_mean_last10" for t in config.TARGETS]
        rows = rows[rows[base_cols].notna().all(axis=1)]
        model = {
            t: float(rows[f"resid_{t}"].abs().mean()) if len(rows) else None for t in config.TARGETS
        }
        baseline = {
            t: float((rows[f"actual_{t}"] - rows[f"{t}_mean_last10"]).abs().mean())
            if len(rows)
            else None
            for t in config.TARGETS
        }
        return _json(
            {
                "available": True,
                "source": "residual files",
                "days_requested": days,
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "n_dates_covered": len(frames),
                "n": int(len(rows)),
                "population": "rows with actuals and a last-10 baseline for every target",
                "model": model,
                "baseline_last10": baseline,
            }
        )
    daily_path = ctx.root / config.REPLAY_DIR / ctx.season / "daily_mae.json"
    if daily_path.exists():
        daily = json.loads(daily_path.read_text())
        days_in = [x for x in daily["days"] if start.isoformat() <= x["date"] <= end.isoformat()]
        n = sum(x["n"] for x in days_in)
        if n:
            model = {t: sum(x["model"][t] * x["n"] for x in days_in) / n for t in config.TARGETS}
            baseline = {
                t: sum(x["baseline_last10"][t] * x["n"] for x in days_in) / n
                for t in config.TARGETS
            }
            return {
                "available": True,
                "source": "replay daily_mae.json",
                "days_requested": days,
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "n_dates_covered": len(days_in),
                "n": int(n),
                "population": daily.get("population"),
                "model": model,
                "baseline_last10": baseline,
            }
    return {
        "available": False,
        "days_requested": days,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "reason": "no residual files or replay daily MAE in the window",
    }


def get_player_recent(ctx: ToolContext, player_id: int, n: int = 10) -> dict[str, Any]:
    """The player's last `n` game lines on or before the run date."""
    n = max(1, min(int(n), MAX_RECENT))
    logs = _game_logs(ctx)
    rows = logs[
        (logs["player_id"] == int(player_id)) & (logs["game_date"] <= pd.Timestamp(ctx.run_date))
    ]
    if rows.empty:
        return {
            "available": False,
            "player_id": int(player_id),
            "reason": "no game logs for this player",
        }
    rows = rows.sort_values(["game_date", "game_id"]).tail(n)
    cols = ["game_date", "game_id", "team", "opponent", "home", "minutes", "pts", "reb", "ast"]
    return _json(
        {
            "available": True,
            "player_id": int(player_id),
            "player_name": str(rows["player_name"].iloc[-1]),
            "n": int(len(rows)),
            "games": rows[cols].to_dict("records"),
            "means": {t: float(rows[t].mean()) for t in ("minutes", "pts", "reb", "ast")},
        }
    )


def get_team_context(ctx: ToolContext, team: str, date: str) -> dict[str, Any]:
    """The team's games in the ten games before the date, with player counts, and any game on it."""
    d = _parse_date(date)
    logs = _game_logs(ctx)
    team = str(team).upper()
    mine = logs[logs["team"] == team]
    if mine.empty:
        return {"available": False, "team": team, "reason": "no game logs for this team"}
    before = mine[mine["game_date"] < pd.Timestamp(d)]
    games = (
        before.groupby(["game_id", "game_date", "opponent", "home"], sort=False)
        .size()
        .rename("n_players")
        .reset_index()
        .sort_values(["game_date", "game_id"])
        .tail(config.ROSTER_LOOKBACK_GAMES)
    )
    on_date = mine[mine["game_date"] == pd.Timestamp(d)]
    return _json(
        {
            "available": True,
            "team": team,
            "date": d.isoformat(),
            "window_games": games.to_dict("records"),
            "distinct_players_in_window": int(
                before[before["game_id"].isin(games["game_id"])]["player_id"].nunique()
            ),
            "game_on_date": {
                "played": not on_date.empty,
                "game_ids": sorted(on_date["game_id"].unique().tolist()),
                "n_players": int(len(on_date)),
            },
        }
    )


def list_data_gaps(ctx: ToolContext) -> dict[str, Any]:
    """Known missing games and seasons short of a full 1,230-game schedule."""
    logs = _game_logs(ctx)
    per_season = logs.groupby("season")["game_id"].nunique()
    short = [
        {"season": s, "games": int(g), "expected": config.FULL_SEASON_GAMES}
        for s, g in per_season.items()
        if g < config.FULL_SEASON_GAMES
    ]
    return {
        "known_missing_count": len(kaggle_dump.KNOWN_MISSING_GAMES),
        "known_missing_games": list(kaggle_dump.KNOWN_MISSING_GAMES),
        "seasons_below_full": short,
        "full_season_games": config.FULL_SEASON_GAMES,
    }


# ---------- registry and schemas ----------

TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_daily_report": get_daily_report,
    "get_upstream_freshness": get_upstream_freshness,
    "get_residuals": get_residuals,
    "get_rolling_metrics": get_rolling_metrics,
    "get_player_recent": get_player_recent,
    "get_team_context": get_team_context,
    "list_data_gaps": list_data_gaps,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_daily_report",
            "description": (
                "Ingest counts (new/changed/unchanged rows), changed-row examples, and push "
                "status for one nightly run date; plus `drift` (the latest drift verdict on "
                "or before the date: status ok/warn/hold/insufficient, flagged features, the "
                "no-schedule streak) and `restatement` (observed restatement lag)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upstream_freshness",
            "description": (
                "Newest regular-season game date stored (and in the downloaded dump if "
                "present, after the regular-season rules) versus the run date, as days stale."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_residuals",
            "description": (
                "For one date: MAE per target, the largest absolute residuals per target with "
                "predicted/actual/minutes, and counts of players who did not play or whose "
                "game is not ingested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "top_n": {"type": "integer", "description": "1-25, default 5"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rolling_metrics",
            "description": (
                "MAE per target for the model and the last-10-game baseline over the last N "
                "days ending on the run date; plus `decisions`, the decision policy's called "
                "rows and hit rate to date per target on the training population."
            ),
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "1-120, default 30"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_recent",
            "description": (
                "A player's last n game lines (date, team, opponent, minutes, pts, reb, ast) "
                "on or before the run date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "n": {"type": "integer", "description": "1-25, default 10"},
                },
                "required": ["player_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_context",
            "description": (
                "A team's last ten games before a date with player counts per game, and "
                "whether it played on that date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "three-letter abbreviation"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["team", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_data_gaps",
            "description": (
                "Known games missing from the data and seasons with fewer than 1,230 games."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(ctx: ToolContext, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Dispatch a tool call; any failure is returned as {"error": ...}."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return fn(ctx, **(args or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the loop
        return {"error": f"{type(exc).__name__}: {exc}"}
