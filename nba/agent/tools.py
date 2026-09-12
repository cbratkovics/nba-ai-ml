"""Read-only tools for the analyst agent.

Every tool is a plain function over a `ToolContext` that points at local copies of the
published products (pulled from the Hugging Face dataset repo) and the stored game
logs. Tools never call the network and never write. Each returns a JSON-serializable
dict; errors are returned as {"error": ...} rather than raised, so a bad argument
never ends the agent loop.

`TOOL_SCHEMAS` holds the JSON-schema definitions handed to the model; `run_tool`
dispatches a call by name.
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
from nba.ingest import kaggle_dump
from nba.predict import residuals as residuals_module
from nba.storage import local

MAX_TOP_N = 25
MAX_RECENT = 25
MAX_DAYS = 120
ROLLING_DEFAULT_DAYS = 30


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


def get_daily_report(ctx: ToolContext, date: str) -> dict[str, Any]:
    """Ingest counts, changed rows, and push status for one nightly run date."""
    d = _parse_date(date)
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
    if ctx.dump_dir is not None and (ctx.dump_dir / kaggle_dump.BOX_SCORE_FILE).exists():
        window_start = pd.Timestamp(stored_max) - pd.Timedelta(days=config.DAILY_LOOKBACK_DAYS)
        box = kaggle_dump.load_box_scores(ctx.dump_dir, start=window_start, allow_empty=True)
        if not box.empty:
            dump_max = box["gameDate"].max().date()
    out["dump_present"] = ctx.dump_dir is not None
    out["dump_max_game_date"] = dump_max.isoformat() if dump_max else None
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


def get_rolling_metrics(ctx: ToolContext, days: int = ROLLING_DEFAULT_DAYS) -> dict[str, Any]:
    """MAE per target for the model and the last-10 baseline over the last `days` days."""
    days = max(1, min(int(days), MAX_DAYS))
    end = ctx.run_date
    start = end - timedelta(days=days - 1)
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
                "dates_covered": [d.isoformat() for d, _ in frames],
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
                "dates_covered": [x["date"] for x in days_in],
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
                "status for one nightly run date."
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
                "Newest game date stored (and in the downloaded dump if present) versus the "
                "run date, as days stale."
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
                "days ending on the run date."
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
