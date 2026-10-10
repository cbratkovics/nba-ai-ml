"""One capture of a game night's lines (ADR-0031).

Order of operations, and why it is this order:

  1. The decision to capture is written and pushed (`captures/<date>.json`) BEFORE any priced
     call. Once that record exists, every qualifying line of the night enters the ledger; a
     night cannot be dropped after its lines are seen. The record names the policy version
     and whether the run was the schedule or a dispatch.
  2. The night's events come from the free `/events` endpoint. Games that commenced before
     the snapshot are out of scope by rule and counted; the rest are the plan.
  3. The plan's cost is computed under the documented rule and the credit guard is asked
     once for the whole plan. A refusal is recorded in the capture file and the usage log and
     the command exits 0 with nothing fetched (fail closed, no retry).
  4. One priced call per in-scope event, in commence-time order, each logged with the cost
     the provider reported. A failed call is logged and skipped, never retried.
  5. Lines are flattened (one row per player, market, book, with both prices), player names
     are matched deterministically, the capture record is completed with the counts, and
     lines + capture + usage log are pushed to the private repo in one commit.

Usage:
    python -m nba.lines.capture --date 2026-10-20 [--trigger schedule|dispatch] [--use-reserve]
                                 [--root .] [--no-pull] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.lines import matching, store
from nba.lines import schedule as lines_schedule
from nba.lines.guard import CreditGuard
from nba.lines.odds_api import OddsApi, OddsApiError, event_odds_cost
from nba.storage import local

LINE_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "snapshot_at_utc",
    "source",
    "book",
    "event_id",
    "commence_time_utc",
    "game_date",
    "home_team",
    "away_team",
    "team",
    "player_name_raw",
    "player_id",
    "match_method",
    "market",
    "line",
    "over_price",
    "under_price",
    "line_last_update_utc",
)
POLICY_VERSION = "capture_policy_v1"


@dataclass
class Capture:
    capture_id: str
    date: str
    trigger: str
    policy_version: str
    snapshot_at_utc: str
    decided_before_fetch: bool = True
    status: str = "decided"  # decided | refused | captured | no_games
    n_events: int = 0
    n_in_scope: int = 0
    n_out_of_scope: int = 0
    planned_cost: int = 0
    charged: int = 0
    n_calls_ok: int = 0
    n_calls_failed: int = 0
    n_lines: int = 0
    n_players: int = 0
    n_unmatched: int = 0
    guard: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    lines_revision: str | None = None


def snapshot_time(d: date, hhmm: str = config.CAPTURE_SNAPSHOT_UTC) -> datetime:
    hh, mm = (int(x) for x in hhmm.split(":"))
    return datetime.combine(d, time(hh, mm), tzinfo=UTC)


def eastern_day_bounds_utc(d: date) -> tuple[str, str]:
    """ISO bounds that cover the whole Eastern-time calendar day in UTC."""
    start = datetime.combine(d, time(0, 0), tzinfo=lines_schedule.EASTERN)
    end = start + timedelta(days=1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return start.astimezone(UTC).strftime(fmt), end.astimezone(UTC).strftime(fmt)


def plan_events(
    events: list[dict[str, Any]], d: date, snapshot: datetime
) -> tuple[list[dict], int]:
    """Events on the Eastern date `d` whose commence time is after the snapshot, in tip-off order;
    second value is how many of the date's games were already under way (out of scope)."""
    todays = [e for e in events if lines_schedule.eastern_date(e["commence_time"]) == d]
    in_scope = [
        e
        for e in todays
        if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) > snapshot
    ]
    in_scope.sort(key=lambda e: e["commence_time"])
    return in_scope, len(todays) - len(in_scope)


def flatten(
    event: dict[str, Any], payload: dict[str, Any], capture: Capture
) -> list[dict[str, Any]]:
    """One row per (bookmaker, market, player) with the over and under prices and the point."""
    home = lines_schedule.abbreviation(event["home_team"])
    away = lines_schedule.abbreviation(event["away_team"])
    rows: dict[tuple, dict[str, Any]] = {}
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                key = (book["key"], market["key"], o.get("description"), o.get("point"))
                row = rows.setdefault(
                    key,
                    {
                        "capture_id": capture.capture_id,
                        "snapshot_at_utc": capture.snapshot_at_utc,
                        "source": "api",
                        "book": book["key"],
                        "event_id": str(event["id"]),
                        "commence_time_utc": event["commence_time"],
                        "game_date": lines_schedule.eastern_date(
                            event["commence_time"]
                        ).isoformat(),
                        "home_team": home,
                        "away_team": away,
                        "team": None,
                        "player_name_raw": o.get("description"),
                        "player_id": None,
                        "match_method": None,
                        "market": market["key"],
                        "line": o.get("point"),
                        "over_price": None,
                        "under_price": None,
                        "line_last_update_utc": market.get("last_update"),
                    },
                )
                if o.get("name") == "Over":
                    row["over_price"] = o.get("price")
                elif o.get("name") == "Under":
                    row["under_price"] = o.get("price")
    return list(rows.values())


def match(lines: pd.DataFrame, game_logs: pd.DataFrame, season: str, previous: str) -> pd.DataFrame:
    """Attach player_id: the player's team is whichever of the two teams rostered the name."""
    if lines.empty:
        return lines
    roster = matching.roster_index(game_logs, (previous, season))
    out = lines.copy()
    for side in ("home_team", "away_team"):
        names = pd.DataFrame({"raw_name": out["player_name_raw"], "team": out[side]})
        got = matching.match_players(names, roster, aliases=None)
        hit = got["method"].isin(["team", "alias"]) & out["player_id"].isna()
        out.loc[hit, "player_id"] = got.loc[hit, "player_id"]
        out.loc[hit, "match_method"] = got.loc[hit, "method"]
        out.loc[hit, "team"] = out.loc[hit, side]
    still = out["player_id"].isna()
    if still.any():
        names = pd.DataFrame({"raw_name": out.loc[still, "player_name_raw"], "team": None})
        got = matching.match_players(names, roster, aliases=None).set_index(out.index[still])
        league = got["method"].eq("league")
        out.loc[got.index[league], "player_id"] = got.loc[league, "player_id"]
        out.loc[got.index[league], "match_method"] = "league"
    out["match_method"] = out["match_method"].fillna("unmatched")
    out["player_id"] = out["player_id"].astype("Int64")
    return out


def run(
    d: date,
    root: Path,
    *,
    trigger: str,
    use_reserve: bool = False,
    pull: bool = True,
    dry_run: bool = False,
    api: OddsApi | None = None,
    now: datetime | None = None,
) -> Capture:
    now = now or datetime.now(UTC)
    snapshot = snapshot_time(d) if trigger == "schedule" else now
    cap = Capture(
        capture_id=f"{d.isoformat()}-{POLICY_VERSION}-{trigger}",
        date=d.isoformat(),
        trigger=trigger,
        policy_version=POLICY_VERSION,
        snapshot_at_utc=now.isoformat(timespec="seconds"),
    )
    captures_dir = root / config.CAPTURES_DIR
    captures_dir.mkdir(parents=True, exist_ok=True)
    cap_path = captures_dir / f"{d.isoformat()}.json"
    if pull:
        store.ensure_private_repo()
        store.pull(root)
    if cap_path.exists() and not dry_run:
        existing = json.loads(cap_path.read_text())
        if existing.get("status") == "captured":
            print(f"CAPTURE {d}: already captured ({existing['capture_id']}); nothing to do")
            return Capture(**existing)

    def save(final: bool = False) -> None:
        cap_path.write_text(json.dumps(asdict(cap), indent=2) + "\n")
        if not dry_run and (final or cap.status == "decided"):
            store.push(root, [cap_path], f"Capture {d}: {cap.status}")

    # 1. the decision, before any fetch
    save()
    print(f"CAPTURE {d}: decided ({trigger}); snapshot {cap.snapshot_at_utc}")

    api = api or OddsApi()
    guard = CreditGuard(root / config.ODDS_USAGE_FILE, now=now)

    # 2. the night's games (free)
    start, end = eastern_day_bounds_utc(d)
    answer = api.events(commence_from=start, commence_to=end)
    guard.record(answer.endpoint, None, 0, "ok", answer.usage)
    in_scope, out_of_scope = plan_events(answer.data, d, snapshot)
    cap.n_events, cap.n_in_scope, cap.n_out_of_scope = (
        len(in_scope) + out_of_scope,
        len(in_scope),
        out_of_scope,
    )
    if not in_scope:
        cap.status = "no_games"
        cap.notes.append("no game on this date commences after the snapshot")
        save(final=True)
        print(f"CAPTURE {d}: no in-scope games ({out_of_scope} already under way)")
        return cap

    # 3. the guard, once, for the whole plan
    per_event = event_odds_cost(len(config.ODDS_MARKETS), len(config.ODDS_BOOKMAKERS))
    cap.planned_cost = per_event * len(in_scope)
    verdict = guard.allow(cap.planned_cost, use_reserve=use_reserve)
    cap.guard = asdict(verdict)
    if not verdict.allowed:
        cap.status = "refused"
        cap.notes.append(f"credit guard: {verdict.reason}")
        guard.record("plan", None, cap.planned_cost, "refused", None, note=verdict.reason)
        save(final=True)
        if not dry_run:
            store.push(root, [root / config.ODDS_USAGE_FILE], f"Usage {d}: refused")
        print(f"CAPTURE {d}: REFUSED by the credit guard ({verdict.reason})")
        return cap
    if dry_run:
        cap.notes.append("dry run: plan allowed, nothing fetched")
        save()
        print(
            f"CAPTURE {d}: dry run, plan of {cap.planned_cost} credits allowed ({verdict.reason})"
        )
        return cap

    # 4. one call per event, no retries
    rows: list[dict[str, Any]] = []
    for e in in_scope:
        try:
            ans = api.event_odds(str(e["id"]))
        except OddsApiError as exc:
            cap.n_calls_failed += 1
            guard.record(
                f"/events/{e['id']}/odds",
                str(e["id"]),
                per_event,
                "error",
                None,
                note=str(exc)[:200],
            )
            cap.notes.append(f"event {e['id']}: {exc}")
            continue
        entry = guard.record(ans.endpoint, str(e["id"]), per_event, "ok", ans.usage)
        cap.n_calls_ok += 1
        cap.charged += entry.charged or 0
        rows.extend(flatten(e, ans.data, cap))

    # 5. flatten, match, record, push
    lines = pd.DataFrame(rows, columns=list(LINE_COLUMNS))
    season = lines_schedule.season_from_date(pd.Timestamp(d))
    previous = f"{int(season[:4]) - 1}-{season[:4][2:]}"
    game_logs = local.read_game_logs(root / config.DATA_DIR)
    lines = match(lines, game_logs, season, previous)
    lines_dir = root / config.LINES_DIR
    lines_dir.mkdir(parents=True, exist_ok=True)
    lines_path = lines_dir / f"{d.isoformat()}.parquet"
    lines.to_parquet(lines_path, index=False)
    cap.n_lines = int(len(lines))
    cap.n_players = int(lines["player_name_raw"].nunique())
    cap.n_unmatched = int(lines["player_id"].isna().sum())
    cap.status = "captured"
    cap_path.write_text(json.dumps(asdict(cap), indent=2) + "\n")
    sha = store.push(
        root,
        [cap_path, lines_path, root / config.ODDS_USAGE_FILE],
        f"Capture {d}: {cap.n_lines} lines, {cap.charged} credits",
    )
    cap.lines_revision = sha
    cap_path.write_text(json.dumps(asdict(cap), indent=2) + "\n")
    print(
        f"CAPTURE {d}: {cap.n_in_scope} games ({cap.n_out_of_scope} out of scope), "
        f"{cap.n_lines} lines, {cap.n_unmatched} unmatched names, {cap.charged} credits "
        f"({guard.summary()['spent_this_month']} this month); pushed at {sha}"
    )
    return cap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--trigger", choices=("schedule", "dispatch"), default="dispatch")
    parser.add_argument(
        "--use-reserve", action="store_true", help="a manual run may dip into the reserve"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="plan and ask the guard; fetch nothing"
    )
    args = parser.parse_args(argv)
    d = args.date or datetime.now(UTC).astimezone(lines_schedule.EASTERN).date()
    run(
        d,
        args.root,
        trigger=args.trigger,
        use_reserve=args.use_reserve,
        pull=not args.no_pull,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
