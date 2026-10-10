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
    n_out_of_season: int = 0  # events on the date that are not regular-season NBA games
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


def classify(events: list[dict[str, Any]], d: date, snapshot: datetime) -> list[dict[str, Any]]:
    """Every event the free endpoint returned, with the Eastern date and one verdict each:
    in_scope (priced), started (on the date but already under way: out of scope by rule),
    out_of_season (not a regular-season `basketball_nba` game: never priced, whatever the
    date), or other_date. The season guard comes first so a preseason, All-Star or play-in
    game is out of season even when it is on the date (ADR-0031)."""
    out = []
    for e in events:
        game_date = lines_schedule.eastern_date(e["commence_time"])
        season_class = lines_schedule.classify_event(e)
        if season_class != "regular_season":
            verdict = "out_of_season"
        elif game_date != d:
            verdict = "other_date"
        elif datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= snapshot:
            verdict = "started"
        else:
            verdict = "in_scope"
        out.append(
            {
                "event_id": str(e.get("id")),
                "sport_key": e.get("sport_key", config.ODDS_SPORT),
                "commence_time": e["commence_time"],
                "game_date": game_date.isoformat(),
                "home_team": e.get("home_team"),
                "away_team": e.get("away_team"),
                "season_class": season_class,
                "verdict": verdict,
            }
        )
    return out


def plan_events(
    events: list[dict[str, Any]], d: date, snapshot: datetime
) -> tuple[list[dict], int, int]:
    """Events to price: regular-season games on the Eastern date `d` that commence after the
    snapshot, in tip-off order; then how many of the date's regular-season games were already
    under way (out of scope), and how many of the date's events were out of season."""
    verdicts = {c["event_id"]: c["verdict"] for c in classify(events, d, snapshot)}
    in_scope = [e for e in events if verdicts[str(e.get("id"))] == "in_scope"]
    in_scope.sort(key=lambda e: e["commence_time"])
    started = sum(1 for v in verdicts.values() if v == "started")
    out_of_season = sum(
        1
        for c in classify(events, d, snapshot)
        if c["verdict"] == "out_of_season" and c["game_date"] == d.isoformat()
    )
    return in_scope, started, out_of_season


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

    # 2. the night's games (free). The provider's usage headers are printed on every run so a
    # log shows the credits spent (x-requests-used) and left (x-requests-remaining).
    start, end = eastern_day_bounds_utc(d)
    answer = api.events(commence_from=start, commence_to=end)
    guard.record(answer.endpoint, None, 0, "ok", answer.usage)
    u = answer.usage
    print(
        f"USAGE after the free events call: remaining={u.remaining} used={u.used} "
        f"last_call_cost={u.last} (events window {start} to {end}, {len(answer.data)} events)"
    )
    if dry_run:
        # A dry run also lists everything the endpoint returns without a date window (still
        # free), so the log shows which upcoming games would and would not be priced.
        upcoming = api.events()
        guard.record(upcoming.endpoint, None, 0, "ok", upcoming.usage)
        print(f"UPCOMING {len(upcoming.data)} events on the endpoint (no date window):")
        for c in classify(upcoming.data, d, snapshot):
            would = "would price" if c["verdict"] == "in_scope" else "would not price"
            why = c["verdict"]
            if c["season_class"] != "regular_season":
                why += ", " + c["season_class"]
            print(
                f"UPCOMING {c['event_id']} {c['commence_time']} ET {c['game_date']} "
                f"{c['away_team']} @ {c['home_team']} [{c['sport_key']}] -> {would} ({why})"
            )
    for c in classify(answer.data, d, snapshot):
        print(
            f"EVENT {c['event_id']} {c['commence_time']} ET {c['game_date']} "
            f"{c['away_team']} @ {c['home_team']} [{c['sport_key']}] -> {c['verdict']}"
            + ("" if c["season_class"] == "regular_season" else f" ({c['season_class']})")
        )
    in_scope, out_of_scope, out_of_season = plan_events(answer.data, d, snapshot)
    cap.n_events = len(in_scope) + out_of_scope + out_of_season
    cap.n_in_scope, cap.n_out_of_scope, cap.n_out_of_season = (
        len(in_scope),
        out_of_scope,
        out_of_season,
    )
    if not lines_schedule.in_regular_season(d):
        cap.status = "out_of_season"
        cap.notes.append(
            f"{d} is outside the regular-season window "
            f"{config.REGULAR_SEASON_WINDOWS}; no priced call is ever made out of season"
        )
        save(final=True)
        print(
            f"CAPTURE {d}: out of season ({out_of_season} non-regular-season events on the "
            "date); no priced call"
        )
        return cap
    if not in_scope:
        cap.status = "no_games"
        cap.notes.append("no regular-season game on this date commences after the snapshot")
        save(final=True)
        print(
            f"CAPTURE {d}: no in-scope games ({out_of_scope} already under way, "
            f"{out_of_season} out of season)"
        )
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
        f"CAPTURE {d}: {cap.n_in_scope} games ({cap.n_out_of_scope} out of scope, "
        f"{cap.n_out_of_season} out of season), "
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
