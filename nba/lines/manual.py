"""Hand-entered lines (ADR-0031): a CSV typed from the owner's own sportsbook account, the only
way rebounds, assists and Caesars enter the ledger at $0. Validated on load, then written in
the same layout as API lines with source = manual and reported separately.

Template (one row per player-market line; prices are American odds):

    read_at_utc,game_date,book,player_name,team,market,line,over_price,under_price,notes
    2026-10-20T20:05:00Z,2026-10-20,caesars,Jayson Tatum,BOS,player_points,27.5,-115,-105,
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from nba import config

TEMPLATE_COLUMNS = (
    "read_at_utc",
    "game_date",
    "book",
    "player_name",
    "team",
    "market",
    "line",
    "over_price",
    "under_price",
    "notes",
)
BOOKS = ("draftkings", "caesars")
MARKETS = ("player_points", "player_rebounds", "player_assists")
LINE_RANGE = {"player_points": (0.5, 60), "player_rebounds": (0.5, 25), "player_assists": (0.5, 20)}


class ManualLinesError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _price_ok(v) -> bool:
    try:
        p = float(v)
    except (TypeError, ValueError):
        return False
    return p.is_integer() and (p <= -100 or p >= 100)


def validate(frame: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    missing = [c for c in TEMPLATE_COLUMNS if c not in frame.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for i, r in frame.iterrows():
        row = f"row {i + 1}"
        try:
            read_at = datetime.fromisoformat(str(r["read_at_utc"]).replace("Z", "+00:00"))
            if read_at.tzinfo is None:
                problems.append(f"{row}: read_at_utc needs a timezone (use Z)")
        except ValueError:
            problems.append(f"{row}: read_at_utc is not ISO-8601")
        try:
            datetime.strptime(str(r["game_date"]), "%Y-%m-%d")
        except ValueError:
            problems.append(f"{row}: game_date is not YYYY-MM-DD")
        if str(r["book"]).strip().lower() not in BOOKS:
            problems.append(f"{row}: book must be one of {BOOKS}")
        market = str(r["market"]).strip()
        if market not in MARKETS:
            problems.append(f"{row}: market must be one of {MARKETS}")
        else:
            lo, hi = LINE_RANGE[market]
            try:
                line = float(r["line"])
                if not (lo <= line <= hi) or (line * 2) != int(line * 2):
                    problems.append(f"{row}: line must be a whole or half number in [{lo}, {hi}]")
            except (TypeError, ValueError):
                problems.append(f"{row}: line is not a number")
        for col in ("over_price", "under_price"):
            if not _price_ok(r[col]):
                problems.append(f"{row}: {col} must be an American price (<= -100 or >= +100)")
        if not str(r["player_name"]).strip() or not str(r["team"]).strip():
            problems.append(f"{row}: player_name and team are required")
    dupes = frame.duplicated(subset=["game_date", "book", "player_name", "market"], keep=False)
    if dupes.any():
        problems.append(
            f"duplicate (game_date, book, player_name, market) rows: {int(dupes.sum())}"
        )
    return problems


def load(path: Path) -> pd.DataFrame:
    """Read and validate the template; raise with every problem at once."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    problems = validate(frame)
    if problems:
        raise ManualLinesError(problems)
    out = pd.DataFrame(
        {
            "read_at_utc": pd.to_datetime(frame["read_at_utc"], utc=True),
            "game_date": pd.to_datetime(frame["game_date"]).dt.date,
            "book": frame["book"].str.strip().str.lower(),
            "player_name_raw": frame["player_name"].str.strip(),
            "team": frame["team"].str.strip().str.upper(),
            "market": frame["market"].str.strip(),
            "line": frame["line"].astype(float),
            "over_price": frame["over_price"].astype(float).astype(int),
            "under_price": frame["under_price"].astype(float).astype(int),
            "notes": frame["notes"],
        }
    )
    out["source"] = "manual"
    return out


def template_text() -> str:
    return ",".join(TEMPLATE_COLUMNS) + "\n"


def to_lines(frame: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Hand-entered rows in the captured-lines layout (nba/lines/capture.py LINE_COLUMNS), with
    the player matched by the same deterministic step as API rows."""
    from nba.lines import capture, matching
    from nba.lines.schedule import season_from_date

    out = pd.DataFrame(
        {
            "capture_id": frame["game_date"].map(lambda d: f"{d.isoformat()}-manual"),
            "snapshot_at_utc": frame["read_at_utc"],
            "source": "manual",
            "book": frame["book"],
            "event_id": None,
            "commence_time_utc": pd.NaT,
            "game_date": frame["game_date"].map(lambda d: d.isoformat()),
            "home_team": None,
            "away_team": None,
            "team": frame["team"],
            "player_name_raw": frame["player_name_raw"],
            "player_id": None,
            "match_method": None,
            "market": frame["market"],
            "line": frame["line"],
            "over_price": frame["over_price"],
            "under_price": frame["under_price"],
            "line_last_update_utc": frame["read_at_utc"],
        },
        columns=list(capture.LINE_COLUMNS),
    )
    seasons = sorted({season_from_date(pd.Timestamp(d)) for d in frame["game_date"]})
    previous = [f"{int(s[:4]) - 1}-{s[:4][2:]}" for s in seasons]
    roster = matching.roster_index(game_logs, tuple(sorted(set(seasons + previous))))
    names = pd.DataFrame({"raw_name": out["player_name_raw"], "team": out["team"]})
    got = matching.match_players(names, roster, aliases=None)
    out["player_id"] = got["player_id"].astype("Int64")
    out["match_method"] = got["method"]
    return out


def ingest(csv_path: Path, root: Path, game_logs: pd.DataFrame, *, push: bool = True) -> list[Path]:
    """Validate, match and write lines/<date>_manual.parquet per game date (merging with any
    earlier manual rows for the date; the later read wins a duplicate), then push."""
    from nba.lines import store

    lines = to_lines(load(csv_path), game_logs)
    written = []
    lines_dir = root / config.LINES_DIR
    lines_dir.mkdir(parents=True, exist_ok=True)
    for d, part in lines.groupby("game_date"):
        path = lines_dir / f"{d}_manual.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        key = ["game_date", "book", "player_name_raw", "market"]
        part = part.sort_values("line_last_update_utc").drop_duplicates(key, keep="last")
        part.to_parquet(path, index=False)
        written.append(path)
    if push and written:
        store.push(root, written, f"Manual lines: {', '.join(p.stem for p in written)}")
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    from nba.storage import local

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template", help="print the CSV template header")
    p_in = sub.add_parser("ingest", help="validate, match and push a hand-entered CSV")
    p_in.add_argument("csv", type=Path)
    p_in.add_argument("--root", type=Path, default=Path("."))
    p_in.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "template":
        print(template_text(), end="")
        return 0
    game_logs = local.read_game_logs(args.root / config.DATA_DIR)
    written = ingest(args.csv, args.root, game_logs, push=not args.no_push)
    for p in written:
        frame = pd.read_parquet(p)
        print(
            f"MANUAL {p.name}: {len(frame)} lines, {int(frame['player_id'].isna().sum())} unmatched"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
