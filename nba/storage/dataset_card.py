"""Fill the per-season table and the DNP note in the Hugging Face dataset card.

The card (README.md in the dataset repo) is maintained by hand. Only four places
are rewritten from backfill output: the rows of the season table, the "did not
play" bullet and the excluded-games bullet under Known limitations, and the
file-layout line under Files. Everything else is left exactly as it was.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import pandas as pd

from nba import config
from nba.storage import local

TABLE_HEADER = "| Season | Rows | Games | First game | Last game |"
TABLE_SEPARATOR = "|---|---:|---:|---|---|"
SEASON_ROW = re.compile(r"^\| (20\d\d-\d\d) \|.*\|\s*$")
MISSING_LINE = re.compile(r"^- Games missing from the dump.*$")
DNP_LINE = re.compile(r"^- Rows for players who did not play \(DNP\) are .*$")
FILES_LINE = re.compile(r"^`game_logs/[^`]*`, one per season\.\s*$")
EXCLUDED_LINE = re.compile(r"^- Playoffs, play-in, (and )?preseason.*excluded\..*$")
EXCLUDED_TEXT = (
    "- Playoffs, play-in, preseason, and All-Star games are excluded. NBA Cup (in-season "
    "tournament) group and knockout games are included and the Cup final is excluded, "
    "matching official regular-season accounting."
)


def season_rows(summary: pd.DataFrame) -> list[str]:
    rows = []
    for season, r in summary.sort_index().iterrows():
        first = pd.Timestamp(r["first_game"]).date().isoformat()
        last = pd.Timestamp(r["last_game"]).date().isoformat()
        rows.append(f"| {season} | {int(r['rows']):,} | {int(r['games']):,} | {first} | {last} |")
    return rows


def missing_games_line(missing_games: Sequence[Mapping[str, str]]) -> str:
    by_season: dict[str, list[str]] = {}
    for g in missing_games:
        by_season.setdefault(g["season"], []).append(
            f"{g['game_id']} ({g['scheduled']}, {g['home']} vs {g['away']})"
        )
    parts = "; ".join(
        f"{season}: " + ", ".join(items) for season, items in sorted(by_season.items())
    )
    return (
        f"- Games missing from the dump: {len(missing_games)} regular-season games were "
        "postponed and never re-captured upstream, so their box scores are absent here; "
        f"they will be backfilled from nba_api. {parts}."
    )


def dnp_line(summary: pd.DataFrame) -> str:
    total = int(summary["dnp_dropped"].sum())
    per_season = "; ".join(
        f"{season}: {int(n):,}" for season, n in summary["dnp_dropped"].sort_index().items()
    )
    return (
        f"- Rows for players who did not play (DNP) are dropped: {total:,} rows "
        f"({per_season}). A row counts as DNP when minutes are missing or 0, or when "
        "the box-score comment is populated."
    )


def files_line() -> str:
    return f"`{config.HF_DATASET_PREFIX}/{local.FILE_PREFIX}YYYY-YY.parquet`, one per season."


def render_dataset_card(
    card: str,
    summary: pd.DataFrame,
    missing_games: Sequence[Mapping[str, str]] = (),
) -> str:
    """Return `card` with the season table, DNP bullet, excluded-games bullet, files line,
    and (when given) the missing-games bullet filled in."""
    lines = card.split("\n")
    try:
        header = next(i for i, ln in enumerate(lines) if ln.strip().startswith("| Season |"))
    except StopIteration as exc:
        raise ValueError(f"dataset card has no table header {TABLE_HEADER!r}") from exc
    start = header + 2  # skip the |---| separator
    end = start
    while end < len(lines) and SEASON_ROW.match(lines[end]):
        end += 1
    if end == start:
        raise ValueError("dataset card season table has no rows to replace")
    lines[header:end] = [TABLE_HEADER, TABLE_SEPARATOR] + season_rows(summary)

    replaced_dnp = replaced_files = replaced_excluded = False
    for i, ln in enumerate(lines):
        if DNP_LINE.match(ln):
            lines[i] = dnp_line(summary)
            replaced_dnp = True
        elif FILES_LINE.match(ln):
            lines[i] = files_line()
            replaced_files = True
        elif EXCLUDED_LINE.match(ln):
            lines[i] = EXCLUDED_TEXT
            replaced_excluded = True
    if not replaced_dnp:
        raise ValueError("dataset card has no DNP bullet to replace")
    if not replaced_files:
        raise ValueError("dataset card has no files line to replace")
    if not replaced_excluded:
        raise ValueError("dataset card has no excluded-games bullet to replace")

    if missing_games:
        bullet = missing_games_line(missing_games)
        existing = [i for i, ln in enumerate(lines) if MISSING_LINE.match(ln)]
        if existing:
            lines[existing[0]] = bullet
        else:
            dnp_at = next(i for i, ln in enumerate(lines) if ln == dnp_line(summary))
            lines.insert(dnp_at + 1, bullet)
    return "\n".join(lines)
