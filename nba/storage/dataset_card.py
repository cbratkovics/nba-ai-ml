"""Fill the per-season table and the DNP note in the Hugging Face dataset card.

The card (README.md in the dataset repo) is maintained by hand. Only three places
are rewritten from backfill output: the rows of the season table, the "did not
play" bullet under Known limitations, and the file-layout line under Files.
Everything else is left exactly as it was.
"""

from __future__ import annotations

import re

import pandas as pd

from nba import config
from nba.storage import local

TABLE_HEADER = "| Season | Rows | First game | Last game |"
SEASON_ROW = re.compile(r"^\| (20\d\d-\d\d) \|.*\|\s*$")
DNP_LINE = re.compile(r"^- Rows for players who did not play \(DNP\) are .*$")
FILES_LINE = re.compile(r"^`game_logs/[^`]*`, one per season\.\s*$")


def season_rows(summary: pd.DataFrame) -> list[str]:
    rows = []
    for season, r in summary.sort_index().iterrows():
        first = pd.Timestamp(r["first_game"]).date().isoformat()
        last = pd.Timestamp(r["last_game"]).date().isoformat()
        rows.append(f"| {season} | {int(r['rows']):,} | {first} | {last} |")
    return rows


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


def render_dataset_card(card: str, summary: pd.DataFrame) -> str:
    """Return `card` with the season table rows, DNP bullet, and files line filled in."""
    lines = card.split("\n")
    try:
        header = next(i for i, ln in enumerate(lines) if ln.strip() == TABLE_HEADER)
    except StopIteration as exc:
        raise ValueError(f"dataset card has no table header {TABLE_HEADER!r}") from exc
    start = header + 2  # skip the |---| separator
    end = start
    while end < len(lines) and SEASON_ROW.match(lines[end]):
        end += 1
    if end == start:
        raise ValueError("dataset card season table has no rows to replace")
    lines[start:end] = season_rows(summary)

    replaced_dnp = replaced_files = False
    for i, ln in enumerate(lines):
        if DNP_LINE.match(ln):
            lines[i] = dnp_line(summary)
            replaced_dnp = True
        elif FILES_LINE.match(ln):
            lines[i] = files_line()
            replaced_files = True
    if not replaced_dnp:
        raise ValueError("dataset card has no DNP bullet to replace")
    if not replaced_files:
        raise ValueError("dataset card has no files line to replace")
    return "\n".join(lines)
