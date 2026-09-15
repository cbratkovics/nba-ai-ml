"""Render the generated parts of the Hugging Face dataset card.

The card (README.md in the dataset repo; canonical copy committed as docs/DATASET_CARD.md)
is prose maintained by hand around a few generated lines: the rows of the season table,
the "did not play" bullet, the missing-games bullet and the excluded-games bullet under
Known limitations, the file-layout line under Files, and the two provenance bullets
(historical backfill, daily updates) which come from `nba.config.source_lines()` so the
card can never name a source the code does not use. Everything else is left exactly as
it was.

Usage:
    python -m nba.storage.dataset_card [--card docs/DATASET_CARD.md | --from-hub]
                                       [--data-dir data/game_logs] [--out docs/DATASET_CARD.md]
                                       [--push]
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from nba import config
from nba.storage import local

CARD_PATH = Path("docs") / "DATASET_CARD.md"

TABLE_HEADER = "| Season | Rows | Games | First game | Last game |"
TABLE_SEPARATOR = "|---|---:|---:|---|---|"
SEASON_ROW = re.compile(r"^\| (20\d\d-\d\d) \|.*\|\s*$")
MISSING_LINE = re.compile(r"^- Games missing from the dump.*$")
DNP_LINE = re.compile(r"^- Rows for players who did not play \(DNP\) are .*$")
FILES_LINE = re.compile(r"^`game_logs/[^`]*`, one per season\.\s*$")
EXCLUDED_LINE = re.compile(r"^- Playoffs, play-in, (and )?preseason.*excluded\..*$")
BACKFILL_LINE = re.compile(r"^- \*\*Historical backfill:\*\*.*$")
DAILY_LINE = re.compile(r"^- \*\*Daily updates.*$")
PROVENANCE_HEADING = "## Provenance"
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
        "postponed and never re-captured upstream, so their box scores are absent here and "
        f"are not filled in from any other source. {parts}."
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


def provenance_lines() -> list[str]:
    """The two source bullets, rendered from nba.config."""
    lines = config.source_lines()
    return [f"- {lines['backfill']}", f"- {lines['daily']}"]


def _render_provenance(lines: list[str]) -> list[str]:
    """Replace the backfill and daily-update bullets, or insert them under ## Provenance."""
    bullets = provenance_lines()

    def replace(pattern: re.Pattern[str], bullet: str) -> bool:
        hit = [i for i, ln in enumerate(lines) if pattern.match(ln)]
        if not hit:
            return False
        i = hit[0]
        # A hand-wrapped bullet continues on indented lines; drop them with the bullet.
        end = i + 1
        while end < len(lines) and lines[end].startswith("  ") and lines[end].strip():
            end += 1
        lines[i:end] = [bullet]
        return True

    hit_backfill = replace(BACKFILL_LINE, bullets[0])
    hit_daily = replace(DAILY_LINE, bullets[1])
    if hit_backfill and hit_daily:
        return lines
    missing = [b for b, hit in zip(bullets, (hit_backfill, hit_daily), strict=True) if not hit]
    heading = [i for i, ln in enumerate(lines) if ln.strip() == PROVENANCE_HEADING]
    if heading:
        at = heading[0] + 1
        while at < len(lines) and lines[at].strip() == "":
            at += 1
        lines[at:at] = missing
        return lines
    # No provenance section yet: add one before the first "## " heading after the title.
    first = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines))
    lines[first:first] = [PROVENANCE_HEADING, "", *missing, ""]
    return lines


def render_dataset_card(
    card: str,
    summary: pd.DataFrame,
    missing_games: Sequence[Mapping[str, str]] = (),
    update_dnp: bool = True,
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
            if update_dnp:
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
    lines = _render_provenance(lines)

    if missing_games:
        bullet = missing_games_line(missing_games)
        existing = [i for i, ln in enumerate(lines) if MISSING_LINE.match(ln)]
        if existing:
            lines[existing[0]] = bullet
        else:
            dnp_at = next(i for i, ln in enumerate(lines) if DNP_LINE.match(ln))
            lines.insert(dnp_at + 1, bullet)
    return "\n".join(lines)


def season_summary_from_parquet(data_dir: Path) -> pd.DataFrame:
    """Rows, games, first and last game per season, from the stored game logs."""
    logs = local.read_game_logs(data_dir)
    return logs.groupby("season").agg(
        rows=("game_id", "size"),
        games=("game_id", "nunique"),
        first_game=("game_date", "min"),
        last_game=("game_date", "max"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--card", type=Path, default=CARD_PATH, help="card to render from")
    parser.add_argument(
        "--from-hub", action="store_true", help="start from the card currently on Hugging Face"
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--out", type=Path, default=CARD_PATH)
    parser.add_argument("--push", action="store_true", help="upload the rendered card (README.md)")
    args = parser.parse_args(argv)

    from nba.ingest import kaggle_dump
    from nba.storage import hf

    base = hf.fetch_dataset_card() if args.from_hub else args.card.read_text()
    summary = season_summary_from_parquet(args.data_dir)
    text = render_dataset_card(
        base, summary, missing_games=kaggle_dump.KNOWN_MISSING_GAMES, update_dnp=False
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"CARD dataset: wrote {args.out}")
    if args.push:
        sha = hf.push_dataset_card(args.out)
        print(f"CARD dataset: pushed README.md to {config.HF_DATASET_REPO} at {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
