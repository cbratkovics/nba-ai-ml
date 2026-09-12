"""Daily incremental ingest from the Kaggle dump.

Downloads only PlayerStatistics.csv, TeamHistories.csv (7 KB, needed to resolve
team abbreviations), and the current season's LeagueSchedule file from
`config.KAGGLE_DATASET` with the Kaggle API single-file download, then:

  1. reads the game logs already stored on Hugging Face (pulled to data_dir),
  2. takes dump rows with game_date > (stored max game_date - DAILY_LOOKBACK_DAYS),
     mapped with exactly the backfill's rules (`nba.ingest.kaggle_dump.prepare`),
  3. classifies each (player_id, game_id) as new, unchanged, or changed vs stored,
  4. appends new rows and replaces changed rows with source="kaggle_daily",
  5. writes a reconciliation summary to data/daily_report.json, and
  6. pushes the affected season Parquet files (and a refreshed card) only if
     something changed.

Kaggle credentials come from KAGGLE_USERNAME / KAGGLE_KEY (environment or .env),
read in `nba.config`. `--local-dir` skips the download and reads an extracted dump.

Usage:
    python -m nba.ingest.kaggle_daily [--date YYYY-MM-DD] [--local-dir data_dump]
                                      [--data-dir data/game_logs] [--report data/daily_report.json]
                                      [--push] [--no-pull]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from nba import config, schema
from nba.ingest import kaggle_dump, schedule
from nba.storage import dataset_card, hf, local

KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{dataset}/{file_name}"
DOWNLOAD_TIMEOUT = 900  # seconds; PlayerStatistics.csv is ~400 MB
MAX_CHANGED_EXAMPLES = 20
# Columns compared to decide whether a stored row changed (everything except provenance).
COMPARE_COLUMNS: tuple[str, ...] = tuple(c for c in schema.COLUMNS if c != "source")


class KaggleAuthError(RuntimeError):
    """Kaggle credentials are missing."""


def download_kaggle_file(
    file_name: str,
    dest_dir: Path,
    username: str | None = None,
    key: str | None = None,
    dataset: str = config.KAGGLE_DATASET,
    timeout: int = DOWNLOAD_TIMEOUT,
) -> Path | None:
    """Download one file of the dataset into dest_dir. Returns None when it does not exist.

    The API answers with either the raw file or a zip containing it; both are handled.
    """
    username = username or config.KAGGLE_USERNAME
    key = key or config.KAGGLE_KEY
    if not username or not key:
        raise KaggleAuthError("KAGGLE_USERNAME and KAGGLE_KEY are required to download")
    url = KAGGLE_DOWNLOAD_URL.format(dataset=dataset, file_name=file_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name
    with requests.get(url, auth=(username, key), stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            return None
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".download")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    with open(tmp, "rb") as fh:
        magic = fh.read(4)
    if magic.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(tmp) as zf:
            members = zf.namelist()
            member = file_name if file_name in members else members[0]
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        tmp.unlink()
    else:
        tmp.replace(dest)
    return dest


def download_dump_subset(dest_dir: Path, today: date) -> dict[str, str | None]:
    """Fetch the box scores, team histories, and the current (or previous) schedule file."""
    fetched: dict[str, str | None] = {}
    for name in (kaggle_dump.BOX_SCORE_FILE, kaggle_dump.TEAM_HISTORY_FILE):
        path = download_kaggle_file(name, dest_dir)
        if path is None:
            raise FileNotFoundError(f"{name} is not in {config.KAGGLE_DATASET}")
        fetched[name] = str(path)
    season = schedule.season_for_date(today)
    previous = (
        f"{schedule.season_start_year(season) - 1}-{schedule.season_start_year(season) % 100:02d}"
    )
    fetched["schedule"] = None
    for candidate in (season, previous):
        name = schedule.schedule_file_name(candidate)
        path = download_kaggle_file(name, dest_dir)
        if path is not None:
            fetched["schedule"] = name
            break
    return fetched


@dataclass
class Classification:
    new: pd.DataFrame
    changed: pd.DataFrame
    unchanged: int
    examples: list[dict[str, Any]] = field(default_factory=list)


def _values_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
        a_num, b_num = pd.to_numeric(a), pd.to_numeric(b)
        return (a_num.round(4) == b_num.round(4)) | (a_num.isna() & b_num.isna())
    return (a == b) | (a.isna() & b.isna())


def classify(stored: pd.DataFrame, incoming: pd.DataFrame) -> Classification:
    """Split incoming rows into new / unchanged / changed relative to stored rows."""
    keys = list(schema.KEY_COLUMNS)
    merged = incoming.merge(stored, on=keys, how="left", suffixes=("", "__stored"), indicator=True)
    is_new = merged["_merge"] == "left_only"
    compare = [c for c in COMPARE_COLUMNS if c not in keys]
    same = pd.Series(True, index=merged.index)
    diff_cols: dict[int, list[str]] = {}
    for col in compare:
        eq = _values_equal(merged[col], merged[f"{col}__stored"]) | is_new
        for idx in merged.index[~eq]:
            diff_cols.setdefault(idx, []).append(col)
        same &= eq
    is_changed = ~is_new & ~same
    examples = []
    for idx in list(merged.index[is_changed])[:MAX_CHANGED_EXAMPLES]:
        row = merged.loc[idx]
        examples.append(
            {
                "player_id": int(row["player_id"]),
                "game_id": str(row["game_id"]),
                "game_date": pd.Timestamp(row["game_date"]).date().isoformat(),
                "player_name": str(row["player_name"]),
                "fields": {
                    c: {"stored": _json_value(row[f"{c}__stored"]), "incoming": _json_value(row[c])}
                    for c in diff_cols.get(idx, [])
                },
            }
        )
    new = incoming.loc[merged.index[is_new]]
    changed = incoming.loc[merged.index[is_changed]]
    return Classification(
        new=new, changed=changed, unchanged=int((~is_new & same).sum()), examples=examples
    )


def _json_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def merge_rows(stored: pd.DataFrame, replacements: pd.DataFrame) -> pd.DataFrame:
    """Stored rows with `replacements` appended or substituted on (player_id, game_id)."""
    if replacements.empty:
        return stored
    keys = list(schema.KEY_COLUMNS)
    drop = stored.set_index(keys).index.isin(replacements.set_index(keys).index)
    merged = pd.concat([stored[~drop], replacements], ignore_index=True)
    merged = merged.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    return schema.validate(schema.coerce(merged))


def run_daily(
    dump_dir: Path,
    data_dir: Path,
    today: date,
    report_path: Path,
    push: bool = False,
    stored_revision: str | None = None,
    schedule_file: str | None = None,
) -> dict[str, Any]:
    """Incremental ingest against the game logs already in data_dir. Returns the report."""
    stored = local.read_game_logs(data_dir)
    stored_max = stored["game_date"].max()
    cutoff = stored_max - pd.Timedelta(days=config.DAILY_LOOKBACK_DAYS)
    box = kaggle_dump.load_box_scores(
        dump_dir, start=cutoff + pd.Timedelta(days=1), allow_empty=True
    )
    if box.empty:
        # Off-season, or the dump has not been updated: nothing to reconcile.
        incoming = stored.iloc[0:0].copy()
        dnp_in_window = 0
    else:
        histories = kaggle_dump.load_team_histories(dump_dir)
        prepared = kaggle_dump.prepare(box, histories, seasons=None, require_primary_label=False)
        incoming = prepared.game_logs.copy()
        incoming["source"] = config.KAGGLE_DAILY_SOURCE
        incoming = schema.validate(schema.coerce(incoming))
        dnp_in_window = int(prepared.dnp_per_season.sum())

    result = classify(stored, incoming)
    replacements = pd.concat([result.new, result.changed], ignore_index=True)
    seasons_written: list[str] = []
    files: list[Path] = []
    if not replacements.empty:
        merged = merge_rows(stored, replacements)
        affected = sorted(replacements["season"].unique())
        written = local.write_per_season(merged[merged["season"].isin(affected)], data_dir)
        seasons_written = sorted(written)
        files = [written[s] for s in seasons_written]
    changed = bool(seasons_written)

    report: dict[str, Any] = {
        "date": today.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_revision_before": stored_revision,
        "stored_rows": int(len(stored)),
        "stored_max_game_date": stored_max.date().isoformat(),
        "window_start_exclusive": cutoff.date().isoformat(),
        "window_rows_in_dump": int(len(box)),
        "window_rows_after_rules": int(len(incoming)),
        "dnp_dropped_in_window": dnp_in_window,
        "counts": {
            "new": int(len(result.new)),
            "changed": int(len(result.changed)),
            "unchanged": int(result.unchanged),
        },
        "changed_examples": result.examples,
        "seasons_written": seasons_written,
        "schedule_file": schedule_file,
        "pushed": False,
        "dataset_revision_after": stored_revision,
    }

    if changed and push:
        merged_all = local.read_game_logs(data_dir)
        summary = merged_all.groupby("season").agg(
            rows=("game_id", "size"),
            games=("game_id", "nunique"),
            first_game=("game_date", "min"),
            last_game=("game_date", "max"),
        )
        card_text = dataset_card.render_dataset_card(
            hf.fetch_dataset_card(),
            summary,
            missing_games=kaggle_dump.KNOWN_MISSING_GAMES,
            update_dnp=False,
        )
        card_path = data_dir / hf.DATASET_CARD_FILE
        card_path.write_text(card_text)
        sha = hf.push_dataset(
            data_dir,
            card_path=card_path,
            files=files,
            message=f"Daily ingest {today.isoformat()}: {report['counts']['new']} new, "
            f"{report['counts']['changed']} changed",
        )
        report["pushed"] = True
        report["dataset_revision_after"] = sha

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--date", type=date.fromisoformat, default=None, help="run date (UTC today)"
    )
    parser.add_argument(
        "--local-dir", type=Path, default=None, help="extracted dump to use instead of downloading"
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--report", type=Path, default=config.DAILY_REPORT_PATH)
    parser.add_argument("--push", action="store_true", help="push changed seasons to HF")
    parser.add_argument(
        "--no-pull", action="store_true", help="use data-dir as-is instead of pulling"
    )
    args = parser.parse_args(argv)
    today = args.date or datetime.now(UTC).date()

    revision = None if args.no_pull else hf.pull_dataset(args.data_dir)
    print(f"DAILY stored dataset revision: {revision or 'local'}")

    if args.local_dir is not None:
        dump_dir = args.local_dir
        sched = schedule.schedule_path(dump_dir, today)
        schedule_file = sched.name if sched else None
        print(f"DAILY using local dump {dump_dir} (no download)")
    else:
        dump_dir = Path(tempfile.mkdtemp(prefix="kaggle_daily_"))
        fetched = download_dump_subset(dump_dir, today)
        schedule_file = fetched["schedule"]
        print(f"DAILY downloaded: {fetched}")

    report = run_daily(
        dump_dir,
        args.data_dir,
        today,
        args.report,
        push=args.push,
        stored_revision=revision,
        schedule_file=schedule_file,
    )
    c = report["counts"]
    print(
        f"DAILY {today}: window > {report['window_start_exclusive']}, "
        f"{report['window_rows_after_rules']} rows after rules; "
        f"new={c['new']} changed={c['changed']} unchanged={c['unchanged']}; "
        f"seasons_written={report['seasons_written']}; pushed={report['pushed']}"
    )
    print(f"DAILY report written to {args.report}")


if __name__ == "__main__":
    main()
