"""Nightly orchestration: ingest -> residuals for yesterday -> slate for today -> push.

One process so the files written by each step are known exactly and only those are
uploaded. Outcomes that are not errors exit 0 with one explicit line each:

    SLATE <date>: no schedule file for season <season> (...)   the dump has no schedule yet
    SLATE <date>: no games on this date (...)                   schedule exists, nothing today
    RESIDUALS <date>: no predictions file ...                   no slate was written yesterday

Schema validation errors and download failures raise and fail the job.

Usage:
    python -m nba.nightly [--date YYYY-MM-DD] [--push] [--local-dump data_dump]
                          [--root .] [--summary data/nightly_summary.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.ingest import kaggle_daily, schedule
from nba.predict import model, residuals, slate
from nba.storage import hf, local

SUMMARY_PATH = Path("data") / "nightly_summary.json"


@dataclass
class NightlySummary:
    date: str
    push: bool
    dataset_revision: str | None = None
    ingest: dict[str, Any] = field(default_factory=dict)
    residuals: dict[str, Any] = field(default_factory=dict)
    slate: dict[str, Any] = field(default_factory=dict)
    products_pushed: list[str] = field(default_factory=list)
    products_revision: str | None = None
    lines: list[str] = field(default_factory=list)

    def log(self, line: str) -> None:
        print(line)
        self.lines.append(line)


def run(
    d: date,
    push: bool,
    root: Path = Path("."),
    local_dump: Path | None = None,
    report_path: Path = config.DAILY_REPORT_PATH,
    load_models=model.load_from_hub,
) -> NightlySummary:
    summary = NightlySummary(date=d.isoformat(), push=push)
    data_dir = root / config.DATA_DIR
    predictions_dir = root / config.PREDICTIONS_DIR
    residuals_dir = root / config.RESIDUALS_DIR

    # 1. Ingest (pulls the stored game logs first when pushing; local runs use data_dir as-is).
    revision = hf.pull_dataset(data_dir) if push else None
    summary.dataset_revision = revision
    if local_dump is not None:
        dump_dir = local_dump
        sched = schedule.schedule_path(dump_dir, d)
        schedule_file = sched.name if sched else None
        summary.log(f"NIGHTLY {d}: using local dump {dump_dir}")
    else:
        dump_dir = root / config.DUMP_DIR
        fetched = kaggle_daily.download_dump_subset(dump_dir, d)
        schedule_file = fetched["schedule"]
        summary.log(f"NIGHTLY {d}: downloaded {sorted(fetched)}; schedule file {schedule_file}")
    report = kaggle_daily.run_daily(
        dump_dir,
        data_dir,
        d,
        report_path,
        push=push,
        stored_revision=revision,
        schedule_file=schedule_file,
    )
    summary.ingest = {
        k: report[k] for k in ("counts", "seasons_written", "pushed", "dataset_revision_after")
    }
    if report["dataset_revision_after"]:
        summary.dataset_revision = report["dataset_revision_after"]
    c = report["counts"]
    summary.log(
        f"INGEST {d}: new={c['new']} changed={c['changed']} unchanged={c['unchanged']} "
        f"seasons_written={report['seasons_written']} pushed={report['pushed']}"
    )
    game_logs = local.read_game_logs(data_dir)
    dataset_revision = summary.dataset_revision or local.dataset_fingerprint(data_dir)

    # 2. Residuals for yesterday (needs prior products for the rolling window).
    written: list[Path] = []
    if push:
        hf.pull_products(root)
    yesterday = d - timedelta(days=1)
    pred_path = predictions_dir / f"{yesterday.isoformat()}.parquet"
    if pred_path.exists():
        result = residuals.compute(yesterday, pd.read_parquet(pred_path), game_logs, residuals_dir)
        rolling = residuals.append_rolling_line(predictions_dir, result.line)
        written += [residuals_dir / f"{yesterday.isoformat()}.parquet", rolling]
        summary.residuals = {
            k: result.line[k] for k in ("n_predicted", "n_with_actuals", "n_missing_actuals", "mae")
        }
        summary.log(result.message)
    else:
        summary.residuals = {"status": "no_predictions"}
        summary.log(f"RESIDUALS {yesterday}: no predictions file {pred_path}; nothing to score")

    # 3. Slate for today.
    outcome = slate.run_slate(d, game_logs, dump_dir, load_models, dataset_revision)
    summary.log(outcome.message)
    summary.slate = {
        "status": outcome.status.value,
        "n_games": outcome.n_games,
        "n_players": outcome.n_players,
    }
    if outcome.status is slate.SlateStatus.OK:
        parquet, latest = slate.write_outputs(outcome, predictions_dir)
        written += [parquet, latest]

    # 4. Push exactly the product files written tonight.
    if push and written:
        sha = hf.push_products(root, files=written, message=f"Nightly {d}")
        summary.products_pushed = [str(p.relative_to(root)) for p in written]
        summary.products_revision = sha
        summary.log(f"NIGHTLY {d}: pushed {len(written)} product files at {sha}")
    elif written:
        summary.log(f"NIGHTLY {d}: wrote {len(written)} product files (push disabled)")
    else:
        summary.log(f"NIGHTLY {d}: no product files written")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="run date")
    parser.add_argument("--push", action="store_true", help="pull from and push to Hugging Face")
    parser.add_argument("--local-dump", type=Path, default=None, help="skip the Kaggle download")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    d = args.date or datetime.now(UTC).date()
    summary = run(d, push=args.push, root=args.root, local_dump=args.local_dump)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(f"NIGHTLY {d}: summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
