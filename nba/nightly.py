"""Nightly orchestration: ingest -> residuals for yesterday -> slate for today -> push.

One process so the files written by each step are known exactly and only those are
uploaded. Steps: ingest -> residuals for yesterday -> analyst brief for yesterday ->
slate for today -> decisions for today -> drift check -> push. Outcomes that are not errors
exit 0 with one explicit line each:

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
from nba.agent import loop as agent_loop
from nba.agent import tools as agent_tools
from nba.decisions import decide
from nba.decisions import evaluate as policy_evaluate
from nba.drift import reference as drift_reference
from nba.drift import run as drift_run
from nba.ingest import kaggle_daily, schedule
from nba.predict import model, residuals, slate
from nba.storage import hf, local

SUMMARY_PATH = Path("data") / "nightly_summary.json"
# Consecutive slated nights without a single box score before the source counts as stalled.
STALL_WARN = 2


def stall_streak(lines: list[dict[str, Any]]) -> int:
    """Consecutive residual lines, newest first, that predicted players and found no actuals."""
    streak = 0
    for line in sorted(lines, key=lambda ln: ln.get("date", ""), reverse=True):
        if (
            int(line.get("n_predicted", 0) or 0) > 0
            and int(line.get("n_with_actuals", 0) or 0) == 0
        ):
            streak += 1
        else:
            break
    return streak


def stall_status(predictions_dir: Path) -> dict[str, Any]:
    path = predictions_dir / residuals.ROLLING_FILE
    lines: list[dict[str, Any]] = []
    if path.exists():
        lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    streak = stall_streak(lines)
    return {
        "stall_streak": streak,
        "stalled": streak >= STALL_WARN,
        "last_residual_date": lines[-1]["date"] if lines else None,
    }


@dataclass
class NightlySummary:
    date: str
    push: bool
    dataset_revision: str | None = None
    ingest: dict[str, Any] = field(default_factory=dict)
    residuals: dict[str, Any] = field(default_factory=dict)
    agent: dict[str, Any] = field(default_factory=dict)
    slate: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
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
    # The daily report is a product too, so the analyst tools can read any past date.
    daily_reports_dir = root / config.DAILY_REPORTS_DIR
    daily_reports_dir.mkdir(parents=True, exist_ok=True)
    report_copy = daily_reports_dir / f"{d.isoformat()}.json"
    report_copy.write_text(json.dumps(report, indent=2) + "\n")
    written: list[Path] = [report_copy]

    # 2. Residuals for yesterday (needs prior products for the rolling window).
    if push:
        hf.pull_products(root)
    yesterday = d - timedelta(days=1)
    pred_path = predictions_dir / f"{yesterday.isoformat()}.parquet"
    if pred_path.exists():
        result = residuals.compute(yesterday, pd.read_parquet(pred_path), game_logs, residuals_dir)
        rolling = residuals.append_rolling_line(predictions_dir, result.line)
        written += [residuals_dir / f"{yesterday.isoformat()}.parquet", rolling]
        summary.residuals = {
            k: result.line[k]
            for k in (
                "n_predicted",
                "n_with_actuals",
                "n_missing_actuals",
                "n_unresolved_provisional_games",
                "mae",
            )
        }
        summary.log(result.message)
    else:
        summary.residuals = {"status": "no_predictions"}
        summary.log(f"RESIDUALS {yesterday}: no predictions file {pred_path}; nothing to score")

    # 2c. Source stall (ADR-0031): nights that were slated but whose box scores never arrived,
    # counted from the rolling residual lines; distinct from "no schedule". Bets stay pending
    # and settle whenever the dump resumes; the workflow opens one issue at STALL_WARN nights.
    summary.source = stall_status(predictions_dir)
    if summary.source["stall_streak"] >= STALL_WARN:
        summary.log(
            f"SOURCE {d}: box scores missing for {summary.source['stall_streak']} consecutive "
            "slated nights (source stalled)"
        )

    # 2b. Analyst brief for yesterday. Never allowed to fail the job: the loop catches
    # every Groq/limit error and writes an agent_unavailable brief; anything else that
    # escapes is caught here and logged.
    try:
        ctx = agent_tools.ToolContext(
            root=root,
            data_dir=data_dir,
            run_date=d,
            dump_dir=dump_dir if dump_dir.exists() else None,
        )
        known_dates = None
        if push:
            try:
                known_dates = hf.list_brief_dates()
            except Exception as exc:  # noqa: BLE001 - the listing is a convenience, never fatal
                summary.log(f"AGENT {yesterday}: brief listing unavailable ({type(exc).__name__})")
        brief, brief_files = agent_loop.run_and_write(
            ctx, yesterday, root / config.BRIEF_DIR, known_dates=known_dates
        )
        written += brief_files
        summary.agent = {
            "status": brief["status"],
            "tool_calls_made": brief["tool_calls_made"],
            "findings": len(brief["findings"]),
            "model_id": brief["model_id"],
            "latency_ms": brief["latency_ms"],
        }
        summary.log(
            f"AGENT {yesterday}: status={brief['status']} tool_calls={brief['tool_calls_made']} "
            f"findings={len(brief['findings'])} latency_ms={brief['latency_ms']}"
        )
    except Exception as exc:  # noqa: BLE001 - the agent must never fail the nightly job
        summary.agent = {"status": "agent_unavailable", "error": f"{type(exc).__name__}: {exc}"}
        summary.log(f"AGENT {yesterday}: agent_unavailable ({type(exc).__name__}: {exc})")

    # 3. Slate for today. Without a schedule file for the season, The Odds API's free events
    # endpoint stands in (ADR-0029); the Kaggle file wins whenever it exists.
    fallback = None
    schedule_source = "kaggle_dump" if schedule.schedule_path(dump_dir, d) else None
    if schedule_source is None and config.ODDS_API_KEY:
        try:
            from nba.lines import capture as lines_capture
            from nba.lines import odds_api
            from nba.lines import schedule as lines_schedule

            start, end = lines_capture.eastern_day_bounds_utc(d)
            events = odds_api.OddsApi().events(commence_from=start, commence_to=end).data
            fallback = lines_schedule.fallback_schedule(events)
            schedule_source = lines_schedule.SOURCE_ODDS_API
            summary.log(f"SLATE {d}: schedule from The Odds API events ({len(fallback)} games)")
        except Exception as exc:  # noqa: BLE001 - the fallback must never fail the night
            summary.log(f"SLATE {d}: schedule fallback unavailable ({type(exc).__name__}: {exc})")
    outcome = slate.run_slate(
        d, game_logs, dump_dir, load_models, dataset_revision, fallback=fallback
    )
    summary.log(outcome.message)
    summary.slate = {
        "status": outcome.status.value,
        "n_games": outcome.n_games,
        "n_players": outcome.n_players,
        "schedule_source": schedule_source
        if outcome.status is not slate.SlateStatus.NO_SCHEDULE
        else None,
    }
    if outcome.status is slate.SlateStatus.OK:
        parquet, latest = slate.write_outputs(outcome, predictions_dir)
        written += [parquet, latest]
        # 3b. Decisions for today's slate under the committed policy (ADR-0001).
        artifact = decide.load_artifact()
        if artifact is None:
            summary.decisions = {"status": "no_policy_report"}
            summary.log(
                f"DECISIONS {d}: no policy report at {policy_evaluate.report_path()}; none written"
            )
        else:
            decision_files = decide.write_outputs(
                outcome.predictions, d, root / config.DECISIONS_DIR, artifact
            )
            written += decision_files
            payload = json.loads(decision_files[0].read_text())
            summary.decisions = {
                "status": "ok",
                "policy_season": payload["policy"]["season"],
                "n_calls": payload["n_calls"],
            }
            summary.log(
                f"DECISIONS {d}: {payload['n_players']} players, calls per target "
                f"(training-population policy) {payload['n_calls']['min10']}"
            )

    # 3c. Drift: the window before today against the day-aligned reference (ADR-0016 to
    # ADR-0018). A HOLD is recorded, never a reason to skip the slate or the push.
    report, drift_path = drift_run.run(root, d, game_logs, outcome.status.value)
    if report is None:
        summary.drift = {"status": "no_reference"}
        summary.log(
            f"DRIFT {d}: no reference at {drift_reference.reference_path()}; nothing written"
        )
    else:
        written.append(drift_path)
        summary.drift = {
            k: report[k]
            for k in (
                "status",
                "reasons",
                "thresholds",
                "flagged",
                "n_flagged",
                "position",
                "reference_mode",
                "no_schedule_streak",
                "no_schedule_warn",
                "blocks_slate",
            )
        }
        summary.drift["n_rows"] = report["window"]["n_rows"]
        summary.log(drift_run.log_line(d, report))

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
