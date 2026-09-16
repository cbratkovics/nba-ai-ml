"""The nightly drift step: check, decide, write drift/<date>.json (ADR-0016 to ADR-0018).

Reads the committed reference and, when it exists, the committed calibration; computes the
window's PSI per feature from the stored game logs with the one feature module; applies the
policy; counts the no-schedule streak from the earlier drift reports the job pulled; writes
one product file the warehouse copies into mart_drift. Never raises for a verdict: a HOLD
is a status in the file, the summary and an issue, and the slate is untouched.

Usage:
    python -m nba.drift.run --date YYYY-MM-DD [--data-dir data/game_logs] [--drift-dir drift]
                            [--slate-status ok|no_games|no_schedule]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.drift import calibrate, check, policy, reference
from nba.models.evaluate import git_sha


def load_history(drift_dir: Path, before: date) -> list[dict[str, Any]]:
    out = []
    if not drift_dir.is_dir():
        return out
    for path in sorted(drift_dir.glob("*.json")):
        try:
            d = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if d < before:
            try:
                report = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            out.append({"date": report.get("date", d.isoformat()), **report})
    return out


def build_report(
    game_logs: pd.DataFrame,
    d: date,
    slate_status: str,
    *,
    ref: dict[str, Any],
    cal: dict[str, Any] | None,
    history: list[dict[str, Any]],
    generated_at: str,
    git_sha: str,
) -> dict[str, Any]:
    feats = check.population_rows(game_logs)
    result = check.check(feats, d, ref)
    streak = policy.no_schedule_streak(history, slate_status)
    verdict = policy.decide(result, calibration=cal, slate_status=slate_status, streak=streak)
    flagged = set(verdict["flagged"])
    features = [
        {"feature": f, "psi": round(v, 6), "flagged": f in flagged}
        for f, v in result["psi"].items()
    ]
    return {
        "kind": "drift_report",
        "date": result["date"],
        "season": result["season"],
        "position": result["position"],
        "reference_mode": result["reference_mode"],
        "feature_version": ref["feature_version"],
        "model_revision": ref["model_revision"],
        "reference_file": reference.reference_path().as_posix(),
        "window": result["window"],
        "generated_at": generated_at,
        "git_sha": git_sha,
        **{k: v for k, v in verdict.items() if k != "flagged"},
        "flagged": verdict["flagged"],
        "features": features,
    }


def run(
    root: Path,
    d: date,
    game_logs: pd.DataFrame,
    slate_status: str,
    *,
    ref: dict[str, Any] | None = None,
    cal: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    ref = ref or reference.load_reference()
    if ref is None:
        return None, None
    cal = cal if cal is not None else calibrate.load_calibration()
    drift_dir = root / config.DRIFT_DIR
    report = build_report(
        game_logs,
        d,
        slate_status,
        ref=ref,
        cal=cal,
        history=load_history(drift_dir, d),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(),
    )
    drift_dir.mkdir(parents=True, exist_ok=True)
    path = drift_dir / f"{d.isoformat()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report, path


def log_line(d: date, report: dict[str, Any]) -> str:
    th = report["thresholds"]
    return (
        f"DRIFT {d}: status={report['status']} position={report['position']} "
        f"reference={report['reference_mode']} rows={report['window']['n_rows']} "
        f"flagged={report['n_flagged']} (psi>={th['psi']}, "
        f"{'calibrated' if th['calibrated'] else 'uncalibrated'}) "
        f"no_schedule_streak={report['no_schedule_streak']}"
    )


def main(argv: list[str] | None = None) -> int:
    from nba.storage import local

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--slate-status", default="no_games")
    args = parser.parse_args(argv)
    report, path = run(args.root, args.date, local.read_game_logs(args.data_dir), args.slate_status)
    if report is None:
        print(f"DRIFT {args.date}: no reference at {reference.reference_path()}; nothing written")
        return 0
    print(log_line(args.date, report))
    print(f"DRIFT {args.date}: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
