"""The drift verdict: pure functions over a check result and the run context (ADR-0018).

    insufficient   fewer than DRIFT_MIN_ROWS rows in the window: no PSI, no verdict on drift
    ok             no feature at or above the threshold
    warn           1 .. min_features - 1 features at or above the threshold; or at least
                   min_features while uncalibrated (the rule cannot HOLD until
                   reports/drift_calibration_<season>.json exists); or a no-schedule streak
    hold           calibrated and at least min_features at or above the threshold

HOLD never blocks the slate: the predictions and decisions are still written and pushed;
the verdict is recorded in drift/<date>.json, the run summary, and an issue labelled
`nightly-hold` (opened by the workflow, idempotently). A WARN for the no-schedule streak
opens an issue labelled `nightly-warn`.
"""

from __future__ import annotations

from typing import Any

from nba import config

STATUSES: tuple[str, ...] = ("ok", "warn", "hold", "insufficient")


def thresholds(calibration: dict[str, Any] | None) -> dict[str, Any]:
    if calibration is None:
        return {
            "psi": config.DRIFT_PROVISIONAL_THRESHOLD,
            "min_features": config.DRIFT_MIN_FEATURES,
            "calibrated": False,
            "calibration_file": None,
        }
    return {
        "psi": float(calibration["chosen"]["threshold"]),
        "min_features": int(calibration["chosen"]["min_features"]),
        "calibrated": True,
        "calibration_file": calibration.get("file"),
    }


def flagged_features(psi_values: dict[str, float], threshold: float) -> list[str]:
    above = [f for f, v in psi_values.items() if v >= threshold]
    return sorted(above, key=lambda f: -psi_values[f])


def no_schedule_streak(history: list[dict[str, Any]], today_slate_status: str) -> int:
    """Consecutive runs ending at the no-schedule line, newest first, today included."""
    if today_slate_status != "no_schedule":
        return 0
    streak = 1
    for report in sorted(history, key=lambda r: r["date"], reverse=True):
        if report.get("slate_status") == "no_schedule":
            streak += 1
        else:
            break
    return streak


def decide(
    check: dict[str, Any],
    *,
    calibration: dict[str, Any] | None,
    slate_status: str,
    streak: int,
) -> dict[str, Any]:
    th = thresholds(calibration)
    reasons: list[str] = []
    values = check.get("psi") or {}
    if not values:
        status = "insufficient"
        reasons.append(
            f"{check['window']['n_rows']} rows in the window, fewer than "
            f"{check['window']['min_rows']}: no drift verdict"
        )
        flagged: list[str] = []
    else:
        flagged = flagged_features(values, th["psi"])
        if not flagged:
            status = "ok"
        elif len(flagged) < th["min_features"]:
            status = "warn"
            reasons.append(
                f"{len(flagged)} feature(s) at or above PSI {th['psi']} "
                f"({', '.join(flagged)}), below the {th['min_features']}-feature rule"
            )
        elif th["calibrated"]:
            status = "hold"
            reasons.append(
                f"{len(flagged)} features at or above PSI {th['psi']} "
                f"({', '.join(flagged[:5])}{'...' if len(flagged) > 5 else ''}); "
                f"calibrated rule ({th['calibration_file']})"
            )
        else:
            status = "warn"
            reasons.append(
                f"{len(flagged)} features at or above PSI {th['psi']}, but thresholds are "
                "uncalibrated: the job cannot HOLD until the calibration artifact is committed"
            )
    streak_warn = streak >= config.NO_SCHEDULE_STREAK_WARN
    if streak_warn:
        reasons.append(
            f"{streak} consecutive runs ended at the no-schedule line "
            f"(warn at {config.NO_SCHEDULE_STREAK_WARN}): the dump may have stopped publishing "
            "the season's schedule file"
        )
        if status in ("ok", "insufficient"):
            status = "warn"
    return {
        "status": status,
        "reasons": reasons,
        "thresholds": th,
        "flagged": flagged,
        "n_flagged": len(flagged),
        "slate_status": slate_status,
        "no_schedule_streak": streak,
        "no_schedule_warn": streak_warn,
        "blocks_slate": False,
    }
