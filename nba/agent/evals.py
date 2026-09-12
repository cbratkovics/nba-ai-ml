"""Deterministic checks on agent briefs. No LLM involved.

(a) Grounding: every number that appears in a finding's text must also appear among the
    numeric leaves of that finding's evidence.values or evidence.args (tolerance 0.01;
    the args of the cited tool call are evidence too, e.g. a 30-day window). A brief with any
    failing finding is marked status="ungrounded" and the failing findings are dropped
    from what latest.json shows; the dropped findings are kept under "dropped_findings"
    for inspection.

(b) Golden set: for a handful of replay dates with a known largest points residual, the
    agent's findings must name that player. Results are written to
    reports/agent_evals.json. In CI the loop is replayed from saved traces
    (tests/traces/<date>.trace.json), so no network is needed.

Usage:
    python -m nba.agent.evals [--traces tests/traces] [--golden reports/agent_golden.json]
                              [--out reports/agent_evals.json]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nba import config

TOLERANCE = 0.01
STATUS_UNGROUNDED = "ungrounded"
HYPHENS = "\u2010\u2011\u2012\u2013-"  # Unicode dashes models emit, then the ASCII hyphen last
NUMBER_PATTERN = re.compile(rf"(?<![\w.{HYPHENS}])[-+]?\d[\d,]*(?:\.\d+)?(?![\w{HYPHENS}]|\.\d)")
GOLDEN_PATH = config.REPORTS_DIR / "agent_golden.json"
EVALS_PATH = config.REPORTS_DIR / "agent_evals.json"
TRACES_DIR = Path("tests") / "traces"


# ---------- grounding ----------


def numbers_in_text(text: str) -> list[float]:
    """Numbers mentioned in prose: 12, 12.5, 1,234, -3.2.

    Skipped because they are labels rather than measurements: ISO dates, ten-digit game
    ids, and numbers glued to a word or another number by a hyphen ("30-day", "last-10",
    "2024-25").
    """
    cleaned = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)  # ISO dates are labels, not measurements
    cleaned = re.sub(r"\b\d{10}\b", " ", cleaned)  # game ids
    values = []
    for m in NUMBER_PATTERN.finditer(cleaned):
        try:
            values.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def numeric_leaves(value: Any) -> list[float]:
    """Every number reachable inside a JSON value (dicts, lists, numeric strings)."""
    out: list[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, int | float):
        out.append(float(value))
    elif isinstance(value, str):
        out.extend(numbers_in_text(value))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(numeric_leaves(v))
    elif isinstance(value, list | tuple):
        for v in value:
            out.extend(numeric_leaves(v))
    return out


def is_grounded(finding: dict[str, Any], tolerance: float = TOLERANCE) -> tuple[bool, list[float]]:
    """True when every number in the text is within tolerance of an evidence value.

    Evidence is the cited tool call's returned values plus its arguments, so a window
    length the finding names ("over 30 days") is grounded by args {"days": 30}. Signs
    are ignored so that "32 days ahead" is grounded by days_stale = -32.
    """
    ev = finding.get("evidence", {})
    evidence = [abs(v) for v in numeric_leaves(ev.get("values", {}))]
    evidence += [abs(v) for v in numeric_leaves(ev.get("args", {}))]
    missing = []
    for n in numbers_in_text(finding.get("text", "")):
        if not any(abs(abs(n) - e) <= tolerance for e in evidence):
            missing.append(n)
    return not missing, missing


def apply_grounding(brief: dict[str, Any], tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Drop ungrounded findings and mark the brief when any were dropped."""
    if brief.get("status") != "ok":
        return brief
    kept, dropped = [], []
    for f in brief.get("findings", []):
        ok, missing = is_grounded(f, tolerance)
        if ok:
            kept.append(f)
        else:
            dropped.append({**f, "ungrounded_numbers": missing})
    out = dict(brief)
    out["findings"] = kept
    if dropped:
        out["status"] = STATUS_UNGROUNDED
        out["dropped_findings"] = dropped
    out["grounding"] = {"checked": len(kept) + len(dropped), "dropped": len(dropped)}
    return out


# ---------- golden set ----------


def golden_hit(brief: dict[str, Any], player_name: str, player_id: int | None = None) -> bool:
    """True when any finding names the golden player (by name in text or by id in evidence)."""
    name = player_name.lower()
    for f in brief.get("findings", []):
        if name in f.get("text", "").lower():
            return True
        blob = json.dumps(f.get("evidence", {})).lower()
        if name in blob or (player_id is not None and f'"player_id": {player_id}' in blob):
            return True
    return False


def evaluate_brief(brief: dict[str, Any], golden: dict[str, Any] | None) -> dict[str, Any]:
    checks = [is_grounded(f) for f in brief.get("findings", [])]
    result: dict[str, Any] = {
        "date": brief.get("date"),
        "status": brief.get("status"),
        "n_findings": len(brief.get("findings", [])),
        "grounded_findings": sum(1 for ok, _ in checks),
        "ungrounded_findings": sum(1 for ok, _ in checks if not ok),
        "grounding_pass": all(ok for ok, _ in checks),
        "tool_calls_made": brief.get("tool_calls_made"),
        "model_id": brief.get("model_id"),
        "latency_ms": brief.get("latency_ms"),
    }
    if golden:
        result["golden"] = {
            "player_id": golden["player_id"],
            "player_name": golden["player_name"],
            "resid_pts": golden.get("resid_pts"),
            "pass": golden_hit(brief, golden["player_name"], golden.get("player_id")),
        }
    return result


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {g["date"]: g for g in data.get("dates", [])}


def run_from_traces(
    traces_dir: Path,
    golden: dict[str, dict[str, Any]],
    ctx_factory,
) -> list[dict[str, Any]]:
    """Replay every saved trace through the loop (Groq mocked) and evaluate the briefs."""
    from nba.agent import loop

    results = []
    for path in sorted(traces_dir.glob("*.trace.json")):
        trace = json.loads(path.read_text())
        d = date.fromisoformat(trace["date"])
        chat = loop.ReplayChat(trace)
        ctx = ctx_factory(trace)
        brief, _ = loop.run_agent(ctx, d, chat, run_tool=chat.recorded_tool_runner())
        brief = apply_grounding(brief)
        results.append({**evaluate_brief(brief, golden.get(trace["date"])), "trace": path.name})
    return results


def write_report(results: list[dict[str, Any]], out: Path, mode: str) -> dict[str, Any]:
    golden_results = [r["golden"]["pass"] for r in results if "golden" in r]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "model_id": config.GROQ_MODEL,
        "n_briefs": len(results),
        "grounding_pass": sum(1 for r in results if r["grounding_pass"]),
        "golden_pass": sum(golden_results),
        "golden_total": len(golden_results),
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--traces", type=Path, default=TRACES_DIR)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=EVALS_PATH)
    args = parser.parse_args(argv)
    from nba.agent import tools

    def ctx_factory(trace):
        return tools.ToolContext(
            root=Path("."), data_dir=config.DATA_DIR, run_date=date.fromisoformat(trace["run_date"])
        )

    results = run_from_traces(args.traces, load_golden(args.golden), ctx_factory)
    payload = write_report(results, args.out, mode="trace_replay")
    for r in results:
        g = r.get("golden")
        print(
            f"EVAL {r['date']}: status={r['status']} grounded={r['grounded_findings']}/"
            f"{r['n_findings']}"
            + (f" golden={'pass' if g['pass'] else 'FAIL'} ({g['player_name']})" if g else "")
        )
    print(
        f"EVAL summary: grounding {payload['grounding_pass']}/{payload['n_briefs']}, "
        f"golden {payload['golden_pass']}/{payload['golden_total']}; wrote {args.out}"
    )
    return 0 if payload["golden_pass"] == payload["golden_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
