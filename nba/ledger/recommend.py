"""The private recommendations view for a captured night (ADR-0030, ADR-0031).

For every captured line with a matched player: the line and prices, the frozen model's
prediction, the edge, the rule's decision and why a row does or does not qualify, with the
plain flag that the model has no availability information. Written to
recommendations/<date>.json in the private lines repo and printed as a table; never to the
public repo or the site. The header carries the coverage counts the brief asks for on every
ledger page: captured nights against game nights so far, and pending bets when the exported
coverage mart is available locally.

Usage:
    python -m nba.ledger.recommend [--date 2026-10-20] [--root .] [--no-push]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.ledger import rule
from nba.lines import schedule as lines_schedule
from nba.lines import store

AVAILABILITY_NOTE = (
    "The model has no availability information: it does not know who is injured, resting or "
    "starting. A line exists because the book priced the player; check availability yourself."
)


def load_lines(root: Path, d: date) -> pd.DataFrame:
    frames = []
    for name in (f"{d.isoformat()}.parquet", f"{d.isoformat()}_manual.parquet"):
        p = root / config.LINES_DIR / name
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_predictions(root: Path, d: date) -> pd.DataFrame:
    p = root / config.PREDICTIONS_DIR / f"{d.isoformat()}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def build(lines: pd.DataFrame, predictions: pd.DataFrame, d: date) -> list[dict[str, Any]]:
    """One row per matched line with the rule applied (nba/ledger/rule.py)."""
    rows: list[dict[str, Any]] = []
    if lines.empty:
        return rows
    preds = predictions.set_index("player_id") if not predictions.empty else None
    for r in lines.itertuples(index=False):
        if pd.isna(r.player_id):
            continue
        pid = int(r.player_id)
        p = preds.loc[pid] if preds is not None and pid in preds.index else None
        stat = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast"}.get(
            r.market
        )
        prediction = float(p[f"pred_{stat}"]) if p is not None and stat else None
        ok, reasons = (
            rule.eligible(
                float(p["games_played_season"]) if p is not None else None,
                float(p["minutes_mean_last10"])
                if p is not None and pd.notna(p["minutes_mean_last10"])
                else None,
                float(p["input_staleness_days"])
                if p is not None and pd.notna(p["input_staleness_days"])
                else None,
            )
            if p is not None
            else (False, ["no prediction for this player tonight"])
        )
        generated = pd.Timestamp(p["generated_at"]) if p is not None else None
        snapshot = pd.Timestamp(r.snapshot_at_utc)
        tip = (
            pd.Timestamp(r.commence_time_utc)
            if pd.notna(r.commence_time_utc)
            else (
                pd.Timestamp(p["tip_off_utc"])
                if p is not None and pd.notna(p["tip_off_utc"])
                else None
            )
        )
        if generated is not None and generated.tzinfo is None:
            generated = generated.tz_localize("UTC")
        if snapshot.tzinfo is None:
            snapshot = snapshot.tz_localize("UTC")
        if tip is not None and tip.tzinfo is None:
            tip = tip.tz_localize("UTC")
        if generated is not None and not generated < snapshot:
            ok, reasons = False, [*reasons, "prediction stamped after the line"]
        if tip is not None and not snapshot < tip:
            ok, reasons = False, [*reasons, "line read after tip-off"]
        side = rule.decide(prediction, float(r.line), r.market) if ok else "pass"
        price = r.over_price if side == "over" else r.under_price if side == "under" else None
        rows.append(
            {
                "player_id": pid,
                "player_name": r.player_name_raw,
                "team": r.team,
                "source": r.source,
                "book": r.book,
                "market": r.market,
                "line": float(r.line),
                "over_price": None if pd.isna(r.over_price) else int(r.over_price),
                "under_price": None if pd.isna(r.under_price) else int(r.under_price),
                "prediction": None if prediction is None else round(prediction, 2),
                "edge": None if prediction is None else round(prediction - float(r.line), 2),
                "eligible": ok,
                "reasons": reasons,
                "decision": side if ok else "pass",
                "price": None if price is None or pd.isna(price) else int(price),
                "tip_off_utc": None if tip is None else tip.isoformat(),
                "availability": "unknown",
            }
        )
    rows.sort(
        key=lambda x: (x["decision"] == "pass", -(abs(x["edge"]) if x["edge"] is not None else 0))
    )
    return rows


def coverage(root: Path) -> dict[str, Any]:
    captures = root / config.CAPTURES_DIR
    captured = (
        sum(
            1
            for p in captures.glob("*.json")
            if json.loads(p.read_text()).get("status") == "captured"
        )
        if captures.exists()
        else 0
    )
    nights = len(list((root / config.PREDICTIONS_DIR).glob("????-??-??.parquet")))
    pending = None
    mart = root / config.HF_GOLD_PREFIX / "mart_ledger_coverage.parquet"
    if mart.exists():
        m = pd.read_parquet(mart)
        if not m.empty:
            pending = int(m["bets_pending"].sum())
    return {"nights_captured": captured, "game_nights_with_slate": nights, "bets_pending": pending}


def write(root: Path, d: date, rows: list[dict[str, Any]], *, push: bool) -> Path:
    out_dir = root / config.RECOMMENDATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": d.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule_version": rule.RULE["version"],
        "rule_hash": rule.rule_hash(),
        "availability_note": AVAILABILITY_NOTE,
        "coverage": coverage(root),
        "n_lines": len(rows),
        "n_bets": sum(1 for r in rows if r["decision"] != "pass"),
        "rows": rows,
    }
    path = out_dir / f"{d.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    if push:
        store.push(root, [path], f"Recommendations {d}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--no-pull", action="store_true")
    args = parser.parse_args(argv)
    d = args.date or datetime.now(UTC).astimezone(lines_schedule.EASTERN).date()
    if not args.no_pull:
        from nba.storage import hf

        hf.pull_products(args.root)
        store.pull(args.root)
    rows = build(load_lines(args.root, d), load_predictions(args.root, d), d)
    path = write(args.root, d, rows, push=not args.no_push)
    bets = [r for r in rows if r["decision"] != "pass"]
    print(f"RECOMMEND {d}: {len(rows)} matched lines, {len(bets)} paper bets; {AVAILABILITY_NOTE}")
    for r in bets:
        print(
            f"  {r['decision']:>5} {r['player_name']:<24} {r['market']:<16} line {r['line']:>5} "
            f"@ {r['price']:>5}  pred {r['prediction']:>6}  edge {r['edge']:>+5}  "
            f"[{r['book']}, {r['source']}]"
        )
    print(f"RECOMMEND {d}: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
