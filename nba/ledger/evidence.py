"""Replay evidence for the pre-registered ledger rule (ADR-0030).

Two of the rule's constants cannot be read off a sportsbook line because no line history is
available at $0 (docs/AE_AUDIT.md §9.1): the pre-game eligibility proxy and the maximum
feature staleness. Both are chosen from the 2025-26 replay, the only evaluated season, and
this module writes that evidence as an artifact so the ADR cites keys rather than numbers:

  proxy        for each last-10 minutes floor, how many 2025-26 rows the proxy selects
               (with at least one prior game this season), what share of them are in the
               training population (precision), what share of the training population they
               keep (recall), and the model's and the last-10 baseline's points MAE on the
               replayed rows of that set;
  staleness    the distribution of days since the player's newest input game (the feature
               module's days_rest) in the training population, and the model's and the
               baseline's MAE by staleness bucket.

The rows are the season's game-log rows with the feature module run over the whole history,
joined to the replay residuals for the errors; only rows with a box score and a last-10
baseline score. Nothing here is tuned on 2026-27 and nothing here touches the model.

Usage:
    python -m nba.ledger.evidence [--season 2025-26] [--warehouse data/warehouse]
                                  [--out reports/ledger_rule_evidence_<season>.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.features import asof
from nba.models.evaluate import git_sha
from nba.storage import local

MINUTES_FLOORS: tuple[float, ...] = (0, 10, 12, 15, 18, 20, 24, 28)
STALENESS_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 10, 14, 30, 400)
STALENESS_CUTS: tuple[int, ...] = (7, 10, 14)
REPORT_TEMPLATE = "ledger_rule_evidence_{season}.json"


def season_rows(game_logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per game-log row of the season with the features and the population flags."""
    feats = asof.build_features(game_logs)
    f = feats[feats["season"] == season].copy()
    f["prior_game_season"] = f["games_played_season"] >= 1
    both = pd.Series(True, index=f.index)
    for t in config.TARGETS:
        both &= f[f"{t}_mean_last10"].notna() & f[f"{t}_mean_season"].notna()
    f["min10"] = (f["minutes"] >= config.MIN_MINUTES) & both
    return f


def with_replay_errors(
    rows: pd.DataFrame, residuals: pd.DataFrame, target: str = "pts"
) -> pd.DataFrame:
    """Rows joined to the replay residuals, with the model's and the baseline's absolute error."""
    r = residuals[residuals["has_actual"]]
    r = r[r[f"{target}_mean_last10"].notna()]
    m = rows.merge(
        r[
            [
                "player_id",
                "game_id",
                f"pred_{target}",
                f"actual_{target}",
                f"{target}_mean_last10",
                "in_metrics_population",
            ]
        ],
        on=["player_id", "game_id"],
        how="inner",
        suffixes=("", "_replay"),
    )
    m["ae_model"] = (m[f"actual_{target}"] - m[f"pred_{target}"]).abs()
    m["ae_baseline"] = (m[f"actual_{target}"] - m[f"{target}_mean_last10_replay"]).abs()
    return m


def proxy_table(
    rows: pd.DataFrame, scored: pd.DataFrame, floors=MINUTES_FLOORS
) -> list[dict[str, Any]]:
    n_min10 = int(rows["min10"].sum())
    out = []
    for floor in floors:
        sel = rows["prior_game_season"] & (rows["minutes_mean_last10"] >= floor)
        tp = int((sel & rows["min10"]).sum())
        n = int(sel.sum())
        s = scored[scored["prior_game_season"] & (scored["minutes_mean_last10"] >= floor)]
        out.append(
            {
                "minutes_floor": float(floor),
                "n_selected": n,
                "n_selected_in_min10": tp,
                "precision": tp / n if n else None,
                "recall": tp / n_min10 if n_min10 else None,
                "n_scored": int(len(s)),
                "model_mae": float(s["ae_model"].mean()) if len(s) else None,
                "baseline_last10_mae": float(s["ae_baseline"].mean()) if len(s) else None,
                "min10_share_scored": float(s["in_metrics_population"].mean()) if len(s) else None,
            }
        )
    return out


def staleness_table(rows: pd.DataFrame, scored: pd.DataFrame) -> dict[str, Any]:
    d = rows.loc[rows["min10"], "days_rest"].dropna()
    q = d.quantile([0.5, 0.9, 0.95, 0.99])
    buckets = pd.cut(scored["days_rest"], list(STALENESS_BUCKETS), right=True)
    by_bucket = (
        scored[scored["in_metrics_population"]]
        .groupby(buckets[scored["in_metrics_population"]], observed=True)
        .agg(
            n=("ae_model", "size"),
            model_mae=("ae_model", "mean"),
            baseline_last10_mae=("ae_baseline", "mean"),
        )
    )
    return {
        "definition": (
            "days_rest: days since the player's newest game in the stored logs at prediction "
            "time (the feature module's input staleness); training-population rows"
        ),
        "n": int(len(d)),
        "p50": float(q[0.5]),
        "p90": float(q[0.9]),
        "p95": float(q[0.95]),
        "p99": float(q[0.99]),
        "max": float(d.max()),
        "share_above": {str(c): float((d > c).mean()) for c in STALENESS_CUTS},
        "by_bucket": [
            {
                "days_rest_from_exclusive": float(iv.left),
                "days_rest_to_inclusive": float(iv.right),
                "n": int(row["n"]),
                "model_mae": float(row["model_mae"]),
                "baseline_last10_mae": float(row["baseline_last10_mae"]),
            }
            for iv, row in by_bucket.iterrows()
        ],
    }


def build_report(
    game_logs: pd.DataFrame, residuals: pd.DataFrame, season: str, *, dataset_revision: str | None
) -> dict[str, Any]:
    rows = season_rows(game_logs, season)
    scored = with_replay_errors(rows, residuals)
    return {
        "kind": "ledger_rule_evidence",
        "season": season,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "feature_version": asof.FEATURE_VERSION,
        "model_revision": config.MODEL_REVISION,
        "dataset_revision": dataset_revision,
        "target": "pts",
        "population": {
            "rows": int(len(rows)),
            "min10": int(rows["min10"].sum()),
            "min10_definition": (
                "minutes >= 10 and both baselines defined (at least one earlier game this season)"
            ),
            "scored_rows": int(len(scored)),
        },
        "proxy": {
            "definition": (
                "pre-game proxy for min10: at least one prior game this season and the "
                "player's last-10 minutes mean at or above the floor; both known before tip-off"
            ),
            "table": proxy_table(rows, scored),
        },
        "staleness": staleness_table(rows, scored),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", default=config.HOLDOUT_SEASON)
    parser.add_argument("--warehouse", type=Path, default=Path("data") / "warehouse")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    game_logs = local.read_game_logs(args.warehouse / config.HF_DATASET_PREFIX)
    folder = args.warehouse / config.HF_REPLAY_PREFIX / args.season / "residuals"
    residuals = pd.concat(
        [pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))], ignore_index=True
    )
    manifest = args.warehouse / "load_manifest.json"
    revision = json.loads(manifest.read_text())["dataset_revision"] if manifest.exists() else None
    report = build_report(game_logs, residuals, args.season, dataset_revision=revision)
    out = args.out or config.REPORTS_DIR / REPORT_TEMPLATE.format(season=args.season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    p = report["proxy"]["table"]
    pop = report["population"]
    print(
        f"EVIDENCE {args.season}: {pop['rows']} rows, min10 {pop['min10']}, "
        f"scored {pop['scored_rows']}"
    )
    for row in p:
        print(
            f"  floor {row['minutes_floor']:>4}: selected {row['n_selected']:6d} "
            f"precision {row['precision']:.3f} recall {row['recall']:.3f} "
            f"MAE model {row['model_mae']:.3f} baseline {row['baseline_last10_mae']:.3f}"
        )
    s = report["staleness"]
    print(
        f"  staleness p50 {s['p50']:.0f} p90 {s['p90']:.0f} p99 {s['p99']:.0f}; "
        f"share > 10 days {s['share_above']['10']:.4f}"
    )
    print(f"EVIDENCE wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
