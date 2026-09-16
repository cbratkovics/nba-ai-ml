"""The drift reference: decile bins per model feature over the training rows (ADR-0016).

Rows come from the gold marts: gold.fct_player_game with population = 'min10' in the
training seasons, which is the population the model was evaluated on (ADR-0009). The
features are computed by the one feature module over those rows (never a second SQL
implementation), so the reference is keyed by FEATURE_VERSION and MODEL_REVISION and lives at
reports/drift_reference_<feature_version>_<model_revision>.json (re-included in git). Bin
edges and the season-long `all` block use every training season; the day-aligned expected
proportions use DRIFT_REFERENCE_SEASONS, the training seasons with an earlier season in the
data (the first season's career-long features are truncated by the data start).

The reference is position-aware (ADR-0017). Bin edges are fixed per feature (deciles over
every training row); the expected proportions are stored per season day s (days since the
season's first game date): the training rows of the four seasons whose season day falls in
[s - DRIFT_WINDOW_DAYS, s - 1], exactly the days a check on season day s compares. A
season-long reference is wrong on normal data at every position, not only the opening:
games_played_season is a season counter, the vs-opponent means fill in as opponents are met,
and days_rest shifts around the calendar gaps; even a week bucket biases the counter. The
fantasy-football warehouse hit the same failure at a season boundary. `all` (every training
row) is kept for comparison.

Usage:
    python -m nba.drift.reference [--duckdb .duckdb/nba.duckdb]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba import config
from nba.drift import psi
from nba.features import asof
from nba.models.evaluate import git_sha

DEFAULT_DUCKDB = Path(".duckdb") / "nba.duckdb"

ROWS_SQL = """
select game_id, game_date, season, player_id, player_name, team, opponent, home, minutes,
       pts, reb, ast, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, stl, blk, tov, pf, plus_minus,
       source, population, dataset_revision
from gold.fct_player_game
order by player_id, game_date, game_id
"""


def reference_path(
    feature_version: str = asof.FEATURE_VERSION, model_revision: str = config.MODEL_REVISION
) -> Path:
    return config.REPORTS_DIR / config.DRIFT_REFERENCE_TEMPLATE.format(
        feature_version=feature_version, model_revision=model_revision[:7]
    )


def load_gold_rows(duckdb_path: Path) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        return con.execute(ROWS_SQL).df()
    finally:
        con.close()


def feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Features for every row (history includes every game), with population and season."""
    logs = rows.drop(columns=["population", "dataset_revision"], errors="ignore").copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    feats = asof.build_features(logs)  # carries season and game_date (ID_COLUMNS)
    keys = ["player_id", "game_id"]
    return feats.merge(rows[keys + ["population"]], on=keys)


def training_rows(feats: pd.DataFrame) -> pd.DataFrame:
    return feats[(feats["population"] == "min10") & feats["season"].isin(config.TRAIN_SEASONS)]


def reference_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Rows of the day-aligned reference: the training seasons with an earlier season in the
    data, so career-long features are not truncated by the data start (ADR-0017)."""
    return feats[
        (feats["population"] == "min10") & feats["season"].isin(config.DRIFT_REFERENCE_SEASONS)
    ]


def season_day(feats: pd.DataFrame) -> pd.Series:
    """Days since the first game date of the row's season (0 on opening night)."""
    first = feats.groupby("season")["game_date"].transform("min")
    return (feats["game_date"].dt.normalize() - first.dt.normalize()).dt.days


def day_key(day: int) -> str:
    return f"day_{max(day, 0):03d}"


def window_for_day(train: pd.DataFrame, day: pd.Series, s: int) -> pd.DataFrame:
    """Training rows a check on season day s compares: season days [s - window, s - 1]."""
    return train[(day >= s - config.DRIFT_WINDOW_DAYS) & (day <= s - 1)]


def _round(values: list[float]) -> list[float]:
    return [round(v, 6) for v in values]


def build_reference(
    rows: pd.DataFrame, *, generated_at: str, git_sha: str, source: dict[str, Any]
) -> dict[str, Any]:
    feats = feature_frame(rows)
    train = training_rows(feats)
    ref_rows = reference_rows(feats)
    day = season_day(ref_rows)
    bins = {f: psi.make_bins(train[f]) for f in asof.FEATURE_COLUMNS}
    max_day = int(day.max())
    daily = {}
    for s in range(1, max_day + 2):
        block = window_for_day(ref_rows, day, s)
        daily[day_key(s)] = {
            "n_rows": int(len(block)),
            "expected": {
                f: _round(psi.proportions(block[f], bins[f])) for f in asof.FEATURE_COLUMNS
            },
        }
    return {
        "kind": "drift_reference",
        "feature_version": asof.FEATURE_VERSION,
        "model_revision": config.MODEL_REVISION,
        "model_commit": config.MODEL_COMMIT,
        "generated_at": generated_at,
        "git_sha": git_sha,
        "source": source,
        "population": "min10",
        "seasons": list(config.TRAIN_SEASONS),
        "reference_seasons": list(config.DRIFT_REFERENCE_SEASONS),
        "reference_rows": int(len(ref_rows)),
        "n_bins": config.DRIFT_BINS,
        "window_days": config.DRIFT_WINDOW_DAYS,
        "features": list(asof.FEATURE_COLUMNS),
        "bins": bins,
        "all": {
            "description": "every training-population row of the training seasons",
            "n_rows": int(len(train)),
            "expected": {
                f: _round(psi.proportions(train[f], bins[f])) for f in asof.FEATURE_COLUMNS
            },
        },
        "max_season_day": max_day,
        "daily": daily,
    }


def validate_reference(ref: dict[str, Any]) -> list[str]:
    problems = []
    if ref.get("feature_version") != asof.FEATURE_VERSION:
        problems.append("feature_version differs from the feature module")
    if ref.get("model_revision") != config.MODEL_REVISION:
        problems.append("model_revision differs from config")
    if ref.get("window_days") != config.DRIFT_WINDOW_DAYS:
        problems.append("window_days differs from config")
    daily = ref.get("daily", {})
    if len(daily) < 100:
        problems.append(f"only {len(daily)} season days")
    for name, block in [("all", ref.get("all", {})), *daily.items()]:
        for f in asof.FEATURE_COLUMNS:
            if f not in ref.get("bins", {}):
                problems.append(f"missing bins for {f}")
                break
            expected = block.get("expected", {}).get(f)
            if expected is None:
                problems.append(f"{name}: missing feature {f}")
                continue
            if len(expected) != psi.n_bins(ref["bins"][f]) + 1:
                problems.append(f"{name}.{f}: {len(expected)} proportions for the bins")
            if block["n_rows"] and abs(sum(expected) - 1.0) > 1e-4:
                problems.append(f"{name}.{f}: proportions sum to {sum(expected)}")
    return problems


def write_reference(duckdb_path: Path = DEFAULT_DUCKDB, out: Path | None = None) -> dict:
    rows = load_gold_rows(duckdb_path)
    if rows.empty:
        raise SystemExit(f"DRIFT: gold.fct_player_game is empty in {duckdb_path}")
    ref = build_reference(
        rows,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        source={
            "table": "gold.fct_player_game",
            "duckdb": str(duckdb_path),
            "n_rows": int(len(rows)),
            "dataset_revision": str(rows["dataset_revision"].iloc[0]),
        },
    )
    problems = validate_reference(ref)
    if problems:
        raise SystemExit("DRIFT: reference failed validation: " + "; ".join(problems))
    out = out or reference_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2) + "\n")
    return ref


def load_reference(path: Path | None = None) -> dict[str, Any] | None:
    path = path or (config.REPO_ROOT / reference_path())
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    ref = write_reference(args.duckdb, args.out)
    out = args.out or reference_path()
    print(
        f"DRIFT reference: wrote {out} ({ref['all']['n_rows']} training rows, "
        f"{len(ref['daily'])} season days, {len(ref['features'])} features)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
