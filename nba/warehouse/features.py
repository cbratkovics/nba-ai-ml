"""Write the feature module's as-of features over the loaded game logs, for the parity proof.

`int_player_game_features` recomputes the 23 `asof_v1` features in SQL over `slv_game_logs`.
The Python module (nba/features/asof.py) stays the one feature implementation and the
reference the model was trained on; this step runs it over the same game-log files the
warehouse loads and writes the result to <root>/features/asof_v1.parquet, which bronze reads
(`brz_features_asof_v1`) so the singular test `assert_int_features_match_feature_module` can
compare the two row for row inside `dbt build` (ADR-0032). Run after `nba.warehouse.load`
(the loader calls it) or alone:

    python -m nba.warehouse.features [--root data/warehouse]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba import config
from nba.features import asof
from nba.storage import local

FEATURES_DIR = "features"


def features_path(root: Path) -> Path:
    return root / FEATURES_DIR / f"{asof.FEATURE_VERSION}.parquet"


def write_features(root: Path) -> tuple[Path, int]:
    """Build the as-of features over <root>/game_logs and write them. Returns (path, rows)."""
    game_logs = local.read_game_logs(root / config.HF_DATASET_PREFIX)
    feats = asof.build_features(game_logs)
    out = feats[list(asof.ID_COLUMNS) + asof.FEATURE_COLUMNS].copy()
    out["feature_version"] = asof.FEATURE_VERSION
    path = features_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path, int(len(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data") / "warehouse")
    args = parser.parse_args(argv)
    path, n = write_features(args.root)
    print(f"FEATURES {asof.FEATURE_VERSION}: {n} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
