"""Local Parquet layout: one file per season under a data directory."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from nba import schema

FILE_PREFIX = "game_logs_"
REVISION_FILE = ".hf_revision"


def season_path(data_dir: Path, season: str) -> Path:
    return data_dir / f"{FILE_PREFIX}{season}.parquet"


def write_per_season(df: pd.DataFrame, data_dir: Path) -> dict[str, Path]:
    """Validate and write one Parquet file per season. Returns season -> path."""
    schema.validate(df)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for season, part in df.groupby("season", sort=True):
        path = season_path(data_dir, str(season))
        part.reset_index(drop=True).to_parquet(path, index=False)
        written[str(season)] = path
    return written


def list_parquet(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob(f"{FILE_PREFIX}*.parquet"))


def read_game_logs(data_dir: Path) -> pd.DataFrame:
    """Read and validate every per-season file in the directory."""
    files = list_parquet(data_dir)
    if not files:
        raise FileNotFoundError(f"no {FILE_PREFIX}*.parquet files in {data_dir}")
    df = pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)
    return schema.validate(schema.coerce(df))


def dataset_fingerprint(data_dir: Path) -> str:
    """Identify the dataset used for a run.

    If the files were pulled from Hugging Face, `pull_dataset` leaves the
    commit sha in `.hf_revision` and that is returned as `hf:<sha>`. Otherwise
    the result is `local:<sha256 prefix>` over the Parquet bytes.
    """
    rev = data_dir / REVISION_FILE
    if rev.exists():
        return f"hf:{rev.read_text().strip()}"
    h = hashlib.sha256()
    for p in list_parquet(data_dir):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return f"local:{h.hexdigest()[:12]}"
