#!/usr/bin/env python3
"""Create isolated synthetic artifacts for offline deployment smoke checks.

These artifacts are not NBA evaluation evidence and are never written to the
serving ``models/`` directory by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURE_NAMES = [
    'PTS_MA5', 'PTS_MA10', 'PTS_MA20', 'REB_MA5', 'REB_MA10', 'REB_MA20',
    'AST_MA5', 'AST_MA10', 'AST_MA20', 'FG_PCT', 'FT_PCT', 'FG3_PCT',
    'MIN', 'GAMES_PLAYED', 'AGE', 'HOME_GAME', 'REST_DAYS', 'BACK_TO_BACK',
    'MATCHUP_DIFFICULTY', 'SEASON_GAME_NUM',
]
TARGETS = ('points', 'rebounds', 'assists')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_synthetic_artifacts(output_dir: Path, *, overwrite: bool = False,
                               n_samples: int = 200) -> Path:
    """Build deterministic smoke-test pipelines and return metadata path."""
    output_dir = output_dir.resolve()
    if output_dir.name == 'models':
        raise ValueError("Refusing to write synthetic artifacts to a serving models directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n_samples, len(FEATURE_NAMES)))
    X[:, 9:12] = np.clip(X[:, 9:12], .2, .95)
    X[:, 12] = np.clip(np.abs(X[:, 12]) * 10 + 25, 15, 40)
    X[:, 15] = rng.integers(0, 2, n_samples)
    X[:, 16] = rng.integers(0, 4, n_samples)
    X[:, 17] = rng.integers(0, 2, n_samples)
    base = np.clip(X[:, 0] * 5 + X[:, 12] * .3 + X[:, 15] * 2 + rng.normal(0, 3, n_samples) + 20, 0, 60)

    hashes = {}
    for target in TARGETS:
        y = base if target == 'points' else np.clip(base * (.3 if target == 'rebounds' else .2) + rng.normal(0, 1, n_samples), 0, 25)
        pipeline = Pipeline([('scale', StandardScaler()), ('model', RandomForestRegressor(n_estimators=20, max_depth=8, random_state=SEED, n_jobs=1))])
        pipeline.fit(X, y)
        path = output_dir / f"synthetic_{target}_pipeline.joblib"
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        joblib.dump(pipeline, path)
        hashes[path.name] = sha256(path)

    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], check=True, capture_output=True, text=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'unavailable', None

    metadata = {
        'source_kind': 'synthetic_fixture',
        'purpose': 'offline artifact-generation smoke test only',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'generator_command': f"python scripts/create_dummy_model.py --output-dir {output_dir}",
        'seed': SEED, 'row_count': n_samples, 'targets': list(TARGETS),
        'feature_contract': FEATURE_NAMES,
        'preprocessing': 'sklearn Pipeline(StandardScaler, RandomForestRegressor)',
        'source_commit': commit, 'working_tree_dirty': dirty,
        'artifact_sha256': hashes,
        'limitations': 'Synthetic training rows; training-fit scores are not NBA forecast evaluation.',
    }
    metadata_path = output_dir / 'provenance.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/synthetic-demo'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()
    print(create_synthetic_artifacts(args.output_dir, overwrite=args.overwrite, n_samples=args.samples))


if __name__ == '__main__':
    main()
