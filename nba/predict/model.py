"""Load the published LightGBM models and score feature rows.

Models are loaded from the Hugging Face model repo at the revision pinned in
`nba.config.MODEL_REVISION` (or from a local directory for tests and offline
runs). lightgbm is imported lazily so the rest of the package imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from nba import config
from nba.features import asof

MODEL_FILE_TEMPLATE = "lgbm_{target}.txt"


@dataclass(frozen=True)
class Models:
    revision: str
    boosters: dict[str, Any]  # target -> object with .predict(X)


def model_file(target: str) -> str:
    return MODEL_FILE_TEMPLATE.format(target=target)


def load_from_dir(models_dir: Path, revision: str = "local") -> Models:
    import lightgbm as lgb

    boosters = {}
    for target in config.TARGETS:
        path = models_dir / model_file(target)
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        boosters[target] = lgb.Booster(model_file=str(path))
    models = Models(revision=revision, boosters=boosters)
    check_feature_names(models)
    return models


def load_from_hub(
    revision: str = config.MODEL_REVISION, repo_id: str = config.HF_MODEL_REPO
) -> Models:
    paths = {
        target: Path(
            hf_hub_download(
                repo_id,
                model_file(target),
                repo_type="model",
                revision=revision,
                token=config.hf_token(),
            )
        )
        for target in config.TARGETS
    }
    # hf_hub_download places every file of one revision in the same snapshot folder.
    folder = paths[config.TARGETS[0]].parent
    return load_from_dir(folder, revision=revision)


def check_feature_names(models: Models) -> None:
    """Raise if a booster was trained on a different feature list than asof produces."""
    for target, booster in models.boosters.items():
        names = getattr(booster, "feature_name", None)
        if names is None:
            continue
        trained = list(names())
        if trained and trained != asof.FEATURE_COLUMNS:
            raise ValueError(f"{target} model features {trained} do not match asof.FEATURE_COLUMNS")


def predict(models: Models, features: pd.DataFrame) -> pd.DataFrame:
    """One `pred_<target>` column per target, aligned to `features`' index."""
    X = features[asof.FEATURE_COLUMNS]
    out = pd.DataFrame(index=features.index)
    for target, booster in models.boosters.items():
        out[f"pred_{target}"] = booster.predict(X)
    return out
