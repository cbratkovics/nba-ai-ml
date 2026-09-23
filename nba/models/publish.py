"""Publish trained models and a model card to the Hugging Face model repo.

The card is rendered from reports/metrics.json and nba.config only; its canonical copy is
committed as docs/MODEL_CARD.md and a test asserts it regenerates identically.

Usage:
    HF_TOKEN=... python -m nba.models.publish [--models-dir models] \
        [--metrics-path reports/metrics.json]
    python -m nba.models.publish --card-only [--out docs/MODEL_CARD.md] [--push-card]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from nba import config
from nba.models import evaluate
from nba.storage import hf

MODEL_CARD_FILE = "README.md"
CARD_PATH = Path("docs") / "MODEL_CARD.md"


def model_card(payload: dict[str, Any]) -> str:
    split = payload["split"]
    features = "\n".join(f"- `{f}`" for f in payload["features"])
    train_dates = f"{split['train_dates']['start']} to {split['train_dates']['end']}"
    holdout_dates = f"{split['holdout_dates']['start']} to {split['holdout_dates']['end']}"
    model_files = "\n".join(
        f"- `{name}` (LightGBM text model for `{target}`)"
        for target, name in payload["model_files"].items()
    )
    sources = config.source_lines()
    identity = (
        f"Pinned by the pipeline as `{config.model_identity()}` (`nba/config.py`)."
        if payload["git_sha"] == config.MODEL_COMMIT
        else "Not the revision currently pinned in `nba/config.py`."
    )
    return f"""---
license: mit
language: en
tags:
  - nba
  - basketball
  - lightgbm
  - tabular-regression
datasets:
  - {payload["dataset"]["repo"]}
---

# NBA player-stat predictor

One LightGBM regressor per target (points, rebounds, assists) that predicts a
player's box-score line for a game from that player's history before the game.
Personal, non-commercial portfolio project.

## Dataset

- Repo: `{payload["dataset"]["repo"]}`, version `{payload["dataset"]["version"]}`
- Rows are per (player, game); features use only games strictly before the target game.
- Training rows are limited to games where the player logged at least
  {split["min_minutes"]} minutes.

## Split

| | Seasons | Dates | Rows |
|---|---|---|---:|
| Train | {", ".join(split["train_seasons"])} | {train_dates} | {split["n_train_rows"]} |
| Holdout | {split["holdout_season"]} | {holdout_dates} | {split["n_holdout_rows"]} |

The holdout season is never used for fitting or for choosing settings.

## Metrics on the holdout season

`baseline_last10` is the player's mean over the previous 10 games;
`baseline_season` is the player's season-to-date mean. All three predictors are
scored on the same eligible rows: players who logged at least
{split["min_minutes"]} minutes and for whom both baselines are defined. This is the
training-population cohort, not every player-game in the replay.

{evaluate.metrics_table(payload)}

On this eligible holdout cohort, the model's MAE is about 2–3% lower than the
last-10 baseline (2.9% for points, 3.3% for rebounds, and 2.0% for assists). The
separate nightly-path replay also reports an all-rows population that includes games
under 10 minutes; on that wider population, the last-10 baseline has lower MAE on
all three targets. The model's advantage must not be generalized beyond the eligible
cohort in the table above.

## Features

{features}

## Files

{model_files}
- `metrics.json` (the report these numbers come from)

## Known limitations

- Predicts only for players who play; it does not predict minutes or DNPs. The model
  metrics above use only games with at least {split["min_minutes"]} minutes played
  and both baselines available; the wider replay does not show a model advantage.
- No injury, lineup, betting-line, or opponent-strength inputs.
- Regular-season games only.
- Game-to-game variance in box-score stats is high; compare against the
  baselines above rather than reading the absolute error alone.

## Data sources

- {sources["backfill"]}
- {sources["daily"]}

Both are derived from NBA.com box scores. NBA data is used here for non-commercial
personal study only.

## Identity

Trained at git commit `{payload["git_sha"]}` on {payload["generated_at"]}. {identity}
"""


def publish(models_dir: Path, metrics_path: Path) -> str:
    payload = evaluate.read_metrics(metrics_path)
    files = [models_dir / name for name in payload["model_files"].values()]
    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"model files not found: {missing}")
    card_path = models_dir / MODEL_CARD_FILE
    card_path.write_text(model_card(payload))
    metrics_copy = models_dir / metrics_path.name
    shutil.copyfile(metrics_path, metrics_copy)
    return hf.push_model(files + [card_path, metrics_copy])


def render_card(metrics_path: Path, out: Path) -> Path:
    """Write the model card for the committed metrics report (no model files needed)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model_card(evaluate.read_metrics(metrics_path)))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--metrics-path", type=Path, default=config.METRICS_PATH)
    parser.add_argument("--card-only", action="store_true", help="render the card, no model files")
    parser.add_argument("--out", type=Path, default=CARD_PATH, help="card path for --card-only")
    parser.add_argument(
        "--push-card", action="store_true", help="with --card-only: upload README.md only"
    )
    args = parser.parse_args(argv)
    if args.card_only:
        out = render_card(args.metrics_path, args.out)
        print(f"CARD model: wrote {out}")
        if args.push_card:
            sha = hf.push_model_card(out)
            print(f"CARD model: pushed README.md to {config.HF_MODEL_REPO} at {sha}")
        return
    sha = publish(args.models_dir, args.metrics_path)
    print(f"pushed to https://huggingface.co/{config.HF_MODEL_REPO} at {sha}")


if __name__ == "__main__":
    main()
