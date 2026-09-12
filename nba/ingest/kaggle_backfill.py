"""One-off backfill of game logs from the Eoin Moore Kaggle dump (version 515).

Dataset: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores
(CC0). Download it manually and pass the extracted directory with --kaggle-dir.
No Kaggle credentials are used or stored here.

All parsing and mapping rules live in `nba.ingest.kaggle_dump` and are shared with
the daily ingest; this module only adds the CLI, the per-season thresholds, and
the dataset-card push. Games.csv and the LeagueSchedule files are not used here.

Usage:
    python -m nba.ingest.kaggle_backfill --kaggle-dir <path> [--out-dir data/game_logs]
                                         [--push] [--show-player 203999]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba import config
from nba.ingest.kaggle_dump import (  # noqa: F401  (re-exported for callers and tests)
    BACKFILL_START,
    BOX_SCORE_COLUMNS,
    BOX_SCORE_DERIVED,
    BOX_SCORE_FILE,
    COUNTING_STATS,
    CUP_FINAL_GAME_ID_PREFIX,
    CUP_FINAL_SUBLABEL,
    CUP_LABEL,
    GAME_DATE_FORMAT,
    GAME_ID_WIDTH,
    KNOWN_MISSING_GAMES,
    TEAM_HISTORY_COLUMNS,
    TEAM_HISTORY_FILE,
    TEAM_HISTORY_LEAGUE,
    TEAM_HISTORY_LEAGUE_COLUMN,
    Prepared,
    ThresholdError,
    backfill,
    check_thresholds,
    game_type_counts,
    is_cup_final,
    is_cup_game,
    is_dnp,
    load_box_scores,
    load_team_histories,
    map_to_schema,
    normalize_game_id,
    parse_minutes,
    prepare,
    resolve_teams,
    season_from_date,
    season_start_year,
    season_summary,
)
from nba.storage import dataset_card, hf, local

__all__ = [
    BACKFILL_START,
    BOX_SCORE_COLUMNS,
    BOX_SCORE_DERIVED,
    BOX_SCORE_FILE,
    COUNTING_STATS,
    CUP_FINAL_GAME_ID_PREFIX,
    CUP_FINAL_SUBLABEL,
    CUP_LABEL,
    GAME_DATE_FORMAT,
    GAME_ID_WIDTH,
    KNOWN_MISSING_GAMES,
    Prepared,
    TEAM_HISTORY_COLUMNS,
    TEAM_HISTORY_FILE,
    TEAM_HISTORY_LEAGUE,
    TEAM_HISTORY_LEAGUE_COLUMN,
    ThresholdError,
    backfill,
    check_thresholds,
    game_type_counts,
    is_cup_final,
    is_cup_game,
    is_dnp,
    load_box_scores,
    load_team_histories,
    map_to_schema,
    normalize_game_id,
    parse_minutes,
    prepare,
    resolve_teams,
    season_from_date,
    season_start_year,
    season_summary,
    "main",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kaggle-dir", type=Path, required=True, help="extracted Kaggle dataset directory"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=config.DATA_DIR, help="where to write Parquet"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help=(
            f"after writing, upload the Parquet files to {config.HF_DATASET_REPO} (needs HF_TOKEN)"
        ),
    )
    parser.add_argument(
        "--show-player", type=int, default=None, help="print this player's last five rows"
    )
    args = parser.parse_args(argv)

    box = load_box_scores(args.kaggle_dir)
    print(f"{BOX_SCORE_FILE}: {len(box)} rows with gameDate >= {BACKFILL_START.date()}")
    print("distinct gameType values:")
    print(game_type_counts(box).to_string())
    print(f"keeping gameType in {list(config.GAME_TYPES)}")

    prepared = prepare(box, load_team_histories(args.kaggle_dir))
    summary = season_summary(prepared)
    for warning in check_thresholds(summary):
        print(f"WARNING: {warning} (known gaps are listed in docs/reconciliation.md)")
    written = local.write_per_season(prepared.game_logs, args.out_dir)
    print(
        "DNP rows (no/zero minutes or populated comment) are dropped; team_by_name counts kept "
        "rows whose team id was empty in the dump; cup_games are NBA Cup group/knockout games "
        "kept; cup_final_dropped are Cup finals excluded:"
    )
    print(summary.to_string())
    for season, path in written.items():
        print(f"{season} -> {path}")

    if args.show_player is not None:
        df = prepared.game_logs
        rows = df[df["player_id"] == args.show_player].sort_values("game_date").tail(5)
        print(f"\nlast five rows for player_id {args.show_player}:")
        print(rows.to_string(index=False) if not rows.empty else "(no rows)")

    if args.push:
        card_text = dataset_card.render_dataset_card(
            hf.fetch_dataset_card(), summary, missing_games=KNOWN_MISSING_GAMES
        )
        card_path = args.out_dir / hf.DATASET_CARD_FILE
        card_path.write_text(card_text)
        sha = hf.push_dataset(args.out_dir, card_path=card_path)
        print(f"pushed to https://huggingface.co/datasets/{config.HF_DATASET_REPO} at {sha}")


if __name__ == "__main__":
    main()
