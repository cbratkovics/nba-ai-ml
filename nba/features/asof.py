"""Point-in-time ("as-of") features.

Every feature for a (player, game) row is computed from that player's games with
game_date strictly before the row's game_date. This is enforced by construction:
each statistic is shifted by one game within the player's chronologically
sorted history before any rolling or expanding aggregate is taken. No column
from the target game itself is used, except `home`, which is fixed by the
schedule before tip-off.

Features
  {stat}_mean_last{w}   rolling mean of the previous w games, for w in ROLLING_WINDOWS
                        and stat in pts, reb, ast, minutes
  {stat}_mean_season    season-to-date mean of previous games in the same season
  games_played_season   number of previous games in the same season
  days_rest             days since the previous game (NaN for the first game)
  back_to_back          1 if the previous game was yesterday
  home                  1 if the player's team is at home
  {stat}_mean_vs_opp    mean of pts/reb/ast in previous games against this opponent
"""

from __future__ import annotations

import pandas as pd

from nba import config, schema

# Statistics that get rolling and season-to-date means.
STAT_COLUMNS: tuple[str, ...] = ("pts", "reb", "ast", "minutes")
# Statistics that get a prior-games-vs-opponent mean.
VS_OPP_STATS: tuple[str, ...] = ("pts", "reb", "ast")

# Non-feature columns carried through for splitting, filtering, and targets.
ID_COLUMNS: tuple[str, ...] = ("player_id", "game_id", "game_date", "season", "team", "opponent")
TARGET_COLUMNS: tuple[str, ...] = ("minutes", "pts", "reb", "ast")


def feature_columns(windows: tuple[int, ...] = config.ROLLING_WINDOWS) -> list[str]:
    cols: list[str] = []
    for stat in STAT_COLUMNS:
        cols.extend(f"{stat}_mean_last{w}" for w in windows)
    cols.extend(f"{stat}_mean_season" for stat in STAT_COLUMNS)
    cols += ["games_played_season", "days_rest", "back_to_back", "home"]
    cols.extend(f"{stat}_mean_vs_opp" for stat in VS_OPP_STATS)
    return cols


FEATURE_COLUMNS: list[str] = feature_columns()


def _prior_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """Mean of the previous `window` values, excluding the current one."""
    return s.shift(1).rolling(window, min_periods=1).mean()


def _prior_expanding_mean(s: pd.Series) -> pd.Series:
    """Mean of all previous values, excluding the current one."""
    return s.shift(1).expanding(min_periods=1).mean()


def build_features(
    game_logs: pd.DataFrame,
    windows: tuple[int, ...] = config.ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Return one row per input row with ID_COLUMNS, TARGET_COLUMNS, and FEATURE_COLUMNS.

    Rows with no prior history have NaN for the history-based features.
    """
    df = schema.validate(game_logs)
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    out = df[list(ID_COLUMNS) + list(TARGET_COLUMNS)].copy()

    by_player = df.groupby("player_id", sort=False)
    by_player_season = df.groupby(["player_id", "season"], sort=False)
    by_player_opp = df.groupby(["player_id", "opponent"], sort=False)

    for stat in STAT_COLUMNS:
        for w in windows:
            out[f"{stat}_mean_last{w}"] = by_player[stat].transform(_prior_rolling_mean, w)
        out[f"{stat}_mean_season"] = by_player_season[stat].transform(_prior_expanding_mean)

    out["games_played_season"] = by_player_season.cumcount().astype("int64")

    days_rest = by_player["game_date"].diff().dt.days.astype("float64")
    out["days_rest"] = days_rest
    out["back_to_back"] = (days_rest == 1).astype("int64")
    out["home"] = df["home"].astype("int64")

    for stat in VS_OPP_STATS:
        out[f"{stat}_mean_vs_opp"] = by_player_opp[stat].transform(_prior_expanding_mean)

    return out[list(ID_COLUMNS) + list(TARGET_COLUMNS) + feature_columns(windows)]


PENDING_COLUMNS: tuple[str, ...] = (
    "player_id",
    "player_name",
    "game_id",
    "game_date",
    "season",
    "team",
    "opponent",
    "home",
)


def features_for_pending(game_logs: pd.DataFrame, pending: pd.DataFrame) -> pd.DataFrame:
    """As-of features for games that have not been played yet.

    `pending` has one row per (player_id, game_id) with PENDING_COLUMNS. Each pending
    row is appended to the history with zeroed statistics so `build_features` can run;
    because every feature is shifted by one game, a pending row's own (zero) stats never
    reach its features, and pending rows for different players cannot see each other.
    Returns ID_COLUMNS + FEATURE_COLUMNS for the pending rows only.
    """
    missing = [c for c in PENDING_COLUMNS if c not in pending.columns]
    if missing:
        raise KeyError(f"pending rows are missing columns: {missing}")
    history = schema.validate(game_logs)
    keys = list(schema.KEY_COLUMNS)
    clash = pending.merge(history[keys], on=keys, how="inner")
    if not clash.empty:
        raise ValueError(f"{len(clash)} pending rows already exist in the game logs")

    filler = pd.DataFrame({c: pending[c] for c in PENDING_COLUMNS})
    for col, dtype in schema.COLUMNS.items():
        if col in filler.columns:
            continue
        filler[col] = 0.0 if dtype == "float64" else 0 if dtype == "int64" else "pending"
    filler = schema.coerce(filler)
    combined = pd.concat([history, filler], ignore_index=True)
    features = build_features(combined)
    pending_keys = pending[keys].drop_duplicates()
    out = features.merge(pending_keys, on=keys, how="inner")
    return out[list(ID_COLUMNS) + FEATURE_COLUMNS].reset_index(drop=True)
