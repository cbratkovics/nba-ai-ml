"""Canonical game-log schema: one row per (player_id, game_id)."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

# Column name -> canonical dtype. Order is the canonical column order.
COLUMNS: dict[str, str] = {
    "game_id": "string",
    "game_date": "datetime64[ns]",
    "season": "string",
    "player_id": "int64",
    "player_name": "string",
    "team": "string",
    "opponent": "string",
    "home": "bool",
    "minutes": "float64",
    "pts": "int64",
    "reb": "int64",
    "ast": "int64",
    "fgm": "int64",
    "fga": "int64",
    "fg3m": "int64",
    "fg3a": "int64",
    "ftm": "int64",
    "fta": "int64",
    "oreb": "int64",
    "dreb": "int64",
    "stl": "int64",
    "blk": "int64",
    "tov": "int64",
    "pf": "int64",
    "plus_minus": "int64",
    "source": "string",
}

KEY_COLUMNS: tuple[str, ...] = ("player_id", "game_id")

# Columns that must never be null.
NOT_NULL_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "season",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "home",
    "minutes",
    "source",
)


class SchemaError(ValueError):
    """Raised when a frame does not match the canonical game-log schema."""


def _dtype_ok(series: pd.Series, expected: str) -> bool:
    if expected == "string":
        return ptypes.is_string_dtype(series)
    if expected == "datetime64[ns]":
        return ptypes.is_datetime64_any_dtype(series)
    if expected == "bool":
        return ptypes.is_bool_dtype(series)
    if expected == "int64":
        return ptypes.is_integer_dtype(series)
    if expected == "float64":
        return ptypes.is_float_dtype(series)
    raise ValueError(f"unknown expected dtype {expected!r}")


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Select the canonical columns in order and cast them to canonical dtypes.

    Integer columns must already be free of nulls; callers decide how to fill
    them (the Kaggle backfill fills counting stats with 0 for rows that played).
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for col, dtype in COLUMNS.items():
        s = df[col]
        if dtype == "datetime64[ns]":
            out[col] = pd.to_datetime(s).dt.tz_localize(None).dt.normalize()
        elif dtype == "bool":
            if s.isna().any():
                raise SchemaError(f"column {col!r} has nulls; cannot cast to bool")
            out[col] = s.astype("bool")
        elif dtype == "int64":
            if s.isna().any():
                raise SchemaError(f"column {col!r} has nulls; cannot cast to int64")
            out[col] = pd.to_numeric(s).astype("int64")
        elif dtype == "float64":
            out[col] = pd.to_numeric(s).astype("float64")
        else:
            out[col] = s.astype("string")
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the frame matches the schema and return it unchanged.

    Checks: exact column set, dtypes, no nulls in key/identity columns, and
    uniqueness of (player_id, game_id).
    """
    expected = list(COLUMNS)
    actual = list(df.columns)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SchemaError(f"column mismatch; missing={missing} extra={extra}")

    bad = [
        f"{c} is {df[c].dtype}, expected {t}" for c, t in COLUMNS.items() if not _dtype_ok(df[c], t)
    ]
    if bad:
        raise SchemaError("dtype mismatch: " + "; ".join(bad))

    null_counts = df[list(NOT_NULL_COLUMNS)].isna().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        raise SchemaError(f"nulls in non-nullable columns: {nulls.to_dict()}")

    dupes = df.duplicated(subset=list(KEY_COLUMNS))
    if dupes.any():
        sample = df.loc[dupes, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise SchemaError(f"{int(dupes.sum())} duplicate (player_id, game_id) rows, e.g. {sample}")

    return df
