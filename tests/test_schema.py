import pandas as pd
import pytest

from nba import schema


def test_fixture_validates(game_logs: pd.DataFrame) -> None:
    out = schema.validate(game_logs)
    assert list(out.columns) == list(schema.COLUMNS)
    assert out is game_logs


def test_duplicate_key_rejected(game_logs: pd.DataFrame) -> None:
    dup = pd.concat([game_logs, game_logs.iloc[[0]]], ignore_index=True)
    with pytest.raises(schema.SchemaError, match="duplicate"):
        schema.validate(dup)


def test_null_key_rejected(game_logs: pd.DataFrame) -> None:
    bad = game_logs.copy()
    bad.loc[bad.index[0], "game_id"] = pd.NA
    with pytest.raises(schema.SchemaError, match="nulls"):
        schema.validate(bad)


def test_wrong_dtype_rejected(game_logs: pd.DataFrame) -> None:
    bad = game_logs.copy()
    bad["pts"] = bad["pts"].astype("float64")
    with pytest.raises(schema.SchemaError, match="dtype"):
        schema.validate(bad)


def test_missing_and_extra_columns_rejected(game_logs: pd.DataFrame) -> None:
    with pytest.raises(schema.SchemaError, match="missing=\\['pf'\\]"):
        schema.validate(game_logs.drop(columns=["pf"]))
    extra = game_logs.assign(fantasy=1.0)
    with pytest.raises(schema.SchemaError, match="extra=\\['fantasy'\\]"):
        schema.validate(extra)


def test_coerce_orders_and_casts(game_logs: pd.DataFrame) -> None:
    shuffled = game_logs[list(reversed(game_logs.columns))].copy()
    shuffled["game_date"] = shuffled["game_date"].dt.strftime("%Y-%m-%d")
    shuffled["player_id"] = shuffled["player_id"].astype("string")
    out = schema.coerce(shuffled)
    assert list(out.columns) == list(schema.COLUMNS)
    schema.validate(out)
    pd.testing.assert_frame_equal(out, game_logs)
