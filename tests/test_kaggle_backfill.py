from pathlib import Path

import pandas as pd
import pytest

from nba import config, schema
from nba.ingest import kaggle_backfill
from nba.storage import local
from tests.conftest import GAMES_PER_SEASON, NICKNAMES, PLAYERS_PER_TEAM, SEASON_STARTS


def test_season_from_date() -> None:
    assert kaggle_backfill.season_from_date(pd.Timestamp("2024-10-22")) == "2024-25"
    assert kaggle_backfill.season_from_date(pd.Timestamp("2025-04-13")) == "2024-25"
    assert kaggle_backfill.season_from_date(pd.Timestamp("2026-06-15")) == "2025-26"


def test_mapping_matches_schema(game_logs: pd.DataFrame) -> None:
    schema.validate(game_logs)
    assert set(game_logs["season"]) == set(SEASON_STARTS)
    # Preseason and playoff games are filtered out; the DNP row is dropped.
    regular_games = (GAMES_PER_SEASON - 2) * (len(NICKNAMES) // 2)
    assert game_logs["game_id"].nunique() == regular_games * len(SEASON_STARTS)
    assert len(game_logs) == game_logs["game_id"].nunique() * 2 * PLAYERS_PER_TEAM
    assert 9999 not in set(game_logs["player_id"])
    # Game ids are zero-padded strings, teams are abbreviations, home is boolean.
    assert game_logs["game_id"].str.len().eq(kaggle_backfill.GAME_ID_WIDTH).all()
    assert set(game_logs["team"]) == {"LAL", "BOS", "GSW", "MIA", "DEN", "MIL"}
    assert (game_logs["team"] != game_logs["opponent"]).all()
    assert game_logs["home"].dtype == bool
    assert (game_logs["source"] == config.KAGGLE_SOURCE).all()
    # Every game has exactly one home team's players and one away team's players.
    per_game = game_logs.groupby("game_id")["home"].agg(["sum", "count"])
    assert (per_game["sum"] * 2 == per_game["count"]).all()


def test_unknown_team_is_an_error(kaggle_dir: Path) -> None:
    box = kaggle_backfill.load_box_scores(kaggle_dir)
    # Rename a team that appears in regular-season games (row 0 is preseason and gets filtered).
    box.loc[box["playerteamName"] == "Lakers", "playerteamName"] = "Sonics"
    with pytest.raises(KeyError, match="Sonics"):
        kaggle_backfill.map_to_schema(box, kaggle_backfill.load_schedule(kaggle_dir))


def test_missing_kaggle_column_is_an_error(kaggle_dir: Path) -> None:
    box = kaggle_backfill.load_box_scores(kaggle_dir).drop(columns=["reboundsTotal"])
    with pytest.raises(KeyError, match="reboundsTotal"):
        kaggle_backfill.map_to_schema(box, kaggle_backfill.load_schedule(kaggle_dir))


def test_backfill_writes_one_parquet_per_season(kaggle_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "game_logs"
    written = kaggle_backfill.backfill(kaggle_dir, out_dir)
    assert sorted(written) == sorted(SEASON_STARTS)
    assert all(p.name == f"game_logs_{s}.parquet" for s, p in written.items())

    round_trip = local.read_game_logs(out_dir)
    expected = kaggle_backfill.map_to_schema(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_schedule(kaggle_dir)
    )
    pd.testing.assert_frame_equal(
        round_trip.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True),
        expected.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True),
    )
    assert local.dataset_fingerprint(out_dir).startswith("local:")


def test_cli_prints_row_counts(
    kaggle_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kaggle_backfill.main(["--kaggle-dir", str(kaggle_dir), "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    for season in SEASON_STARTS:
        assert season in out
