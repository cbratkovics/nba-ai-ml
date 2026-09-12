from pathlib import Path

import pandas as pd
import pytest

from nba import config, schema
from nba.ingest import kaggle_backfill
from nba.storage import local
from tests.conftest import (
    ALT_CITY_PLAYER_ID,
    CLOCK_MINUTES,
    CLOCK_PLAYER_ID,
    CUP_GAMES_PER_SEASON,
    DNP_PLAYER_IDS,
    FIXTURE_GAMES_PER_SEASON,
    GAMES_PER_SEASON,
    NO_ID_SEASON,
    PLAYERS_PER_TEAM,
    SEASON_STARTS,
    TEAMS,
)


def test_season_from_date() -> None:
    assert kaggle_backfill.season_from_date(pd.Timestamp("2024-10-22")) == "2024-25"
    assert kaggle_backfill.season_from_date(pd.Timestamp("2024-09-30")) == "2023-24"
    assert kaggle_backfill.season_from_date(pd.Timestamp("2025-04-13")) == "2024-25"
    assert kaggle_backfill.season_from_date(pd.Timestamp("2026-06-15")) == "2025-26"
    assert kaggle_backfill.BACKFILL_START == pd.Timestamp("2021-10-01")


def test_streamed_read_drops_rows_before_cutoff(kaggle_dir: Path) -> None:
    box = kaggle_backfill.load_box_scores(kaggle_dir)
    assert (box["gameDate"] >= kaggle_backfill.BACKFILL_START).all()
    assert 2019 not in set(box["gameDate"].dt.year)
    raw = pd.read_csv(kaggle_dir / kaggle_backfill.BOX_SCORE_FILE)
    assert len(box) == len(raw) - 2 * PLAYERS_PER_TEAM


def test_game_type_counts_lists_distinct_values(kaggle_dir: Path) -> None:
    counts = kaggle_backfill.game_type_counts(kaggle_backfill.load_box_scores(kaggle_dir))
    assert set(counts.index) == {
        "Regular Season",
        "Preseason",
        "Playoffs",
        "NBA Emirates Cup",
        "NBA Cup",
    }
    assert counts["Regular Season"] > counts["Preseason"]


def test_configured_game_type_must_exist(kaggle_dir: Path) -> None:
    box = kaggle_backfill.load_box_scores(kaggle_dir)
    hist = kaggle_backfill.load_team_histories(kaggle_dir)
    with pytest.raises(ValueError, match="Regular season"):
        kaggle_backfill.map_to_schema(box, hist, game_types=("Regular season",))
    # Secondary (Cup) labels may be absent from a given file without failing.
    kaggle_backfill.map_to_schema(box, hist, game_types=("Regular Season", "Not A Label"))


def test_mapping_matches_schema(game_logs: pd.DataFrame) -> None:
    schema.validate(game_logs)
    assert set(game_logs["season"]) == set(SEASON_STARTS)
    # Preseason, playoff, and Cup-final games are filtered out; Cup group games are kept.
    regular_games = (GAMES_PER_SEASON - 2) * (len(TEAMS) // 2) + CUP_GAMES_PER_SEASON
    assert game_logs["game_id"].nunique() == regular_games * len(SEASON_STARTS)
    assert len(game_logs) == game_logs["game_id"].nunique() * 2 * PLAYERS_PER_TEAM
    assert not set(DNP_PLAYER_IDS) & set(game_logs["player_id"])
    # Game ids are zero-padded strings, teams are abbreviations, home is boolean.
    assert game_logs["game_id"].str.len().eq(kaggle_backfill.GAME_ID_WIDTH).all()
    assert set(game_logs["team"]) == {"LAL", "BOS", "GSW", "MIA", "DEN", "OLD", "NEW"}
    assert (game_logs["team"] != game_logs["opponent"]).all()
    assert game_logs["home"].dtype == bool
    assert (game_logs["source"] == "kaggle_v515").all()
    assert (game_logs["source"] == config.KAGGLE_SOURCE).all()
    # Every game has exactly one home team's players and one away team's players.
    per_game = game_logs.groupby("game_id")["home"].agg(["sum", "count"])
    assert (per_game["sum"] * 2 == per_game["count"]).all()


def test_team_abbreviation_follows_season(game_logs: pd.DataFrame) -> None:
    renamed = game_logs[game_logs["player_name"].str.endswith("Renamed")]
    assert set(renamed.loc[renamed["season"] == "2023-24", "team"]) == {"OLD"}
    assert set(renamed.loc[renamed["season"] == "2024-25", "team"]) == {"NEW"}
    assert set(renamed.loc[renamed["season"] == "2025-26", "team"]) == {"NEW"}
    assert "GONE" not in set(game_logs["team"]) | set(game_logs["opponent"])


def test_unknown_team_id_falls_back_to_name(kaggle_dir: Path) -> None:
    box = kaggle_backfill.load_box_scores(kaggle_dir)
    box.loc[box["playerteamId"] == TEAMS[0][0], "playerteamId"] = 424242
    prepared = kaggle_backfill.prepare(box, kaggle_backfill.load_team_histories(kaggle_dir))
    lakers = prepared.game_logs[prepared.game_logs["player_name"].str.endswith("Lakers")]
    assert set(lakers["team"]) == {"LAL"}


def test_ambiguous_team_history_is_an_error(kaggle_dir: Path) -> None:
    hist = kaggle_backfill.load_team_histories(kaggle_dir)
    dup = pd.concat([hist, hist.iloc[[0]]], ignore_index=True)
    with pytest.raises(KeyError, match="multiple active"):
        kaggle_backfill.map_to_schema(kaggle_backfill.load_box_scores(kaggle_dir), dup)


def test_missing_kaggle_column_is_an_error(kaggle_dir: Path) -> None:
    path = kaggle_dir / kaggle_backfill.BOX_SCORE_FILE
    pd.read_csv(path).drop(columns=["reboundsTotal"]).to_csv(path, index=False)
    with pytest.raises(KeyError, match="reboundsTotal"):
        kaggle_backfill.load_box_scores(kaggle_dir)


def test_backfill_writes_one_parquet_per_season(kaggle_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "game_logs"
    written = kaggle_backfill.backfill(kaggle_dir, out_dir)
    assert sorted(written) == sorted(SEASON_STARTS)
    assert all(p.name == f"game_logs_{s}.parquet" for s, p in written.items())

    round_trip = local.read_game_logs(out_dir)
    expected = kaggle_backfill.map_to_schema(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_team_histories(kaggle_dir)
    )
    pd.testing.assert_frame_equal(
        round_trip.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True),
        expected.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True),
    )
    assert local.dataset_fingerprint(out_dir).startswith("local:")


def test_cli_prints_summary(
    kaggle_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pid = 1000
    args = ["--kaggle-dir", str(kaggle_dir), "--out-dir", str(tmp_path / "out")]
    kaggle_backfill.main(args + ["--show-player", str(pid)])
    out = capsys.readouterr().out
    assert "distinct gameType values" in out and "Preseason" in out
    assert "dnp_dropped" in out
    for season in SEASON_STARTS:
        assert season in out
    assert f"last five rows for player_id {pid}" in out


def test_minutes_parse_decimal_and_clock() -> None:
    parsed = kaggle_backfill.parse_minutes(
        pd.Series(["39.166666", "23:21", "12:30", "", None, "0", " 7.5 "])
    )
    expected = [39.166666, 23 + 21 / 60, 12.5, float("nan"), float("nan"), 0.0, 7.5]
    assert parsed.dtype == "float64"
    for got, want in zip(parsed.tolist(), expected, strict=True):
        assert (pd.isna(got) and pd.isna(want)) or got == pytest.approx(want)


def test_clock_minutes_row_is_kept_and_parsed(game_logs: pd.DataFrame) -> None:
    rows = game_logs[(game_logs["player_id"] == CLOCK_PLAYER_ID)]
    assert CLOCK_MINUTES in set(rows["minutes"].round(6))


def test_dnp_rows_dropped_and_counted_per_season(kaggle_dir: Path) -> None:
    prepared = kaggle_backfill.prepare(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_team_histories(kaggle_dir)
    )
    # One no-minutes row per season; the second season also has a 0-minute row and a
    # played-but-commented row.
    assert prepared.dnp_per_season.to_dict() == {"2023-24": 1, "2024-25": 3, "2025-26": 1}
    assert not set(DNP_PLAYER_IDS) & set(prepared.game_logs["player_id"])
    assert (prepared.game_logs["minutes"] > 0).all()

    summary = kaggle_backfill.season_summary(prepared)
    assert list(summary.columns) == [
        "rows",
        "games",
        "first_game",
        "last_game",
        "dnp_dropped",
        "team_by_name",
        "cup_games",
        "cup_final_dropped",
    ]
    assert summary["dnp_dropped"].to_dict() == {"2023-24": 1, "2024-25": 3, "2025-26": 1}
    expected_games = (GAMES_PER_SEASON - 2) * (len(TEAMS) // 2) + CUP_GAMES_PER_SEASON
    assert (summary["games"] == expected_games).all()


def test_is_dnp_rules() -> None:
    minutes = pd.Series([None, 0.0, 12.0, 30.0, 5.0])
    comment = pd.Series([None, "", "", "DNP - Coach's Decision", "  "])
    assert kaggle_backfill.is_dnp(minutes, comment).tolist() == [True, True, False, True, False]


def test_teams_resolved_by_name_when_ids_are_empty(kaggle_dir: Path) -> None:
    prepared = kaggle_backfill.prepare(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_team_histories(kaggle_dir)
    )
    logs = prepared.game_logs
    no_id = logs[logs["season"] == NO_ID_SEASON]
    assert len(no_id) > 0
    assert set(no_id["team"]) == {"LAL", "BOS", "GSW", "MIA", "DEN", "OLD"}
    # Every kept row of the id-less season was resolved by name; one row in 2024-25 too.
    counts = prepared.name_resolved_per_season.to_dict()
    assert counts[NO_ID_SEASON] == len(no_id)
    assert counts["2024-25"] == 1
    assert counts["2025-26"] == 0
    alt = logs[(logs["player_id"] == ALT_CITY_PLAYER_ID) & (logs["season"] == "2024-25")]
    assert set(alt["team"]) == {"LAL"}


def test_non_nba_history_rows_are_ignored(kaggle_dir: Path) -> None:
    hist = kaggle_backfill.load_team_histories(kaggle_dir)
    assert "MAD" not in set(hist["teamAbbrev"])
    assert (hist["teamAbbrev"].str.len() == 3).all() or "LAL" in set(hist["teamAbbrev"])


def test_unresolvable_team_is_an_error(kaggle_dir: Path) -> None:
    box = kaggle_backfill.load_box_scores(kaggle_dir)
    lakers = box["playerteamName"] == "Lakers"
    box.loc[lakers, "playerteamId"] = 424242
    box.loc[lakers, "playerteamName"] = "Sonics"
    with pytest.raises(KeyError, match="could not resolve teams"):
        kaggle_backfill.map_to_schema(box, kaggle_backfill.load_team_histories(kaggle_dir))


def test_cup_group_games_kept_and_finals_dropped(kaggle_dir: Path) -> None:
    prepared = kaggle_backfill.prepare(
        kaggle_backfill.load_box_scores(kaggle_dir), kaggle_backfill.load_team_histories(kaggle_dir)
    )
    logs = prepared.game_logs
    # Every season keeps its Cup group games and drops exactly one final, whether the
    # final was labelled "NBA Cup" with a 006 game id or "Regular Season"/"Championship".
    assert prepared.cup_games_per_season.to_dict() == dict.fromkeys(
        SEASON_STARTS, CUP_GAMES_PER_SEASON
    )
    assert prepared.cup_final_per_season.to_dict() == dict.fromkeys(SEASON_STARTS, 1)
    assert not logs["game_id"].str.startswith("006").any()
    box = kaggle_backfill.load_box_scores(kaggle_dir)
    finals = box[kaggle_backfill.is_cup_final(box["gameId"].str.zfill(10), box["gameSubLabel"])]
    assert finals["gameId"].nunique() == len(SEASON_STARTS)
    assert not set(finals["gameId"].str.zfill(10)) & set(logs["game_id"])
    group = box[box["gameType"] == "NBA Emirates Cup"]
    assert set(group["gameId"].str.zfill(10)) <= set(logs["game_id"])


def test_cup_final_rules() -> None:
    ids = pd.Series(["0062300001", "0022400999", "0022300010", "0062500001"])
    subs = pd.Series([None, "Championship", "East Group A", "Championship"])
    assert kaggle_backfill.is_cup_final(ids, subs).tolist() == [True, True, False, True]
    assert config.GAME_TYPES[0] == "Regular Season"
    assert {"NBA Emirates Cup", "Emirates NBA Cup", "NBA Cup", "in-season-knockout"} <= set(
        config.GAME_TYPES
    )


def _summary(rows: dict[str, int], games: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"rows": rows, "games": games},
        index=pd.Index(sorted(rows), name="season"),
    )


def test_thresholds_hard_stop_below_minimums() -> None:
    summary = _summary({"2021-22": 25_000, "2022-23": 19_999}, {"2021-22": 1230, "2022-23": 1230})
    with pytest.raises(kaggle_backfill.ThresholdError, match="2022-23: 19999 rows < 20000"):
        kaggle_backfill.check_thresholds(summary, min_rows=20_000, min_games=1_200, full_games=1230)
    summary = _summary({"2021-22": 25_000, "2022-23": 25_000}, {"2021-22": 1230, "2022-23": 1199})
    with pytest.raises(kaggle_backfill.ThresholdError, match="2022-23: 1199 games < 1200"):
        kaggle_backfill.check_thresholds(summary, min_rows=20_000, min_games=1_200, full_games=1230)


def test_thresholds_warn_on_incomplete_but_acceptable_season() -> None:
    summary = _summary({"2023-24": 26_095, "2024-25": 26_158}, {"2023-24": 1230, "2024-25": 1223})
    warnings = kaggle_backfill.check_thresholds(
        summary, min_rows=20_000, min_games=1_200, full_games=1230
    )
    assert warnings == ["2024-25: 1223 games, expected 1230"]


def test_thresholds_use_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MIN_ROWS_PER_SEASON", 5)
    monkeypatch.setattr(config, "MIN_GAMES_PER_SEASON", 1)
    monkeypatch.setattr(config, "FULL_SEASON_GAMES", 3)
    summary = _summary({"2021-22": 10}, {"2021-22": 2})
    assert kaggle_backfill.check_thresholds(summary) == ["2021-22: 2 games, expected 3"]
    monkeypatch.setattr(config, "MIN_ROWS_PER_SEASON", 11)
    with pytest.raises(kaggle_backfill.ThresholdError):
        kaggle_backfill.check_thresholds(summary)


def test_cli_hard_stops_and_warns(
    kaggle_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["--kaggle-dir", str(kaggle_dir), "--out-dir", str(tmp_path / "out")]
    monkeypatch.setattr(config, "FULL_SEASON_GAMES", FIXTURE_GAMES_PER_SEASON + 1)
    kaggle_backfill.main(args)
    out = capsys.readouterr().out
    assert f"WARNING: 2023-24: {FIXTURE_GAMES_PER_SEASON} games, expected" in out
    assert (tmp_path / "out" / "game_logs_2023-24.parquet").exists()

    monkeypatch.setattr(config, "MIN_GAMES_PER_SEASON", FIXTURE_GAMES_PER_SEASON + 1)
    with pytest.raises(kaggle_backfill.ThresholdError):
        kaggle_backfill.main(["--kaggle-dir", str(kaggle_dir), "--out-dir", str(tmp_path / "out2")])
    assert not (tmp_path / "out2").exists()


def test_known_missing_games_are_well_formed() -> None:
    games = kaggle_backfill.KNOWN_MISSING_GAMES
    assert len(games) == 7
    assert all(g["game_id"].startswith("002") and len(g["game_id"]) == 10 for g in games)
    assert all(g["season"] == "2024-25" for g in games)
    assert len({g["game_id"] for g in games}) == 7
