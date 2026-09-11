import numpy as np
import pandas as pd
import pytest

from nba.features import asof


def _features_for(features: pd.DataFrame, player_id: int, game_id: str) -> pd.Series:
    row = features[(features["player_id"] == player_id) & (features["game_id"] == game_id)]
    assert len(row) == 1
    return row.iloc[0][asof.FEATURE_COLUMNS]


def _pick_target(game_logs: pd.DataFrame) -> tuple[int, str, pd.Timestamp]:
    """A player's game somewhere in the middle of their history."""
    pid = int(game_logs["player_id"].iloc[0])
    history = game_logs[game_logs["player_id"] == pid].sort_values("game_date")
    row = history.iloc[len(history) // 2]
    return pid, str(row["game_id"]), row["game_date"]


def test_no_future_leakage_by_construction(game_logs: pd.DataFrame) -> None:
    """Changing the target game's own stats, or any later game, must not move its features."""
    pid, gid, gdate = _pick_target(game_logs)
    before = _features_for(asof.build_features(game_logs), pid, gid)

    perturbed = game_logs.copy()
    same_or_later = (perturbed["player_id"] == pid) & (perturbed["game_date"] >= gdate)
    assert same_or_later.sum() > 1
    for col in ("pts", "reb", "ast"):
        perturbed.loc[same_or_later, col] = perturbed.loc[same_or_later, col] + 100
    perturbed.loc[same_or_later, "minutes"] = 48.0

    after = _features_for(asof.build_features(perturbed), pid, gid)
    pd.testing.assert_series_equal(before, after, check_names=False)


def test_past_games_do_change_features(game_logs: pd.DataFrame) -> None:
    """Sanity check for the leakage test: an earlier game must influence the features."""
    pid, gid, gdate = _pick_target(game_logs)
    before = _features_for(asof.build_features(game_logs), pid, gid)

    perturbed = game_logs.copy()
    earlier = (perturbed["player_id"] == pid) & (perturbed["game_date"] < gdate)
    assert earlier.any()
    perturbed.loc[earlier, "pts"] = perturbed.loc[earlier, "pts"] + 100

    after = _features_for(asof.build_features(perturbed), pid, gid)
    assert after["pts_mean_last5"] == pytest.approx(before["pts_mean_last5"] + 100)
    assert after["reb_mean_last5"] == pytest.approx(before["reb_mean_last5"])


def test_feature_values_by_hand(game_logs: pd.DataFrame) -> None:
    features = asof.build_features(game_logs)
    pid = int(game_logs["player_id"].iloc[0])
    hist = game_logs[game_logs["player_id"] == pid].sort_values("game_date").reset_index(drop=True)
    feat = features[features["player_id"] == pid].sort_values("game_date").reset_index(drop=True)

    # First game ever: no history.
    first = feat.iloc[0]
    assert np.isnan(first["pts_mean_last5"])
    assert np.isnan(first["pts_mean_season"])
    assert np.isnan(first["days_rest"])
    assert first["games_played_season"] == 0
    assert first["back_to_back"] == 0

    # Fourth game: mean of the first three, rest days from the third.
    i = 3
    assert feat.loc[i, "pts_mean_last5"] == pytest.approx(hist.loc[:2, "pts"].mean())
    assert feat.loc[i, "pts_mean_last20"] == pytest.approx(hist.loc[:2, "pts"].mean())
    assert feat.loc[i, "minutes_mean_last10"] == pytest.approx(hist.loc[:2, "minutes"].mean())
    assert (
        feat.loc[i, "days_rest"] == (hist.loc[i, "game_date"] - hist.loc[i - 1, "game_date"]).days
    )
    assert feat.loc[i, "home"] == int(hist.loc[i, "home"])

    # Season-to-date resets at a season boundary.
    season_starts = hist.index[hist["season"] != hist["season"].shift(1)].tolist()
    second_season_first = season_starts[1]
    assert np.isnan(feat.loc[second_season_first, "pts_mean_season"])
    assert feat.loc[second_season_first, "games_played_season"] == 0
    # ... but the rolling window does not reset.
    assert not np.isnan(feat.loc[second_season_first, "pts_mean_last5"])

    # Back-to-back flag matches a one-day gap.
    b2b = feat.index[feat["days_rest"] == 1]
    assert len(b2b) > 0
    assert (feat.loc[b2b, "back_to_back"] == 1).all()

    # Prior-games-vs-opponent mean uses only earlier games against that opponent.
    j = len(hist) - 1
    opp = hist.loc[j, "opponent"]
    prior_vs = hist.loc[: j - 1]
    prior_vs = prior_vs[prior_vs["opponent"] == opp]
    if prior_vs.empty:
        assert np.isnan(feat.loc[j, "pts_mean_vs_opp"])
    else:
        assert feat.loc[j, "pts_mean_vs_opp"] == pytest.approx(prior_vs["pts"].mean())


def test_no_same_game_stat_columns_among_features() -> None:
    for col in asof.FEATURE_COLUMNS:
        assert col not in ("pts", "reb", "ast", "minutes")
    assert asof.FEATURE_COLUMNS == asof.feature_columns()
    assert len(asof.FEATURE_COLUMNS) == len(set(asof.FEATURE_COLUMNS))
