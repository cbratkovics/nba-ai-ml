import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nba import config, schema
from nba.ingest import kaggle_backfill, kaggle_daily, kaggle_dump
from nba.storage import hf, local


# ---------- download ----------
class _Resp:
    def __init__(self, status: int, body: bytes = b""):
        self.status_code = status
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_raw_zip_and_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("TeamHistories.csv", "teamId,teamAbbrev\n1,AAA\n")
    bodies = {
        "PlayerStatistics.csv": _Resp(200, b"a,b\n1,2\n"),
        "TeamHistories.csv": _Resp(200, zipped.getvalue()),
        "LeagueSchedule26_27.csv": _Resp(404),
    }

    def fake_get(url, auth, stream, timeout):
        calls.append((url, auth))
        return bodies[url.rsplit("/", 1)[1]]

    monkeypatch.setattr(kaggle_daily.requests, "get", fake_get)
    raw = kaggle_daily.download_kaggle_file("PlayerStatistics.csv", tmp_path, username="u", key="k")
    assert raw == tmp_path / "PlayerStatistics.csv" and raw.read_bytes() == b"a,b\n1,2\n"
    unzipped = kaggle_daily.download_kaggle_file(
        "TeamHistories.csv", tmp_path, username="u", key="k"
    )
    assert unzipped.read_text() == "teamId,teamAbbrev\n1,AAA\n"
    assert (
        kaggle_daily.download_kaggle_file(
            "LeagueSchedule26_27.csv", tmp_path, username="u", key="k"
        )
        is None
    )
    assert all(auth == ("u", "k") for _, auth in calls)
    assert calls[0][0].endswith(f"{config.KAGGLE_DATASET}/PlayerStatistics.csv")
    assert not list(tmp_path.glob("*.download"))


def test_download_requires_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "KAGGLE_USERNAME", None)
    monkeypatch.setattr(config, "KAGGLE_KEY", None)
    with pytest.raises(kaggle_daily.KaggleAuthError):
        kaggle_daily.download_kaggle_file("PlayerStatistics.csv", tmp_path)


def test_download_subset_falls_back_to_previous_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []

    def fake_download(name, dest_dir, username=None, key=None, dataset=None, timeout=None):
        fetched.append(name)
        if name.startswith("LeagueSchedule26_27"):
            return None
        p = dest_dir / name
        p.write_text("x")
        return p

    monkeypatch.setattr(kaggle_daily, "download_kaggle_file", fake_download)
    # October belongs to 2026-27, whose file is not published yet: fall back to 25_26.
    got = kaggle_daily.download_dump_subset(tmp_path, date(2026, 10, 12))
    assert fetched == [
        "PlayerStatistics.csv",
        "TeamHistories.csv",
        "LeagueSchedule26_27.csv",
        "LeagueSchedule25_26.csv",
    ]
    assert got["schedule"] == "LeagueSchedule25_26.csv"


# ---------- classification and merge ----------
def _stored(kaggle_dir: Path, tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    data_dir = tmp_path / "game_logs"
    kaggle_backfill.backfill(kaggle_dir, data_dir)
    return data_dir, local.read_game_logs(data_dir)


def _modified_dump(kaggle_dir: Path, tmp_path: Path, stored: pd.DataFrame) -> tuple[Path, dict]:
    """Copy of the fixture dump with one changed row and one new game after the stored max."""
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / kaggle_dump.TEAM_HISTORY_FILE).write_bytes(
        (kaggle_dir / kaggle_dump.TEAM_HISTORY_FILE).read_bytes()
    )
    box = pd.read_csv(kaggle_dir / kaggle_dump.BOX_SCORE_FILE)
    last = stored.sort_values("game_date").iloc[-1]
    # Change the points of the last stored row for one player.
    mask = (box["personId"] == last["player_id"]) & (
        box["gameId"].astype(str).str.zfill(10) == last["game_id"]
    )
    assert mask.sum() == 1
    box.loc[mask, "points"] = box.loc[mask, "points"] + 5
    # A brand-new regular-season game two days after the stored max, same teams as the last game.
    template_rows = box[box["gameId"].astype(str).str.zfill(10) == last["game_id"]].copy()
    new_gid = 22399999
    new_date = (pd.Timestamp(last["game_date"]) + pd.Timedelta(days=2)).strftime(
        "%Y-%m-%d 19:30:00"
    )
    template_rows["gameId"] = new_gid
    template_rows["gameDate"] = new_date
    box = pd.concat([box, template_rows], ignore_index=True)
    box.to_csv(dump / kaggle_dump.BOX_SCORE_FILE, index=False)
    return dump, {
        "changed_player": int(last["player_id"]),
        "changed_game": last["game_id"],
        "new_game": f"{new_gid:010d}",
        "new_rows": len(template_rows),
    }


def test_run_daily_classifies_and_merges(kaggle_dir: Path, tmp_path: Path) -> None:
    data_dir, stored = _stored(kaggle_dir, tmp_path)
    dump, info = _modified_dump(kaggle_dir, tmp_path, stored)
    report_path = tmp_path / "daily_report.json"

    report = kaggle_daily.run_daily(dump, data_dir, date(2026, 9, 12), report_path, push=False)

    assert report["counts"]["new"] == info["new_rows"]
    assert report["counts"]["changed"] == 1
    assert report["counts"]["unchanged"] > 0
    (example,) = report["changed_examples"]
    assert (
        example["player_id"] == info["changed_player"]
        and example["game_id"] == info["changed_game"]
    )
    assert set(example["fields"]) == {"pts"}
    assert example["fields"]["pts"]["incoming"] == example["fields"]["pts"]["stored"] + 5
    assert report["seasons_written"] == ["2025-26"]
    assert report["pushed"] is False
    assert json.loads(report_path.read_text()) == report

    merged = local.read_game_logs(data_dir)
    schema.validate(merged)
    assert len(merged) == len(stored) + info["new_rows"]
    new_rows = merged[merged["game_id"] == info["new_game"]]
    assert len(new_rows) == info["new_rows"] and (new_rows["source"] == "kaggle_daily").all()
    changed_row = merged[
        (merged["player_id"] == info["changed_player"])
        & (merged["game_id"] == info["changed_game"])
    ].iloc[0]
    old_row = stored[
        (stored["player_id"] == info["changed_player"])
        & (stored["game_id"] == info["changed_game"])
    ].iloc[0]
    assert changed_row["pts"] == old_row["pts"] + 5 and changed_row["source"] == "kaggle_daily"
    # Untouched rows keep their original provenance.
    assert (merged[merged["source"] == config.KAGGLE_SOURCE].shape[0]) == len(stored) - 1


def test_run_daily_no_changes_writes_nothing_and_does_not_push(
    kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, stored = _stored(kaggle_dir, tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in local.list_parquet(data_dir)}
    monkeypatch.setattr(hf, "push_dataset", lambda *a, **k: pytest.fail("must not push"))
    monkeypatch.setattr(hf, "fetch_dataset_card", lambda: pytest.fail("must not fetch card"))
    report = kaggle_daily.run_daily(
        kaggle_dir, data_dir, date(2026, 9, 12), tmp_path / "r.json", push=True
    )
    assert report["counts"]["new"] == 0 and report["counts"]["changed"] == 0
    assert report["counts"]["unchanged"] > 0
    assert report["seasons_written"] == [] and report["pushed"] is False
    assert {p.name: p.stat().st_mtime_ns for p in local.list_parquet(data_dir)} == before


def test_run_daily_pushes_only_affected_seasons_with_card(
    kaggle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_dataset_card import PLACEHOLDER_CARD

    data_dir, stored = _stored(kaggle_dir, tmp_path)
    dump, _ = _modified_dump(kaggle_dir, tmp_path, stored)
    pushed: dict = {}

    def fake_push(data_dir, card_path=None, files=None, message=""):
        pushed.update(files=[p.name for p in files], card=card_path.read_text(), message=message)
        return "newsha"

    monkeypatch.setattr(hf, "push_dataset", fake_push)
    monkeypatch.setattr(hf, "fetch_dataset_card", lambda: PLACEHOLDER_CARD)
    report = kaggle_daily.run_daily(
        dump, data_dir, date(2026, 9, 12), tmp_path / "r.json", push=True, stored_revision="oldsha"
    )
    assert report["pushed"] is True and report["dataset_revision_after"] == "newsha"
    assert report["dataset_revision_before"] == "oldsha"
    assert pushed["files"] == ["game_logs_2025-26.parquet"]
    assert "Daily ingest 2026-09-12" in pushed["message"]
    assert "| 2025-26 |" in pushed["card"] and "[n]" not in pushed["card"]
    # The DNP bullet is left as-is by the daily ingest (no DNP history for old seasons).
    assert "[kept with 0 minutes / dropped" in pushed["card"]


def test_classify_treats_nan_and_float_rounding_as_equal(game_logs: pd.DataFrame) -> None:
    incoming = game_logs.head(5).copy()
    incoming["minutes"] = incoming["minutes"] + 1e-7
    result = kaggle_daily.classify(game_logs, incoming)
    assert result.unchanged == 5 and result.new.empty and result.changed.empty


def test_cli_dry_run_against_local_dump(
    kaggle_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir, _ = _stored(kaggle_dir, tmp_path)
    report = tmp_path / "daily_report.json"
    kaggle_daily.main(
        [
            "--date",
            "2026-09-12",
            "--local-dir",
            str(kaggle_dir),
            "--data-dir",
            str(data_dir),
            "--report",
            str(report),
            "--no-pull",
        ]
    )
    out = capsys.readouterr().out
    assert "DAILY using local dump" in out and "new=0 changed=0" in out
    assert json.loads(report.read_text())["schedule_file"] is None


def test_run_daily_with_no_rows_in_window_reports_zero(kaggle_dir: Path, tmp_path: Path) -> None:
    """Off-season: the dump has nothing newer than the stored max minus 7 days."""
    data_dir, stored = _stored(kaggle_dir, tmp_path)
    # Push the stored max far into the future by adding one late row.
    late = stored.iloc[[-1]].copy()
    late["game_id"] = "0029999999"
    late["game_date"] = pd.Timestamp("2027-01-01")
    late["season"] = "2026-27"
    local.write_per_season(pd.concat([stored, late], ignore_index=True), data_dir)
    report = kaggle_daily.run_daily(kaggle_dir, data_dir, date(2027, 1, 2), tmp_path / "r.json")
    assert report["window_rows_in_dump"] == 0
    assert report["counts"] == {"new": 0, "changed": 0, "unchanged": 0}
    assert report["seasons_written"] == [] and report["dnp_dropped_in_window"] == 0


def test_load_box_scores_empty_window(kaggle_dir: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        kaggle_dump.load_box_scores(kaggle_dir, start=pd.Timestamp("2099-01-01"))
    empty = kaggle_dump.load_box_scores(
        kaggle_dir, start=pd.Timestamp("2099-01-01"), allow_empty=True
    )
    assert empty.empty and "gameDate" in empty.columns and "numMinutes" in empty.columns
