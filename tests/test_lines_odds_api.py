"""The Odds API client: request shape, usage headers, no retries, the key never in an error."""

from __future__ import annotations

import pytest

from nba import config
from nba.lines import odds_api


class FakeResponse:
    def __init__(self, status: int, payload, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


def test_events_and_event_odds_requests_and_usage_headers() -> None:
    headers = {"x-requests-remaining": "424", "x-requests-used": "76", "x-requests-last": "1"}
    session = FakeSession(
        [
            FakeResponse(200, [{"id": "e1", "commence_time": "2026-10-21T00:00:00Z"}], headers),
            FakeResponse(200, {"id": "e1", "bookmakers": []}, headers),
        ]
    )
    api = odds_api.OddsApi(api_key="k", session=session)
    ev = api.events(commence_from="2026-10-20T00:00:00Z", commence_to="2026-10-21T10:00:00Z")
    assert ev.data[0]["id"] == "e1" and ev.usage.remaining == 424 and ev.usage.last == 1
    url, params, timeout = session.calls[0]
    assert url == f"{config.ODDS_API_BASE}/sports/basketball_nba/events"
    assert params["apiKey"] == "k" and params["commenceTimeFrom"] == "2026-10-20T00:00:00Z"
    assert timeout == odds_api.TIMEOUT

    odds = api.event_odds("e1")
    url, params, _ = session.calls[1]
    assert url.endswith("/events/e1/odds")
    assert params["markets"] == "player_points" and params["bookmakers"] == "draftkings"
    assert params["oddsFormat"] == "american" and odds.usage.used == 76


def test_errors_carry_the_status_and_never_the_key() -> None:
    session = FakeSession([FakeResponse(429, None, {}, text="quota exceeded for key k")])
    api = odds_api.OddsApi(api_key="k", session=session)
    with pytest.raises(odds_api.OddsApiError, match="HTTP 429") as info:
        api.events()
    assert "***" in str(info.value) and " k" not in str(info.value)
    assert len(session.calls) == 1, "no retry"


def test_missing_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ODDS_API_KEY", None)
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        odds_api.OddsApi()


def test_usage_parses_missing_or_odd_headers() -> None:
    u = odds_api.Usage.from_headers({"x-requests-remaining": "12.0", "x-requests-last": "n/a"})
    assert u.remaining == 12 and u.used is None and u.last is None
