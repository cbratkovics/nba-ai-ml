import json
from datetime import date
from pathlib import Path

import pytest

from nba import config
from nba.agent import evals, loop, tools
from nba.storage import local


def _ctx(tmp_path: Path, game_logs) -> tools.ToolContext:
    root = tmp_path / "root"
    local.write_per_season(game_logs, root / config.DATA_DIR)
    return tools.ToolContext(root=root, data_dir=root / config.DATA_DIR, run_date=date(2026, 1, 15))


def _tool_call(cid: str, name: str, args: dict) -> dict:
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptedChat:
    """Returns a fixed sequence of assistant messages."""

    model_id = "scripted-model"

    def __init__(self, messages):
        self._messages = list(messages)
        self.calls = 0

    def complete(self, messages, tool_schemas):
        self.calls += 1
        if not self._messages:
            raise RuntimeError("script exhausted")
        return self._messages.pop(0), {"total_tokens": 10}


def _final(findings) -> dict:
    return {
        "role": "assistant",
        "content": json.dumps({"summary": "Short summary.", "findings": findings}),
    }


def test_happy_path_runs_tools_and_parses_brief(tmp_path: Path, game_logs) -> None:
    ctx = _ctx(tmp_path, game_logs)
    chat = ScriptedChat(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("c1", "get_upstream_freshness", {}),
                    _tool_call("c2", "list_data_gaps", {}),
                ],
            },
            _final(
                [
                    {
                        "kind": "data_gap",
                        "severity": "info",
                        "evidence": {
                            "tool": "list_data_gaps",
                            "args": {},
                            "values": {"known_missing_games": 7},
                        },
                        "text": "There are 7 known missing games.",
                    }
                ]
            ),
        ]
    )
    brief, trace = loop.run_agent(ctx, date(2026, 1, 14), chat)
    assert brief["status"] == "ok" and brief["tool_calls_made"] == 7  # 5 prefetched + 2
    assert [r["name"] for r in trace.prefetch] == [
        n for n, _ in loop.standard_calls(ctx, date(2026, 1, 14))
    ]
    assert brief["date"] == "2026-01-14" and brief["run_date"] == "2026-01-15"
    assert brief["model_id"] == "scripted-model" and brief["latency_ms"] >= 0
    assert brief["findings"][0]["kind"] == "data_gap"
    assert len(trace.steps) == 2
    assert [r["name"] for r in trace.steps[0].tool_results] == [
        "get_upstream_freshness",
        "list_data_gaps",
    ]
    assert trace.steps[0].tool_results[1]["result"]["full_season_games"] == config.FULL_SEASON_GAMES
    # Prefetched and model-requested tool results were fed back as tool messages.
    roles = [m["role"] for m in trace.steps[1].request_messages]
    assert roles == ["system", "user", "assistant"] + ["tool"] * 5 + ["assistant", "tool", "tool"]


def test_tool_budget_and_turn_limits_end_in_unavailable(tmp_path: Path, game_logs) -> None:
    ctx = _ctx(tmp_path, game_logs)
    # Five calls are prefetched; four more in one turn means the fourth is refused.
    many = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call(f"c{i}", "list_data_gaps", {}) for i in range(4)],
    }
    chat = ScriptedChat(
        [
            many,
            {"role": "assistant", "content": "not json"},
            {"role": "assistant", "content": "still not"},
            {"role": "assistant", "content": "nope"},
        ]
    )
    brief, trace = loop.run_agent(ctx, date(2026, 1, 14), chat)
    assert brief["status"] == "agent_unavailable"
    assert brief["tool_calls_made"] == 8
    assert "tool budget of 8 calls exhausted" in trace.steps[0].tool_results[3]["result"]["error"]
    assert "no valid JSON brief after 3 turns" in brief["error"]
    assert brief["findings"] == []


def test_groq_error_never_raises(tmp_path: Path, game_logs) -> None:
    ctx = _ctx(tmp_path, game_logs)

    class Broken:
        model_id = "broken"

        def complete(self, messages, tool_schemas):
            raise ConnectionError("groq down")

    brief, trace = loop.run_agent(ctx, date(2026, 1, 14), Broken())
    assert brief["status"] == "agent_unavailable" and "groq down" in brief["error"]
    assert trace.status == "agent_unavailable"


def test_wall_clock_limit(tmp_path: Path, game_logs, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, game_logs)
    monkeypatch.setattr(loop, "WALL_CLOCK_SECONDS", 0.0)
    brief, _ = loop.run_agent(ctx, date(2026, 1, 14), ScriptedChat([_final([])]))
    assert brief["status"] == "agent_unavailable" and "wall clock" in brief["error"]


def test_parse_brief_accepts_fenced_and_embedded_json() -> None:
    body = {"summary": "s", "findings": []}
    assert loop._parse_brief("```json\n" + json.dumps(body) + "\n```") == body
    assert loop._parse_brief("Here you go: " + json.dumps(body) + " thanks") == body
    assert loop._parse_brief("no json here") is None
    assert loop._parse_brief(json.dumps({"summary": "missing findings"})) is None


def test_findings_are_normalized_and_capped() -> None:
    raw = [{"kind": "x" * 80, "severity": "bogus", "evidence": "not a dict", "text": " t "}] * 7
    out = loop._normalize_findings(raw)
    assert len(out) == 5 and out[0]["severity"] == "info" and len(out[0]["kind"]) == 40
    assert out[0]["evidence"] == {"tool": "", "args": {}, "values": {}} and out[0]["text"] == "t"


def test_write_outputs_scrubs_keys_and_maintains_index(tmp_path: Path, game_logs) -> None:
    ctx = _ctx(tmp_path, game_logs)
    chat = ScriptedChat(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "list_data_gaps", {})],
            },
            _final([]),
        ]
    )
    brief, trace = loop.run_agent(ctx, date(2026, 1, 14), chat)
    trace.steps[0].response["content"] = "leaked gsk_abcDEF123456 token"
    brief_dir = tmp_path / "brief"
    written = loop.write_outputs(brief, trace, brief_dir)
    assert [p.name for p in written] == [
        "2026-01-14.json",
        "2026-01-14.trace.json",
        "index.json",
        "latest.json",
    ]
    trace_text = (brief_dir / "2026-01-14.trace.json").read_text()
    assert "gsk_abcDEF123456" not in trace_text and "gsk_REDACTED" in trace_text
    assert len(trace_text.encode()) <= loop.MAX_TRACE_BYTES
    # An earlier date does not replace latest.json; the index lists both.
    older = dict(brief, date="2026-01-10")
    written2 = loop.write_outputs(older, trace, brief_dir)
    assert [p.name for p in written2] == ["2026-01-10.json", "2026-01-10.trace.json", "index.json"]
    index = json.loads((brief_dir / "index.json").read_text())
    assert index == {"dates": ["2026-01-10", "2026-01-14"], "latest": "2026-01-14"}
    assert json.loads((brief_dir / "latest.json").read_text())["date"] == "2026-01-14"


def test_replay_chat_reproduces_brief_from_trace(tmp_path: Path, game_logs) -> None:
    ctx = _ctx(tmp_path, game_logs)
    chat = ScriptedChat(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "list_data_gaps", {})],
            },
            _final(
                [
                    {
                        "kind": "data_gap",
                        "severity": "info",
                        "evidence": {"tool": "list_data_gaps", "args": {}, "values": {"n": 7}},
                        "text": "7 games missing.",
                    }
                ]
            ),
        ]
    )
    brief, trace = loop.run_agent(ctx, date(2026, 1, 14), chat)
    replay = loop.ReplayChat(trace.to_dict())
    again, trace2 = loop.run_agent(
        ctx, date(2026, 1, 14), replay, run_tool=replay.recorded_tool_runner()
    )
    for key in ("status", "summary", "findings", "tool_calls_made"):
        assert again[key] == brief[key]
    assert trace2.steps[0].tool_results[0]["result"] == trace.steps[0].tool_results[0]["result"]
    assert trace2.prefetch == trace.prefetch


def test_run_and_write_applies_grounding_and_never_raises(
    tmp_path: Path, game_logs, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, game_logs)
    ungrounded = {
        "kind": "x",
        "severity": "warning",
        "evidence": {"tool": "t", "args": {}, "values": {"a": 1}},
        "text": "MAE was 9.99.",
    }
    grounded = {
        "kind": "y",
        "severity": "info",
        "evidence": {"tool": "t", "args": {}, "values": {"a": 1}},
        "text": "value 1.",
    }
    chat = ScriptedChat([_final([ungrounded, grounded])])
    brief, written = loop.run_and_write(ctx, date(2026, 1, 14), tmp_path / "brief", chat=chat)
    assert brief["status"] == evals.STATUS_UNGROUNDED
    assert [f["kind"] for f in brief["findings"]] == ["y"]
    assert brief["dropped_findings"][0]["ungrounded_numbers"] == [9.99]
    assert (tmp_path / "brief" / "latest.json").exists()
    # No GROQ key and no chat: still writes an agent_unavailable brief.
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    brief, _ = loop.run_and_write(ctx, date(2026, 1, 13), tmp_path / "brief2")
    assert brief["status"] == "agent_unavailable" and "GROQ_API_KEY" in brief["error"]


def test_recover_brief_from_pseudo_tool_call() -> None:
    body = {
        "summary": "s",
        "findings": [{"kind": "k", "severity": "info", "evidence": {}, "text": "t"}],
    }

    class Exc(Exception):
        def __init__(self, body):
            self.body = body

    exc = Exc(
        {
            "error": {
                "code": "tool_use_failed",
                "failed_generation": json.dumps({"name": "json", "arguments": body}),
            }
        }
    )
    assert json.loads(loop.recover_brief_from_tool_error(exc)) == body
    # String-encoded arguments are accepted too; anything else is left to raise.
    exc.body["error"]["failed_generation"] = json.dumps(
        {"name": "json", "arguments": json.dumps(body)}
    )
    assert json.loads(loop.recover_brief_from_tool_error(exc)) == body
    assert loop.recover_brief_from_tool_error(Exc({"error": {"code": "other"}})) is None
    assert loop.recover_brief_from_tool_error(Exc(None)) is None
    exc.body["error"]["failed_generation"] = json.dumps({"name": "get_residuals", "arguments": {}})
    assert loop.recover_brief_from_tool_error(exc) is None
