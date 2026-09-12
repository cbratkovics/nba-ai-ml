"""Bounded tool-calling loop on Groq that writes a structured JSON brief.

Hard limits: at most 8 tool calls, at most 3 model turns after the last tool call,
60 s wall clock, temperature 0. The model may only report what tools returned and must
cite the tool call behind every number; the deterministic evals in `nba.agent.evals`
check that afterwards.

Any Groq error or limit breach produces a brief with status "agent_unavailable" and
exits 0: the agent must never fail the nightly job. The full request/response/tool
trace is saved next to the brief as brief/<date>.trace.json (no headers or keys are
ever recorded, and the writer refuses to save a trace containing a Groq key).

Usage:
    python -m nba.agent.loop --date YYYY-MM-DD [--root .] [--data-dir data/game_logs]
                             [--dump-dir data/dump] [--push] [--no-pull] [--trace-replay PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nba import config
from nba.agent import tools
from nba.storage import hf

MAX_TOOL_CALLS = 8
MAX_TURNS_AFTER_TOOLS = 3
WALL_CLOCK_SECONDS = 60.0
TEMPERATURE = 0
MAX_FINDINGS = 5
MAX_OUTPUT_TOKENS = 2000
MAX_TRACE_BYTES = 100_000
KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]+")
SEVERITIES = ("info", "warning", "critical")
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "agent_unavailable"

SYSTEM_PROMPT = (
    "You are the nightly analyst for an NBA player-stat prediction pipeline. You have "
    "read-only tools over the pipeline's published files. Rules:\n"
    "1. Report only what the tools returned. Every number in a finding's text must come "
    "from that finding's evidence.values, copied exactly.\n"
    "2. Cite the tool call (tool name and arguments) for every number you report.\n"
    '3. If a tool returns available=false or an error, say "no evidence" for that topic '
    "instead of guessing.\n"
    "4. Never speculate about injuries, lineups, trades, rest, or any reason a player did "
    "or did not play. Report only counts and values.\n"
    "5. Flag at most 5 findings, most important first. Severity is one of info, warning, "
    "critical.\n"
    "6. Be brief: the summary is at most 120 words.\n"
    "7. The standard evidence (ingest report, freshness, residuals, rolling metrics, data "
    "gaps) has already been fetched for you and appears as tool results. You may call "
    "get_player_recent or get_team_context a few more times if a residual needs context; "
    "otherwise answer immediately. evidence.values is a small flat object holding only "
    "the keys and numbers the text cites, including window lengths (days_requested) and "
    "counts; never copy whole lists or nested tool output into it. Round numbers to two "
    "decimals. A residual finding must name the player and give predicted and actual "
    "values.\n"
    "8. The brief is your final message content, plain JSON with no code fence. Never "
    "wrap it in a tool call; there is no tool named json.\n\n"
    "When you are done, respond with ONLY a JSON object (no markdown, no prose) with "
    "exactly these keys:\n"
    '{"summary": string, "findings": [{"kind": string, "severity": '
    '"info"|"warning"|"critical", "evidence": {"tool": string, "args": object, '
    '"values": object}, "text": string}]}\n'
    '"kind" is a short label such as ingest, freshness, residual_outlier, '
    "rolling_accuracy, data_gap.\n"
    '"evidence.values" must contain every number used in "text", as returned by the tool.'
)


@dataclass
class Step:
    request_messages: list[dict[str, Any]]
    response: dict[str, Any]
    usage: dict[str, Any] | None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0


@dataclass
class Trace:
    date: str
    run_date: str
    model_id: str
    started_at: str
    limits: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    prefetch: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0
    status: str = STATUS_OK
    error: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "run_date": self.run_date,
            "model_id": self.model_id,
            "started_at": self.started_at,
            "limits": self.limits,
            "status": self.status,
            "error": self.error,
            "tool_calls_made": self.tool_calls_made,
            "latency_ms": self.latency_ms,
            "prefetch": self.prefetch,
            "steps": [
                {
                    "request_messages": s.request_messages,
                    "response": s.response,
                    "usage": s.usage,
                    "tool_results": s.tool_results,
                    "latency_ms": s.latency_ms,
                }
                for s in self.steps
            ],
        }


class AgentLimit(RuntimeError):
    """A hard limit was reached before the model produced a brief."""


def standard_calls(ctx: tools.ToolContext, d: date) -> list[tuple[str, dict[str, Any]]]:
    """The five tool calls every brief needs, fetched before the model's first turn.

    Doing this deterministically keeps the conversation short (one round trip instead
    of five) and inside Groq's free-tier tokens-per-minute budget. The calls are recorded
    in the trace and counted against the tool budget like any model-initiated call.
    """
    return [
        ("get_daily_report", {"date": ctx.run_date.isoformat()}),
        ("get_upstream_freshness", {}),
        ("get_residuals", {"date": d.isoformat()}),
        ("get_rolling_metrics", {"days": 30}),
        ("list_data_gaps", {}),
    ]


# ---------- chat backends ----------


class GroqChat:
    """Thin wrapper over the Groq client returning plain dicts."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = config.GROQ_MODEL,
        reasoning_effort: str | None = config.GROQ_REASONING_EFFORT,
        fallback_model_id: str | None = config.GROQ_MODEL_FALLBACK,
    ):
        from groq import Groq

        key = api_key or config.groq_api_key()
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=key)
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.fallback_model_id = fallback_model_id
        self.fell_back = False

    def complete(
        self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        from groq import BadRequestError, RateLimitError

        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}

        def create():
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                **extra,
            )

        try:
            try:
                r = create()
            except RateLimitError:
                # Free-tier quotas are per model per day. Switch to the fallback for the
                # rest of this brief; the brief records the model that answered.
                if not self.fallback_model_id or self.fallback_model_id == self.model_id:
                    raise
                self.model_id = self.fallback_model_id
                self.fell_back = True
                r = create()
        except BadRequestError as exc:
            recovered = recover_brief_from_tool_error(exc)
            if recovered is None:
                raise
            return {"role": "assistant", "content": recovered}, None
        m = r.choices[0].message
        message: dict[str, Any] = {"role": "assistant", "content": m.content}
        if m.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in m.tool_calls
            ]
        usage = r.usage.model_dump() if r.usage else None
        return message, usage


def recover_brief_from_tool_error(exc: Exception) -> str | None:
    """The brief when the model wrapped its final answer in a pseudo tool call.

    gpt-oss models sometimes emit the JSON brief as a call to a tool named "json" (or
    the name of the schema) instead of as message content. Groq rejects that with a 400
    tool_use_failed error whose body carries the generated text. If that text is a
    call whose arguments hold a brief, hand it back as content so the loop can parse it
    like any other answer.
    """
    body = getattr(exc, "body", None)
    err = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(err, dict) or err.get("code") != "tool_use_failed":
        return None
    raw = err.get("failed_generation")
    if not isinstance(raw, str):
        return None
    try:
        gen = json.loads(raw)
    except json.JSONDecodeError:
        return None
    args = gen.get("arguments") if isinstance(gen, dict) else None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if isinstance(args, dict) and "summary" in args and "findings" in args:
        return json.dumps(args)
    return None


class ReplayChat:
    """Replays the assistant messages recorded in a trace; used by the offline evals."""

    def __init__(self, trace: dict[str, Any]):
        self.model_id = trace["model_id"]
        self._responses = [s["response"] for s in trace["steps"]]
        self._results = [trace.get("prefetch", [])] + [s["tool_results"] for s in trace["steps"]]
        self._i = 0

    def complete(self, messages, tool_schemas):
        if self._i >= len(self._responses):
            raise AgentLimit("trace has no more recorded responses")
        message = self._responses[self._i]
        self._i += 1
        return message, None

    def recorded_tool_runner(self) -> Callable[[tools.ToolContext, str, dict | None], dict]:
        """A run_tool replacement that returns the recorded result for each call, in order."""
        queue = [r for step in self._results for r in step]

        def run(ctx, name, args):
            if not queue:
                return {"error": "no recorded result"}
            rec = queue.pop(0)
            if rec["name"] != name:
                return {"error": f"recorded call was {rec['name']}, not {name}"}
            return rec["result"]

        return run


# ---------- the loop ----------


def _parse_brief(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) and "findings" in data else None


def _normalize_findings(raw: Any) -> list[dict[str, Any]]:
    findings = []
    for f in raw if isinstance(raw, list) else []:
        if not isinstance(f, dict):
            continue
        ev = f.get("evidence") if isinstance(f.get("evidence"), dict) else {}
        findings.append(
            {
                "kind": str(f.get("kind", "other"))[:40],
                "severity": f.get("severity") if f.get("severity") in SEVERITIES else "info",
                "evidence": {
                    "tool": str(ev.get("tool", "")),
                    "args": ev.get("args") if isinstance(ev.get("args"), dict) else {},
                    "values": ev.get("values") if isinstance(ev.get("values"), dict | list) else {},
                },
                "text": str(f.get("text", "")).strip(),
            }
        )
    return findings[:MAX_FINDINGS]


def unavailable_brief(
    d: date, run_date: date, model_id: str, error: str, tool_calls: int, latency_ms: int
) -> dict[str, Any]:
    return {
        "date": d.isoformat(),
        "run_date": run_date.isoformat(),
        "status": STATUS_UNAVAILABLE,
        "summary": f"The analyst agent did not produce a brief: {error}",
        "findings": [],
        "tool_calls_made": tool_calls,
        "model_id": model_id,
        "latency_ms": latency_ms,
        "error": error,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_agent(
    ctx: tools.ToolContext,
    d: date,
    chat: Any,
    run_tool: Callable[[tools.ToolContext, str, dict | None], dict] = tools.run_tool,
    prefetch: bool = True,
) -> tuple[dict[str, Any], Trace]:
    """Run the bounded loop for brief date `d`. Never raises for model or limit errors."""
    model_id = getattr(chat, "model_id", config.GROQ_MODEL)
    started = time.perf_counter()
    trace = Trace(
        date=d.isoformat(),
        run_date=ctx.run_date.isoformat(),
        model_id=model_id,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        limits={
            "max_tool_calls": MAX_TOOL_CALLS,
            "max_turns_after_tools": MAX_TURNS_AFTER_TOOLS,
            "wall_clock_seconds": WALL_CLOCK_SECONDS,
            "temperature": TEMPERATURE,
        },
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Run date: {ctx.run_date.isoformat()}. Brief date: {d.isoformat()}. "
                f"Write the analyst brief for {d.isoformat()}: ingest status for the run date, "
                f"upstream freshness, residual outliers for {d.isoformat()}, rolling accuracy "
                "versus the last-10 baseline, and known data gaps. Use the tools, then answer "
                "with the JSON object only."
            ),
        },
    ]
    turns_after_tools = 0
    tools_exhausted = False
    try:
        if prefetch:
            calls = []
            for i, (name, args) in enumerate(standard_calls(ctx, d)):
                cid = f"pre-{i + 1}"
                result = run_tool(ctx, name, args)
                trace.tool_calls_made += 1
                trace.prefetch.append({"id": cid, "name": name, "args": args, "result": result})
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                )
            messages.append({"role": "assistant", "content": None, "tool_calls": calls})
            for rec in trace.prefetch:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": rec["id"],
                        "name": rec["name"],
                        "content": json.dumps(rec["result"]),
                    }
                )
        while True:
            if time.perf_counter() - started > WALL_CLOCK_SECONDS:
                raise AgentLimit(f"wall clock exceeded {WALL_CLOCK_SECONDS:.0f}s")
            t0 = time.perf_counter()
            message, usage = chat.complete(messages, tools.TOOL_SCHEMAS)
            step = Step(
                request_messages=[dict(m) for m in messages],
                response=message,
                usage=usage,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            trace.steps.append(step)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if calls and not tools_exhausted:
                turns_after_tools = 0
                for call in calls:
                    if trace.tool_calls_made >= MAX_TOOL_CALLS:
                        tools_exhausted = True
                        result = {"error": f"tool budget of {MAX_TOOL_CALLS} calls exhausted"}
                        args = {}
                    else:
                        try:
                            args = json.loads(call["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        result = run_tool(ctx, call["function"]["name"], args)
                        trace.tool_calls_made += 1
                    step.tool_results.append(
                        {
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "args": args,
                            "result": result,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": call["function"]["name"],
                            "content": json.dumps(result),
                        }
                    )
                if tools_exhausted:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The tool budget is exhausted. Answer now with the JSON "
                                "object only."
                            ),
                        }
                    )
                continue
            if calls and tools_exhausted:
                for call in calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": call["function"]["name"],
                            "content": json.dumps({"error": "tool budget exhausted"}),
                        }
                    )
            turns_after_tools += 1
            brief_body = _parse_brief(message.get("content"))
            if brief_body is not None:
                break
            if turns_after_tools >= MAX_TURNS_AFTER_TOOLS:
                raise AgentLimit(f"no valid JSON brief after {MAX_TURNS_AFTER_TOOLS} turns")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That was not a valid JSON object. Respond with ONLY the JSON object "
                        "described in the instructions."
                    ),
                }
            )
    except AgentLimit as exc:
        trace.status, trace.error = STATUS_UNAVAILABLE, str(exc)
    except Exception as exc:  # noqa: BLE001 - Groq/network errors must not propagate
        trace.status, trace.error = STATUS_UNAVAILABLE, f"{type(exc).__name__}: {exc}"

    trace.latency_ms = int((time.perf_counter() - started) * 1000)
    if trace.status != STATUS_OK:
        return unavailable_brief(
            d,
            ctx.run_date,
            model_id,
            trace.error or "unknown",
            trace.tool_calls_made,
            trace.latency_ms,
        ), trace

    summary = str(brief_body.get("summary", "")).strip()
    words = summary.split()
    if len(words) > 120:
        summary = " ".join(words[:120])
    brief = {
        "date": d.isoformat(),
        "run_date": ctx.run_date.isoformat(),
        "status": STATUS_OK,
        "summary": summary,
        "findings": _normalize_findings(brief_body.get("findings")),
        "tool_calls_made": trace.tool_calls_made,
        "model_id": model_id,
        "latency_ms": trace.latency_ms,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return brief, trace


# ---------- outputs ----------


def scrub(text: str) -> str:
    return KEY_PATTERN.sub("gsk_REDACTED", text)


def write_outputs(
    brief: dict[str, Any], trace: Trace | dict[str, Any], brief_dir: Path
) -> list[Path]:
    """Write brief/<date>.json, brief/<date>.trace.json, brief/index.json, brief/latest.json."""
    brief_dir.mkdir(parents=True, exist_ok=True)
    d = brief["date"]
    written = []
    brief_path = brief_dir / f"{d}.json"
    brief_path.write_text(json.dumps(brief, indent=2) + "\n")
    written.append(brief_path)

    trace_dict = trace.to_dict() if isinstance(trace, Trace) else trace
    text = json.dumps(trace_dict, indent=1)
    if KEY_PATTERN.search(text):
        text = scrub(text)
    if len(text.encode()) > MAX_TRACE_BYTES:
        # Keep the trace readable but bounded: drop request snapshots first.
        for step in trace_dict["steps"]:
            step["request_messages"] = (
                f"<{len(step['request_messages'])} messages omitted for size>"
            )
        text = json.dumps(trace_dict, indent=1)
    trace_path = brief_dir / f"{d}.trace.json"
    trace_path.write_text(text + "\n")
    written.append(trace_path)

    dates = sorted(
        {p.stem for p in brief_dir.glob("*.json") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem)}
    )
    index = {"dates": dates, "latest": dates[-1] if dates else None}
    index_path = brief_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    written.append(index_path)
    if index["latest"] == d:
        latest_path = brief_dir / "latest.json"
        latest_path.write_text(json.dumps(brief, indent=2) + "\n")
        written.append(latest_path)
    return written


def run_and_write(
    ctx: tools.ToolContext,
    d: date,
    brief_dir: Path,
    chat: Any | None = None,
    run_tool: Callable[[tools.ToolContext, str, dict | None], dict] = tools.run_tool,
) -> tuple[dict[str, Any], list[Path]]:
    """Build the chat backend if needed, run the loop, write outputs. Never raises."""
    try:
        chat = chat or GroqChat()
    except Exception as exc:  # noqa: BLE001
        brief = unavailable_brief(
            d, ctx.run_date, config.GROQ_MODEL, f"{type(exc).__name__}: {exc}", 0, 0
        )
        trace = Trace(
            d.isoformat(),
            ctx.run_date.isoformat(),
            config.GROQ_MODEL,
            datetime.now(UTC).isoformat(timespec="seconds"),
            {},
            status=STATUS_UNAVAILABLE,
            error=brief["error"],
        )
        return brief, write_outputs(brief, trace, brief_dir)
    brief, trace = run_agent(ctx, d, chat, run_tool=run_tool)
    from nba.agent import evals

    brief = evals.apply_grounding(brief)
    return brief, write_outputs(brief, trace, brief_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="brief date")
    parser.add_argument(
        "--run-date", type=date.fromisoformat, default=None, help="default: today UTC"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument("--push", action="store_true", help="push brief/ to the HF dataset repo")
    parser.add_argument("--no-pull", action="store_true", help="use local dirs as-is")
    parser.add_argument(
        "--trace-replay",
        type=Path,
        default=None,
        help="replay a recorded trace instead of calling Groq",
    )
    parser.add_argument("--model", default=None, help="override the pinned model id")
    args = parser.parse_args(argv)
    run_date = args.run_date or datetime.now(UTC).date()

    if not args.no_pull:
        hf.pull_dataset(args.data_dir)
        hf.pull_products(args.root)
    ctx = tools.ToolContext(
        root=args.root, data_dir=args.data_dir, run_date=run_date, dump_dir=args.dump_dir
    )
    chat = GroqChat(model_id=args.model) if args.model else None
    run_tool = tools.run_tool
    if args.trace_replay:
        replay = ReplayChat(json.loads(args.trace_replay.read_text()))
        chat, run_tool = replay, replay.recorded_tool_runner()
    brief, written = run_and_write(
        ctx, args.date, args.root / config.BRIEF_DIR, chat=chat, run_tool=run_tool
    )
    print(
        f"AGENT {args.date}: status={brief['status']} tool_calls={brief['tool_calls_made']} "
        f"findings={len(brief['findings'])} model={brief['model_id']} "
        f"latency_ms={brief['latency_ms']}"
    )
    if brief.get("error"):
        print(f"AGENT {args.date}: {brief['error']}")
    if args.push:
        sha = hf.push_products(args.root, files=written, message=f"Agent brief {args.date}")
        print(f"AGENT {args.date}: pushed {len(written)} files at {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
