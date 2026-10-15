"""Turn an event log into a trace.

The log is already a complete record of a run, so a trace is *derived* from it by a pure function, not recorded
separately. That is why this works for runs started in other processes, for runs that happened last month, and for
replays: `spans_from_events(events)` returns the same spans for the same events, every time.

Spans (names follow the OpenTelemetry GenAI vocabulary, translated by the exporter's dialect):

* every run is an **agent** span (`invoke_agent <algo>`); child runs are in the same trace, parented to the tool
  call (`run_tag` = the call id) or session turn (`turn:N`) that started them, else to the parent run;
* every model call (`aaa.llm.requested` + its result) is an **llm** span with model, token usage, cost, finish
  reason, and how the request was sent (provider state, context compaction);
* every tool call is a **tool** span (attempts, delivery, approval waits and fencing as span events);
* memory calls are **memory** spans, context/session summaries and session turns have their own spans;
* guardrail verdicts and eval scores (when those features are used) become **guardrail** / **evaluator** spans.

Ids are deterministic hashes of event ids (never Python's `hash()`), so exporting the same run twice produces the
same ids and a backend can deduplicate. Timestamps are *event times*: an LLM span lasts from the request event to its
result event, which also covers queueing time and clock differences between processes. This is documented, not hidden."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from aaa.core.parts import is_media_dict
from aaa.events import Event

SpanKind = Literal[
    "agent", "llm", "tool", "memory", "retriever", "guardrail", "evaluator", "session", "internal"
]
Status = Literal["ok", "error", "incomplete"]


def _hex(prefix: str, key: str, n: int) -> str:
    return hashlib.sha256(f"{prefix}:{key}".encode()).hexdigest()[:n]


def trace_id_of(root_run_id: str) -> str:
    """The 32-hex-digit trace id for a trace whose root run is `root_run_id`."""
    return _hex("trace", root_run_id, 32)


def span_id_of(kind: str, key: str) -> str:
    """The 16-hex-digit span id for the span of `kind` identified by `key`."""
    return _hex(f"span-{kind}", key, 16)


@dataclass
class SpanEvent:
    name: str
    time: datetime
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    start: datetime
    end: datetime | None  # None: still open
    status: Status = "ok"
    status_message: str | None = None
    # Neutral attribute names (see `A`); a dialect translates them. Never contains content unless captured.
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    run_id: str = ""
    tenant: str | None = None
    user: str | None = None
    # message/tool content, present only when the caller asked to capture it
    input: Any = None
    output: Any = None

    @property
    def closed(self) -> bool:
        return self.end is not None

    @property
    def duration_s(self) -> float | None:
        return None if self.end is None else (self.end - self.start).total_seconds()


class A:
    """The neutral attribute names a `Span` uses. Dialects (`aaa.observe`) map these to gen_ai.*, OpenInference, etc."""

    OPERATION = "operation"  # chat | execute_tool | invoke_agent | memory | summarize | guard | evaluate | turn
    MODEL = "model"
    FINISH_REASON = "finish_reason"
    INPUT_TOKENS = "usage.input_tokens"
    OUTPUT_TOKENS = "usage.output_tokens"
    CACHED_TOKENS = "usage.cached_tokens"
    TOTAL_TOKENS = "usage.total_tokens"
    COST = "cost"
    LLM_STATE = "llm.state"  # fresh | continued | fallback (provider-held conversation)
    CONTEXT_BEFORE = "context.tokens_before"
    CONTEXT_AFTER = "context.tokens_after"
    CONTEXT_APPLIED = "context.applied"
    TOOL_NAME = "tool.name"
    TOOL_CALL_ID = "tool.call_id"
    TOOL_ATTEMPTS = "tool.attempts"
    TOOL_DELIVERY = "tool.delivery"
    AGENT_NAME = "agent.name"
    OUTCOME = "outcome.kind"
    SESSION_ID = "session.id"
    TURN = "session.turn"
    MEMORY_OP = "memory.op"
    MEMORY_ITEMS = "memory.items"
    GUARD_NAME = "guard.name"
    GUARD_STAGE = "guard.stage"
    GUARD_ACTION = "guard.action"
    SCORE_NAME = "score.name"
    SCORE_VALUE = "score.value"
    ERROR_TYPE = "error.type"
    PARTIAL = "aaa.partial"  # the span was exported before its run finished


def _usage(d: Mapping[str, Any]) -> dict[str, Any]:
    u = d.get("usage") or {}
    out: dict[str, Any] = {}
    if "prompt_tokens" in u:
        out[A.INPUT_TOKENS] = u["prompt_tokens"]
    if "completion_tokens" in u:
        out[A.OUTPUT_TOKENS] = u["completion_tokens"]
    if u.get("cached_tokens"):
        out[A.CACHED_TOKENS] = u["cached_tokens"]
    if "total_tokens" in u:
        out[A.TOTAL_TOKENS] = u["total_tokens"]
    if d.get("cost") is not None:
        out[A.COST] = d["cost"]
    return out


def _strip_media(v: Any) -> Any:
    """Attachments never go into a trace: only their kind, type and size (bytes, base64 and blob references are dropped)."""
    if is_media_dict(v):
        return {
            "type": v["type"],
            "media_type": v["media_type"],
            "size": v.get("size") or (v.get("blob") or {}).get("size"),
        }
    if isinstance(v, dict):
        return {k: _strip_media(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_strip_media(x) for x in v]
    return v


def _clip(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        value = _strip_media(value)
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return (
        text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more characters]"
    )


def spans_from_events(
    events: Sequence[Event],
    *,
    parent_events: Sequence[Event] | None = None,
    capture_content: bool = False,
    content_limit: int = 4000,
) -> list[Span]:
    """The spans of ONE run, from that run's events (in log order).

    `parent_events` (the parent run's events) lets a child run's root span be parented to the exact tool call or session
    turn that started it; without it the child hangs under the parent's run span. Spans still open (no end event) have
    `end=None`. With `capture_content`, prompts, replies, tool arguments and results are attached (clipped);
    events are already redacted if the engine has a redactor, but content leaves your process, so it is opt-in."""
    started = next((e for e in events if e.type == "aaa.run.started"), None)
    if started is None:
        return []
    run_id = started.run_id or started.stream
    completed = next((e for e in reversed(events) if e.type == "aaa.run.completed"), None)
    parent_run = started.data.get("parent")
    tag = started.data.get("run_tag")
    user = started.meta.get("user")
    root_trace = started.trace_id or run_id
    tid = trace_id_of(root_trace)
    run_sid = span_id_of("run", run_id)

    parent_sid: str | None = None
    if parent_run:
        parent_sid = span_id_of("run", parent_run)
        if tag and parent_events is not None:
            if any(
                e.type == "aaa.tool.requested" and e.data.get("call_id") == tag
                for e in parent_events
            ):
                parent_sid = span_id_of("tool", f"{parent_run}:{tag}")
            elif str(tag).startswith("turn:"):
                parent_sid = span_id_of("turn", f"{parent_run}:{tag}")

    def make(
        kind: SpanKind,
        sid: str,
        parent: str | None,
        name: str,
        start: datetime,
        end: datetime | None,
    ) -> Span:
        return Span(
            trace_id=tid,
            span_id=sid,
            parent_id=parent,
            name=name,
            kind=kind,
            start=start,
            end=end,
            run_id=run_id,
            tenant=started.tenant,
            user=str(user) if user is not None else None,
        )

    spans: list[Span] = []
    algo = str(started.data.get("algo", "agent"))
    root = make(
        "agent",
        run_sid,
        parent_sid,
        f"invoke_agent {algo}",
        started.time,
        completed.time if completed else None,
    )
    root.attributes |= {A.OPERATION: "invoke_agent", A.AGENT_NAME: algo}
    if started.meta.get("session") or algo.startswith("aaa.session"):
        root.attributes[A.SESSION_ID] = run_id
    if completed is not None:
        kind = completed.data.get("kind")
        root.attributes[A.OUTCOME] = kind
        if kind != "success":
            root.status, root.status_message = "error", str(completed.data.get("reason") or kind)
    else:
        root.status = "incomplete"
    if capture_content:
        root.input = _clip(started.data.get("input"), content_limit)
        if completed is not None:
            root.output = _clip(completed.data.get("value"), content_limit)
    spans.append(root)

    by_cause: dict[str, list[Event]] = {}
    for e in events:
        if e.causation_id:
            by_cause.setdefault(e.causation_id, []).append(e)

    def result_of(req: Event, *types: str) -> Event | None:
        return next(
            (r for r in by_cause.get(req.id, []) if r.type in types and r.id == f"{req.id}:result"),
            None,
        )

    tool_reqs: dict[str, Event] = {}
    tool_started: dict[str, list[Event]] = {}
    tool_done: dict[str, Event] = {}
    tool_notes: dict[str, list[Event]] = {}
    for e in events:
        cid = e.data.get("call_id") if isinstance(e.data, dict) else None
        if not cid:
            continue
        if e.type == "aaa.tool.requested":
            tool_reqs[cid] = e
        elif e.type == "aaa.tool.started":
            tool_started.setdefault(cid, []).append(e)
        elif e.type in ("aaa.tool.completed", "aaa.tool.failed"):
            tool_done[cid] = e
        elif e.type in (
            "aaa.approval.requested",
            "aaa.approval.granted",
            "aaa.approval.denied",
            "aaa.tool.fenced",
            "aaa.tool.stale_result",
        ):
            tool_notes.setdefault(cid, []).append(e)

    for e in events:
        if e.type == "aaa.llm.requested":
            res = result_of(e, "aaa.llm.completed", "aaa.llm.failed")
            req = e.data.get("request") or {}
            model = req.get("model")
            sp = make(
                "llm",
                span_id_of("llm", e.id),
                run_sid,
                f"chat {model or 'model'}",
                e.time,
                res.time if res else None,
            )
            sp.attributes |= {A.OPERATION: "chat", **({A.MODEL: model} if model else {})}
            if tg := e.data.get("tag"):
                sp.attributes["aaa.tag"] = tg
            if res is None:
                sp.status = "incomplete"
            elif res.type == "aaa.llm.failed":
                err = res.data.get("error") or {}
                sp.status, sp.status_message = "error", f"{err.get('type')}: {err.get('message')}"
                sp.attributes[A.ERROR_TYPE] = err.get("type")
            else:
                r = res.data.get("response") or {}
                sp.attributes |= _usage(res.data)
                if r.get("model"):
                    sp.attributes[A.MODEL] = r["model"]
                if r.get("finish_reason"):
                    sp.attributes[A.FINISH_REASON] = r["finish_reason"]
                if res.data.get("state"):
                    sp.attributes[A.LLM_STATE] = res.data["state"]
                if ctx := res.data.get("context"):
                    sp.attributes |= {
                        A.CONTEXT_BEFORE: ctx.get("tokens_before"),
                        A.CONTEXT_AFTER: ctx.get("tokens_after"),
                        A.CONTEXT_APPLIED: ",".join(ctx.get("applied") or []),
                    }
                if capture_content:
                    sp.output = _clip(r.get("message"), content_limit)
            sp.name = f"chat {sp.attributes.get(A.MODEL) or 'model'}"  # the response's model if the reply named one
            if capture_content:
                conv = e.data.get("conv")
                sp.input = _clip(conv["delta"] if conv else req.get("messages"), content_limit)
            spans.append(sp)

        elif e.type == "aaa.tool.requested":
            cid = e.data.get("call_id", e.id)
            starts, done = tool_started.get(cid, []), tool_done.get(cid)
            name = e.data.get("name", "tool")
            sp = make(
                "tool",
                span_id_of("tool", f"{run_id}:{cid}"),
                run_sid,
                f"execute_tool {name}",
                e.time,
                done.time if done else None,
            )
            sp.attributes |= {
                A.OPERATION: "execute_tool",
                A.TOOL_NAME: name,
                A.TOOL_CALL_ID: cid,
                A.TOOL_ATTEMPTS: len(starts),
            }
            if done is None:
                sp.status = "incomplete"
            elif done.type == "aaa.tool.failed":
                err = done.data.get("error") or {}
                sp.status, sp.status_message = "error", f"{err.get('type')}: {err.get('message')}"
                sp.attributes[A.ERROR_TYPE] = err.get("type")
            for n in tool_notes.get(cid, []):
                sp.events.append(
                    SpanEvent(
                        n.type.removeprefix("aaa."),
                        n.time,
                        {
                            k: v
                            for k, v in n.data.items()
                            if k in ("attempt", "reason", "superseded_by")
                        },
                    )
                )
            for s in starts[1:]:
                sp.events.append(
                    SpanEvent("tool.retry", s.time, {"attempt": s.data.get("attempt")})
                )
            if capture_content:
                sp.input = _clip(e.data.get("arguments"), content_limit)
                if done is not None:
                    sp.output = _clip(
                        done.data.get("result", done.data.get("error")), content_limit
                    )
            spans.append(sp)

        elif e.type == "aaa.memory.requested":
            res = result_of(e, "aaa.memory.completed", "aaa.memory.failed")
            op = e.data.get("op", "?")
            sp = make(
                "memory",
                span_id_of("memory", e.id),
                run_sid,
                f"memory {op}",
                e.time,
                res.time if res else None,
            )
            sp.attributes |= {A.OPERATION: "memory", A.MEMORY_OP: op}
            if res is None:
                sp.status = "incomplete"
            elif res.type == "aaa.memory.failed":
                sp.status, sp.status_message = (
                    "error",
                    str((res.data.get("error") or {}).get("message")),
                )
            else:
                sp.attributes |= _usage(res.data)
                if "items" in res.data:
                    sp.attributes[A.MEMORY_ITEMS] = len(res.data["items"])
            if capture_content:
                sp.input = _clip(e.data.get("query") or e.data.get("entries"), content_limit)
            spans.append(sp)

        elif e.type in ("aaa.context.summary", "aaa.session.summary"):
            sp = make("internal", span_id_of("summary", e.id), run_sid, "summarize", e.time, e.time)
            sp.attributes |= {A.OPERATION: "summarize", **_usage(e.data)}
            spans.append(sp)

        elif e.type == "aaa.session.message":
            n = e.data.get("turn")
            reply = next(
                (r for r in events if r.type == "aaa.session.reply" and r.data.get("turn") == n),
                None,
            )
            sp = make(
                "session",
                span_id_of("turn", f"{run_id}:turn:{n}"),
                run_sid,
                f"turn {n}",
                e.time,
                reply.time if reply else None,
            )
            sp.attributes |= {A.OPERATION: "turn", A.TURN: n, A.SESSION_ID: run_id}
            if reply is None:
                sp.status = "incomplete"
            elif reply.data.get("kind") != "success":
                sp.status, sp.status_message = (
                    "error",
                    str(reply.data.get("reason") or reply.data.get("kind")),
                )
            if capture_content:
                sp.input = _clip(e.data.get("text"), content_limit)
                if reply is not None:
                    sp.output = _clip(reply.data.get("text"), content_limit)
            spans.append(sp)

        elif e.type == "aaa.guard.verdict":
            sp = make(
                "guardrail",
                span_id_of("guard", e.id),
                run_sid,
                f"guard {e.data.get('guard', '?')}",
                e.time,
                e.time,
            )
            sp.attributes |= {
                A.OPERATION: "guard",
                A.GUARD_NAME: e.data.get("guard"),
                A.GUARD_STAGE: e.data.get("stage"),
                A.GUARD_ACTION: e.data.get("action"),
                **_usage(e.data),
            }
            if e.data.get("action") == "block":
                sp.status, sp.status_message = (
                    "error",
                    f"blocked: {','.join(e.data.get('categories') or [])}",
                )
            spans.append(sp)

        elif e.type == "aaa.eval.score":
            sp = make(
                "evaluator",
                span_id_of("eval", e.id),
                run_sid,
                f"evaluate {e.data.get('name', '?')}",
                e.time,
                e.time,
            )
            sp.attributes |= {
                A.OPERATION: "evaluate",
                A.SCORE_NAME: e.data.get("name"),
                A.SCORE_VALUE: e.data.get("value"),
                **_usage(e.data),
            }
            spans.append(sp)

        elif e.type in ("aaa.run.cancel", "aaa.timer.fired", "aaa.budget.exceeded"):
            root.events.append(
                SpanEvent(
                    e.type.removeprefix("aaa."),
                    e.time,
                    {"reason": e.data.get("reason", e.data.get("kind"))},
                )
            )

    spans.sort(key=lambda s: (s.start, s.span_id))
    return spans


def children_of(events: Sequence[Event]) -> list[str]:
    """Run ids of the child runs started by this run (from its `aaa.child.started` events)."""
    return [
        e.data["child_run_id"]
        for e in events
        if e.type == "aaa.child.started" and e.data.get("child_run_id")
    ]
