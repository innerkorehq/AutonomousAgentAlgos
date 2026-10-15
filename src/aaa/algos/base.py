from __future__ import annotations

import json
import re
from typing import Any

from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import LLMRequest, Message


def llm_request(
    parent: Event, req: LLMRequest, tag: str = "main", *, prev: str | None = None
) -> Event:
    """Build an `aaa.llm.requested` event.

    With `prev` (the id of the previous request event of the SAME conversation), `req.messages`
    must hold only the messages appended since that request's reply; the event then stores just
    that delta plus a pointer. The LLM handler reconstructs the full transcript the provider
    needs, so the log (and algorithm state) grows linearly with turns, not quadratically."""
    body = req.model_dump(mode="json")
    data: dict[str, Any] = {"request": body, "tag": tag}
    if prev is not None:
        data["conv"] = {"prev": prev, "delta": body["messages"]}
        body["messages"] = []
    return parent.caused("aaa.llm.requested", data)


def finish(parent: Event, kind: OutcomeKind, value: Any = None, reason: str | None = None) -> Event:
    return parent.caused(
        "aaa.run.completed", {"kind": kind.value, "value": value, "reason": reason}
    )


def llm_failed(e: Event) -> Event:
    """End the run because its model call failed: `BLOCKED` when a guardrail did it, else `FAILED`."""
    err = e.data["error"]
    kind = OutcomeKind.BLOCKED if err["type"] == "GuardrailBlocked" else OutcomeKind.FAILED
    return finish(e, kind, reason=f"{err['type']}: {err['message']}")


def tool_request(parent: Event, call_id: str, name: str, arguments: dict[str, Any]) -> Event:
    return parent.caused(
        "aaa.tool.requested", {"call_id": call_id, "name": name, "arguments": arguments}
    )


_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_fence(text: str) -> str:
    t = text.strip()
    m = _FENCE.match(t)
    return m.group(1) if m else t


def parse_json_obj(text: str | None) -> dict[str, Any] | None:
    """Best-effort: a JSON object from raw model text (fenced or surrounded by prose)."""
    if not text:
        return None
    t = strip_fence(text)
    for cand in (t, t[t.find("{") : t.rfind("}") + 1] if "{" in t else ""):
        try:
            v = json.loads(cand)
        except ValueError:
            continue
        if isinstance(v, dict):
            return v
    return None


def render_result(payload: Any) -> str:
    return payload if isinstance(payload, str) else json.dumps(payload, default=str)


def first_started(hist: list[Event]) -> Event:
    return next(e for e in hist if e.type == "aaa.run.started")


def task_messages(inp: dict[str, Any], system: str) -> list[Message]:
    if "messages" in inp:
        return [Message.model_validate(m) for m in inp["messages"]]
    msgs = [Message(role="system", content=system)] if system else []
    msgs.append(Message(role="user", content=str(inp.get("task", ""))))
    return msgs
