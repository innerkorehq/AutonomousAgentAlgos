from __future__ import annotations

import json
import re
from typing import Any

from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import LLMRequest, Message


def llm_request(parent: Event, req: LLMRequest, tag: str = "main") -> Event:
    return parent.caused("aaa.llm.requested", {"request": req.model_dump(mode="json"), "tag": tag})


def finish(parent: Event, kind: OutcomeKind, value: Any = None, reason: str | None = None) -> Event:
    return parent.caused(
        "aaa.run.completed", {"kind": kind.value, "value": value, "reason": reason}
    )


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
