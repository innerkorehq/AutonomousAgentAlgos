from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, model_serializer

from aaa.algos.base import (
    finish,
    llm_failed,
    llm_request,
    render_result,
    task_messages,
    tool_request,
)
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import LLMRequest, LLMResponse, Message, ResumeHint
from aaa.runtime import Ctx
from aaa.tools import ToolRegistry

log = logging.getLogger("aaa.loops")
MEMORY_REQUESTED = "aaa.memory.requested"
MEMORY_COMPLETED = "aaa.memory.completed"
MEMORY_FAILED = "aaa.memory.failed"


def _default_query(inp: dict[str, Any]) -> str:
    if inp.get("task") is not None:
        return str(inp["task"])
    users = [m for m in inp.get("messages") or [] if m.get("role") == "user"]
    return str(users[-1].get("content", "")) if users else ""


@dataclass
class RecallConfig:
    """Look up long-term memory before the first model call and put it in the prompt.

    Needs a `MemoryHandler` attached to the engine (without one, recall is skipped with a warning).
    A failing or slow memory backend never fails the run: the loop simply continues without memories."""

    k: int = 5
    query: Callable[[dict[str, Any]], str] = (
        _default_query  # what to search for, from the run input
    )
    max_chars: int = 2000  # how much recalled text may enter the prompt
    as_of: str | None = None  # ISO timestamp for point-in-time recall on temporal backends


class LoopState(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None  # last `main` request: the next one continues from it
    turns: int = 0
    phase: str = "start"  # start | llm | tools | done
    round_calls: list[str] = Field(default_factory=list)
    names: dict[str, str] = Field(default_factory=dict)
    results: dict[str, tuple[bool, Any]] = Field(default_factory=dict)
    recalled: list[str] = Field(default_factory=list)  # memory lines to show the model
    # Swarm only. An ordinary loop's state serializes exactly as it did before swarms existed (see _omit_swarm_fields).
    handoffs: list[list[str]] = Field(default_factory=list)  # [from, to] pairs, in order
    agent: str | None = None  # the active agent, set by `aaa.swarm.handoff`

    @model_serializer(mode="wrap")
    def _omit_swarm_fields(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        for k in ("agent", "handoffs"):
            if not d.get(k):
                d.pop(k, None)
        return d


def apply(st: LoopState, e: Event) -> LoopState:
    """Incremental fold of the run log into loop state: nothing lives in the algorithm object."""
    d = e.data
    if e.type == "aaa.run.started":
        st.input = d.get("input", {})
    elif e.type == MEMORY_REQUESTED and d.get("tag") == "recall":
        st.phase = "recall"
    elif e.type == MEMORY_COMPLETED and d.get("tag") == "recall":
        st.recalled = [_memory_line(m) for m in d.get("items", [])]
    elif e.type == "aaa.llm.requested" and d.get("tag") == "main":
        st.request_id = e.id
        st.phase, st.round_calls, st.results = "llm", [], {}
    elif e.type == "aaa.llm.completed" and d.get("tag") == "main":
        msg = LLMResponse.model_validate(d["response"]).message
        st.turns += 1
        st.phase = "tools" if msg.tool_calls else "done"
        st.round_calls = [c.id for c in msg.tool_calls]
        st.names = {c.id: c.name for c in msg.tool_calls}
    elif e.type == "aaa.swarm.handoff":
        st.agent = d["to"]
        st.handoffs.append([d["from"], d["to"]])
    elif e.type == "aaa.tool.requested":
        if d["call_id"] not in st.round_calls:
            st.round_calls.append(d["call_id"])
        st.names[d["call_id"]] = d["name"]
        st.phase = "tools"
    elif e.type == "aaa.tool.completed":
        st.results[d["call_id"]] = (True, d["result"])
    elif e.type == "aaa.tool.failed":
        st.results[d["call_id"]] = (False, d["error"])
    return st


def _memory_line(m: dict[str, Any]) -> str:
    since = f" (since {str(m['valid_at'])[:10]})" if m.get("valid_at") else ""
    return f"- {m['text']}{since}"


LOOP = Projection(LoopState, apply, name="aaa.loop", version=4, state_type=LoopState)


class LLMLoop:
    """Shared engine for tool-using LLM loops. Subclasses decide how the model's reply maps
    to actions (`on_reply`) and how tool results are shown back (`observations`)."""

    name = "llm_loop"

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        system: str = "",
        model: str | None = None,
        max_turns: int = 10,
        params: dict[str, Any] | None = None,
        tool_names: list[str] | None = None,
        name: str | None = None,
        stream: bool = False,
        recall: RecallConfig | bool | None = None,
    ) -> None:
        if name:
            self.name = name
        self.tools, self.system, self.model = tools, system, model
        self.stream = stream  # ask the LLM handler to stream (emits aaa.llm.delta events)
        self.max_turns, self.params, self.tool_names = max_turns, params or {}, tool_names
        self.recall = RecallConfig() if recall is True else (recall or None)

    def subscriptions(self) -> list[str]:
        return [
            "aaa.run.started",
            "aaa.llm.completed",
            "aaa.llm.failed",
            "aaa.tool.completed",
            "aaa.tool.failed",
            *([MEMORY_COMPLETED, MEMORY_FAILED] if self.recall else []),
        ]

    # hooks -----------------------------------------------------------------------------
    def initial_messages(self, inp: dict[str, Any]) -> list[Message]:
        return task_messages(inp, self.system)

    def request(self, messages: list[Message]) -> LLMRequest:
        return LLMRequest(
            model=self.model, messages=messages, params=self.params, stream=self.stream
        )

    def request_for(self, st: LoopState, messages: list[Message]) -> LLMRequest:
        """The request for the next model call given the loop state (a hook for loops whose tools or model
        depend on state, like `Swarm`). Ordinary loops ignore the state."""
        return self.request(messages)

    def first_request(self, st: LoopState, messages: list[Message]) -> LLMRequest:
        """The run's first model request. If the input says a provider already holds the start of this
        conversation (`input["provider_state"] = {"state_id": ..., "covered": n}`, set by sessions), say so."""
        req = self.request_for(st, messages)
        if hint := st.input.get("provider_state"):
            req.resume = ResumeHint.model_validate(hint)
        return req

    def on_reply(self, e: Event, st: LoopState, msg: Message) -> list[Event]:
        raise NotImplementedError

    def observations(self, st: LoopState) -> list[Message]:
        raise NotImplementedError

    def with_memory(self, messages: list[Message], st: LoopState) -> list[Message]:
        """Insert the recalled memories as a system message after the existing system prompt."""
        if not st.recalled or self.recall is None:
            return messages
        lines: list[str] = []
        used = 0
        for line in st.recalled:
            if used + len(line) > self.recall.max_chars and lines:
                break
            lines.append(line)
            used += len(line) + 1
        note = Message(
            role="system",
            content="What you remember from earlier (use only if relevant):\n" + "\n".join(lines),
        )
        at = next((i for i, m in enumerate(messages) if m.role != "system"), len(messages))
        return [*messages[:at], note, *messages[at:]]

    # driver ----------------------------------------------------------------------------
    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type in (MEMORY_COMPLETED, MEMORY_FAILED):
            if e.data.get("tag") != "recall":
                return []
            st = await ctx.project(LOOP)  # recall finished (or failed): now ask the model
            msgs = self.with_memory(self.initial_messages(st.input), st)
            return [llm_request(e, self.first_request(st, msgs))]
        if e.data.get("tag", "main") != "main":
            return []
        st = await ctx.project(LOOP)
        if e.type == "aaa.run.started":
            if self.recall is not None:
                if ctx.engine.memories:
                    return [
                        e.caused(
                            MEMORY_REQUESTED,
                            {
                                "op": "recall",
                                "tag": "recall",
                                "query": self.recall.query(st.input),
                                "k": self.recall.k,
                                **({"as_of": self.recall.as_of} if self.recall.as_of else {}),
                            },
                        )
                    ]
                log.warning("recall is enabled but no memory backend is attached; skipping it")
            return [llm_request(e, self.first_request(st, self.initial_messages(st.input)))]
        if e.type == "aaa.llm.failed":
            return [llm_failed(e)]
        if e.type == "aaa.llm.completed":
            return self.on_reply(e, st, LLMResponse.model_validate(e.data["response"]).message)
        # tool result: continue once every call of this round has resolved
        if st.phase == "tools" and st.round_calls and all(c in st.results for c in st.round_calls):
            return [llm_request(e, self.request_for(st, self.observations(st)), prev=st.request_id)]
        return []

    def _budget_exhausted(self, e: Event, st: LoopState) -> Event | None:
        if st.turns >= self.max_turns:
            return finish(
                e,
                OutcomeKind.NO_SOLUTION,
                reason=f"max_turns ({self.max_turns}) reached without a final answer",
            )
        return None


class ToolLoop(LLMLoop):
    """Native function-calling loop: model proposes tool calls, results are fed back, until it
    answers in plain text. Bad/unknown/failed calls come back to the model as errors."""

    name = "tool_loop"

    def request(self, messages: list[Message]) -> LLMRequest:
        r = super().request(messages)
        r.tools = self.tools.specs(self.tool_names)
        return r

    def on_reply(self, e: Event, st: LoopState, msg: Message) -> list[Event]:
        if not msg.tool_calls:
            return [finish(e, OutcomeKind.SUCCESS, {"text": msg.content or "", "turns": st.turns})]
        if stop := self._budget_exhausted(e, st):
            return [stop]
        out: list[Event] = []
        for c in msg.tool_calls:
            if c.raw_arguments is not None:  # model emitted malformed JSON: tell it, don't guess
                out.append(
                    e.caused(
                        "aaa.tool.failed",
                        {
                            "call_id": c.id,
                            "name": c.name,
                            "error": {
                                "type": "InvalidArguments",
                                "message": f"arguments were not valid JSON: {c.raw_arguments[:200]}",
                            },
                        },
                    )
                )
            else:
                out.append(tool_request(e, c.id, c.name, c.arguments))
        return out

    def observations(self, st: LoopState) -> list[Message]:
        msgs = []
        for cid in st.round_calls:
            ok, payload = st.results[cid]
            body = (
                render_result(payload) if ok else f"Error ({payload['type']}): {payload['message']}"
            )
            msgs.append(
                Message(role="tool", tool_call_id=cid, name=st.names.get(cid), content=body)
            )
        return msgs


_REACT_PROMPT = """Answer the task by reasoning step by step and using tools when needed.

Tools:
{tools}

Use exactly this format:
Thought: your reasoning
Action: <one tool name>
Action Input: <JSON object of arguments>
(then stop and wait; the system replies with "Observation: ...")
... repeat Thought/Action/Action Input/Observation as needed ...
When you know the answer:
Thought: I now know the final answer
Final Answer: <answer>"""

_FINAL = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
_ACTION = re.compile(r"Action:\s*([^\n]+)")
_INPUT = re.compile(r"Action Input:\s*(.*)", re.DOTALL)


class ReAct(LLMLoop):
    """Classic text-protocol ReAct (Thought/Action/Observation) for models or gateways
    without native tool calling. Prefer ToolLoop when tool calling is available."""

    name = "react"

    def __init__(self, tools: ToolRegistry, **kw: Any) -> None:
        super().__init__(tools, **kw)
        self.params.setdefault("stop", ["\nObservation:"])
        if not self.system:
            lines = "\n".join(
                f"- {s.name}: {s.description} | args schema: {json.dumps(s.parameters.get('properties', {}))}"
                for s in tools.specs(self.tool_names)
            )
            self.system = _REACT_PROMPT.format(tools=lines)

    def on_reply(self, e: Event, st: LoopState, msg: Message) -> list[Event]:
        text = msg.content or ""
        action, final = _ACTION.search(text), _FINAL.search(text)
        if final and (not action or final.start() < action.start()):
            return [
                finish(e, OutcomeKind.SUCCESS, {"text": final.group(1).strip(), "turns": st.turns})
            ]
        if not action:  # no protocol markers: treat the reply as the answer
            return [finish(e, OutcomeKind.SUCCESS, {"text": text.strip(), "turns": st.turns})]
        if stop := self._budget_exhausted(e, st):
            return [stop]
        name = action.group(1).strip().strip("`")
        m_input = _INPUT.search(text)
        raw = (m_input.group(1).strip() if m_input else "{}") or "{}"
        try:
            args = json.loads(raw.split("\nObservation:")[0].strip())
            if not isinstance(args, dict):
                raise TypeError
        except (ValueError, TypeError):
            return [
                self._feedback(
                    e, st, f"Error: Action Input must be a JSON object, got: {raw[:200]}"
                )
            ]
        return [tool_request(e, f"react-{e.id}", name, args)]

    def _feedback(self, e: Event, st: LoopState, text: str) -> Event:
        msgs = [Message(role="user", content=f"Observation: {text}")]
        return llm_request(e, self.request(msgs), prev=st.request_id)

    def observations(self, st: LoopState) -> list[Message]:
        cid = st.round_calls[-1]
        ok, payload = st.results[cid]
        body = render_result(payload) if ok else f"Error ({payload['type']}): {payload['message']}"
        return [Message(role="user", content=f"Observation: {body}")]
