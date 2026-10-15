"""Swarm-style handoffs: one conversation, several specialised agents, and the model decides who speaks next.

Each `Agent` has its own instructions, its own subset of the registry's tools, and a list of agents it may hand off to. A handoff is
a tool call (`transfer_to_<agent>`) that the algorithm itself answers: it records an `aaa.swarm.handoff` event, and from the next
model call on, the target agent's tools are the only ones offered. The conversation continues unbroken.

Safety properties, because a model chooses the transitions:

* a call to a tool that the ACTIVE agent does not have (including another agent's tool, or a transfer it is not allowed to make) is
  refused as an unknown tool, whatever the model was told;
* at most one transfer per model turn, and at most `max_handoffs` per run, so two agents cannot hand a task back and forth forever;
* the registry's usual protections (argument validation, approval, the guard) still apply to every real tool call."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from aaa.algos.base import finish, tool_request
from aaa.algos.loops import LoopState, ToolLoop
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import LLMRequest, Message, ToolSpec
from aaa.tools import ToolRegistry

TRANSFER = "transfer_to_"


@dataclass
class Agent:
    instructions: str
    tools: list[str] = field(default_factory=list)  # names in the shared registry
    handoffs: list[str] = field(default_factory=list)  # names of agents it may transfer to
    description: str = ""  # shown to the other agents when they may transfer here
    model: str | None = None


class Swarm(ToolLoop):
    """A `ToolLoop` whose active agent can change. Attach `ToolHandler(swarm.tools)` as usual (the registry holds every tool;
    each agent sees only its own). Result is the final agent's reply plus the agent that gave it:
    `{"text", "turns", "agent", "handoffs": [[from, to], ...]}`."""

    name = "swarm"

    def __init__(
        self,
        agents: Mapping[str, Agent],
        tools: ToolRegistry,
        *,
        start: str,
        max_handoffs: int = 6,
        **kw: Any,
    ) -> None:
        if start not in agents:
            raise ValueError(f"start agent {start!r} is not defined")
        for n, a in agents.items():
            for t in a.tools:
                if t not in tools:
                    raise ValueError(f"agent {n!r} uses unknown tool {t!r}")
            for h in a.handoffs:
                if h not in agents:
                    raise ValueError(f"agent {n!r} may hand off to unknown agent {h!r}")
        self.agents, self.start, self.max_handoffs = dict(agents), start, max_handoffs
        kw.setdefault("system", agents[start].instructions)
        super().__init__(tools, **kw)

    # -- what each agent may do -----------------------------------------------------------------
    def _active(self, st: LoopState) -> str:
        return st.agent or self.start

    def _transfer_specs(self, name: str) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=f"{TRANSFER}{h}",
                description=f"Hand the conversation to the {h} agent. {self.agents[h].description}".strip(),
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
            )
            for h in self.agents[name].handoffs
        ]

    def request_for(self, st: LoopState, messages: list[Message]) -> LLMRequest:
        agent = self.agents[self._active(st)]
        r = self.request(messages)
        r.tools = [*self.tools.specs(agent.tools), *self._transfer_specs(self._active(st))]
        if agent.model:
            r.model = agent.model
        return r

    def request(self, messages: list[Message]) -> LLMRequest:
        r = LLMRequest(model=self.model, messages=messages, params=self.params, stream=self.stream)
        return r

    # -- handling a reply -----------------------------------------------------------------------
    def on_reply(self, e: Event, st: LoopState, msg: Message) -> list[Event]:
        active = self._active(st)
        if not msg.tool_calls:
            return [
                finish(
                    e,
                    OutcomeKind.SUCCESS,
                    {
                        "text": msg.content or "",
                        "turns": st.turns,
                        "agent": active,
                        "handoffs": self._handoffs(st),
                    },
                )
            ]
        if stop := self._budget_exhausted(e, st):
            return [stop]
        agent = self.agents[active]
        out: list[Event] = []
        transferred = False
        used = len(self._handoffs(st))
        for c in msg.tool_calls:
            if c.raw_arguments is not None:
                out.append(
                    self._fail(e, c.id, c.name, "InvalidArguments", "arguments were not valid JSON")
                )
            elif c.name.startswith(TRANSFER) and c.name[len(TRANSFER) :] in agent.handoffs:
                target = c.name[len(TRANSFER) :]
                if transferred:
                    out.append(
                        self._fail(e, c.id, c.name, "TransferRefused", "only one transfer per turn")
                    )
                elif used >= self.max_handoffs:
                    out.append(
                        self._fail(
                            e,
                            c.id,
                            c.name,
                            "TransferRefused",
                            "too many handoffs; answer the user yourself",
                        )
                    )
                else:
                    transferred = True
                    note = self.agents[target].instructions
                    out.append(
                        e.caused(
                            "aaa.swarm.handoff", {"from": active, "to": target, "call_id": c.id}
                        )
                    )
                    out.append(
                        e.caused(
                            "aaa.tool.completed",
                            {
                                "call_id": c.id,
                                "name": c.name,
                                "result": f"Transferred to {target}. You are now the {target} agent; from here on follow these instructions instead of your earlier role: {note}",
                            },
                            id=f"{e.id}:handoff:{c.id}",
                        )
                    )
            elif c.name in agent.tools:
                out.append(tool_request(e, c.id, c.name, c.arguments))
            else:  # not this agent's tool (maybe another agent's, maybe invented)
                names = [*agent.tools, *(f"{TRANSFER}{h}" for h in agent.handoffs)]
                out.append(
                    self._fail(
                        e,
                        c.id,
                        c.name,
                        "UnknownTool",
                        f"no tool named {c.name!r}; available: {names}",
                    )
                )
        return out

    @staticmethod
    def _fail(e: Event, call_id: str, name: str, etype: str, msg: str) -> Event:
        return e.caused(
            "aaa.tool.failed",
            {"call_id": call_id, "name": name, "error": {"type": etype, "message": msg}},
        )

    def _handoffs(self, st: LoopState) -> list[list[str]]:
        return st.handoffs

    def initial_messages(self, inp: dict[str, Any]) -> list[Message]:
        return super().initial_messages(inp)
