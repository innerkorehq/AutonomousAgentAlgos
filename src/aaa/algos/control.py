from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from aaa.algos.base import finish, tool_request
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.runtime import Ctx


@dataclass
class Node:
    """One workflow step = one tool call. `args(results, input)` builds its arguments from
    upstream outputs; `when(results, input)` can skip it (skipped nodes yield None)."""

    id: str
    tool: str
    after: list[str] = field(default_factory=list)
    args: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | dict[str, Any] | None = None
    when: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None
    on_error: str = "fail"  # "fail" | "continue"


class _WF(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    errors: dict[str, str] = Field(
        default_factory=dict
    )  # raw failures; on_error policy applied at read time
    requested: list[str] = Field(default_factory=list)


def _apply_wf(st: _WF, e: Event) -> _WF:
    d = e.data
    if e.type == "aaa.run.started":
        st.input = d["input"]
    elif e.type == "aaa.tool.requested":
        st.requested.append(d["call_id"])
    elif e.type == "aaa.node.skipped":
        st.requested.append(d["node_id"])
        st.results[d["node_id"]] = None
    elif e.type == "aaa.tool.completed":
        st.results[d["call_id"]] = d["result"]
    elif e.type == "aaa.tool.failed":
        st.errors[d["call_id"]] = f"{d['error']['type']}: {d['error']['message']}"
    return st


WF = Projection(_WF, _apply_wf, name="aaa.workflow", version=1, state_type=_WF)


class WorkflowGraph:
    """Deterministic DAG execution: no LLM decides control flow. Ready nodes run in parallel;
    tools bring approval gates, timeouts, idempotency and budgets. Cyclic/stateful control
    flow belongs to FSM. Output = `output` node id, a callable(results), or all results."""

    name = "workflow"

    def __init__(
        self, nodes: list[Node], *, output: str | Callable[[dict[str, Any]], Any] | None = None
    ) -> None:
        ids = [n.id for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate node ids")
        for n in nodes:
            if missing := [a for a in n.after if a not in ids]:
                raise ValueError(f"node {n.id} depends on unknown {missing}")
        self.nodes = {n.id: n for n in nodes}
        self._check_acyclic()
        self.output = output

    def _check_acyclic(self) -> None:
        seen: set[str] = set()
        todo = set(self.nodes)
        while todo:
            ready = {i for i in todo if all(a in seen for a in self.nodes[i].after)}
            if not ready:
                raise ValueError("workflow has a dependency cycle (use FSM for loops)")
            seen |= ready
            todo -= ready

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.tool.completed", "aaa.tool.failed"]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(WF)
        inp, requested = st.input, set(st.requested)
        results: dict[str, Any] = dict(st.results)
        failed: dict[str, str] = {}
        for k, msg in st.errors.items():
            if self.nodes[k].on_error == "continue":
                results[k] = None
            else:
                failed[k] = msg
        if failed:
            k, msg = next(iter(failed.items()))
            return [finish(e, OutcomeKind.FAILED, reason=f"node {k} failed: {msg}")]
        out: list[Event] = []
        progress = True
        while progress:  # fixpoint: skipping a node can make its dependents ready
            progress = False
            for n in self.nodes.values():
                if n.id in requested or not all(a in results for a in n.after):
                    continue
                requested.add(n.id)
                progress = True
                if n.when and not n.when(results, inp):
                    results[n.id] = None
                    out.append(e.caused("aaa.node.skipped", {"node_id": n.id}))
                    continue
                args = n.args(results, inp) if callable(n.args) else dict(n.args or {})
                out.append(tool_request(e, n.id, n.tool, args))
        if all(i in results for i in self.nodes):
            val = (
                results[self.output]
                if isinstance(self.output, str)
                else (self.output(results) if self.output else results)
            )
            return [*out, finish(e, OutcomeKind.SUCCESS, val)]
        return out


class _FSM(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    path: list[str] = Field(default_factory=list)  # states entered after `initial`


def _apply_fsm(st: _FSM, e: Event) -> _FSM:
    if e.type == "aaa.run.started":
        st.input = e.data["input"]
    elif e.type == "aaa.fsm.transitioned":
        st.path.append(e.data["to"])
    return st


FSM_STATE = Projection(_FSM, _apply_fsm, name="aaa.fsm", version=1, state_type=_FSM)

Transition = str | Callable[[Event, dict[str, Any]], str | None]


class FSM:
    """Explicit finite-state machine driven by external/internal events. `transitions[state]
    [event_type]` is the next state or a function(event, input) -> next state | None (guard).
    `on_enter[state](event, input)` may return events (request a tool, an LLM call, ...).
    Reaching a terminal state completes the run with the path taken. The current state is a
    fold of `aaa.fsm.transitioned` events, so it survives restarts."""

    name = "fsm"

    def __init__(
        self,
        initial: str,
        transitions: dict[str, dict[str, Transition]],
        terminal: set[str],
        on_enter: dict[str, Callable[[Event, dict[str, Any]], list[Event]]] | None = None,
    ) -> None:
        self.initial, self.transitions, self.terminal = initial, transitions, terminal
        self.on_enter = on_enter or {}
        self._types = sorted({t for m in transitions.values() for t in m})

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", *self._types]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(FSM_STATE)
        inp = st.input
        path = [self.initial, *st.path]
        cur = path[-1]
        if e.type == "aaa.run.started":
            return [*self.on_enter.get(cur, lambda *_: [])(e, inp)]
        rule = self.transitions.get(cur, {}).get(e.type)
        nxt = rule(e, inp) if callable(rule) else rule
        if nxt is None:
            return [e.caused("aaa.fsm.ignored", {"state": cur, "event": e.type})]
        out = [e.caused("aaa.fsm.transitioned", {"from": cur, "to": nxt, "on": e.type})]
        out += self.on_enter.get(nxt, lambda *_: [])(e, inp)
        if nxt in self.terminal:
            out.append(finish(e, OutcomeKind.SUCCESS, {"state": nxt, "path": [*path, nxt]}))
        return out
