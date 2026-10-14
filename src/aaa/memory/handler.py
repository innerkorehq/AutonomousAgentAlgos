"""Memory as events: `aaa.memory.requested` in, `aaa.memory.completed|failed` out.

Routing memory through the event log (instead of calling a library from inside an algorithm) buys:

* **replay**: the recalled items are in the log, so a replayed run sees exactly what the original saw
  even though the backend has since changed;
* **privacy**: requests and results pass through the engine's redactor before storage, and the scope
  comes from the run's tenant and metadata, never from anything the model wrote;
* **crash safety**: results use the deterministic id `<request id>:result`, and writes to a backend that
  cannot de-duplicate are recorded as an attempt first, so a worker that dies mid-write is reported
  as `OutcomeUnknown` instead of silently writing twice (the same idea as the tool execution guard);
* **degradation**: a failing memory backend produces a `failed` event, never a failed run.

Request payload (`aaa.memory.requested`): `op` (`recall` | `remember` | `forget`), `tag`, and
`query`/`k`/`filters`/`as_of` (recall), `entries` (remember), `ids` (forget by id)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from aaa.core.aio import wait_for
from aaa.core.retry import RetryPolicy, retrying
from aaa.events import Event
from aaa.memory.base import Deletable, Memory, is_idempotent
from aaa.memory.types import AddResult, MemoryEntry, Scope
from aaa.runtime import Engine

log = logging.getLogger("aaa.memory")

REQUESTED = "aaa.memory.requested"
STARTED = "aaa.memory.started"
COMPLETED = "aaa.memory.completed"
FAILED = "aaa.memory.failed"
RUN_COMPLETED = "aaa.run.completed"

Delivery = Literal["at_least_once", "at_most_once"]


def default_scope(e: Event) -> Scope:
    """Whose memory an event touches: the run's tenant, plus `user`/`agent` from the run metadata
    set at `start(meta=...)`. The session is the run. Nothing here comes from model output."""
    user, agent = e.meta.get("user"), e.meta.get("agent")
    return Scope(
        tenant=e.tenant,
        user=str(user) if user is not None else None,
        agent=str(agent) if agent is not None else None,
        session=e.run_id,
    )


EntryBuilder = Callable[[Event, Event], Sequence[MemoryEntry]]


def default_entries(started: Event, completed: Event) -> list[MemoryEntry]:
    """The task and the answer of a finished run, as a conversation turn pair."""
    inp = started.data.get("input") or {}
    task = inp.get("task")
    if task is None and inp.get("messages"):
        users = [m for m in inp["messages"] if m.get("role") == "user"]
        task = users[-1].get("content") if users else None
    value = completed.data.get("value")
    answer = value.get("text") if isinstance(value, dict) else value
    out: list[MemoryEntry] = []
    if task:
        out.append(MemoryEntry(text=str(task), role="user", kind="episode"))
    if answer:
        out.append(MemoryEntry(text=str(answer), role="assistant", kind="episode"))
    return out


@dataclass
class WriteBack:
    """Remember what happened when a run finishes. `build(started, completed)` chooses what; the
    default stores the task and the answer. Only successful runs are written unless `only_success=False`."""

    build: EntryBuilder = default_entries
    only_success: bool = True


class MemoryHandler:
    def __init__(
        self,
        memory: Memory,
        *,
        scope_for: Callable[[Event], Scope] = default_scope,
        write_back: WriteBack | None = None,
        delivery: Delivery | None = None,
        retry: RetryPolicy | None = None,
        concurrency: int = 8,
        timeout_s: float = 20.0,
    ) -> None:
        self.memory, self.scope_for, self.write_back = memory, scope_for, write_back
        # A backend that can de-duplicate keyed writes can be retried freely; one that cannot must not be.
        self.delivery: Delivery = delivery or (
            "at_least_once" if is_idempotent(memory) else "at_most_once"
        )
        self.retry = retry or RetryPolicy(max_attempts=3, max_elapsed_s=10.0, base_delay_s=0.2)
        self.concurrency, self.timeout_s = concurrency, timeout_s
        self.engine: Engine | None = None

    def attach(self, engine: Engine) -> None:
        self.engine = engine
        if self.memory not in engine.memories:
            engine.memories.append(self.memory)
        engine.subscribe(REQUESTED, self.handle, group="aaa.memory", concurrency=self.concurrency)
        if self.write_back is not None:
            engine.subscribe(RUN_COMPLETED, self.on_run_completed, group="aaa.memory.writeback")

    # -- request/response -------------------------------------------------------------------
    async def handle(self, e: Event) -> list[Event]:
        op = str(e.data.get("op", ""))
        tag = e.data.get("tag", "main")
        try:
            body = await self._execute(op, e, self.scope_for(e), key=e.id)
        except _OutcomeUnknown as exc:
            return [self._failed(e, op, tag, "OutcomeUnknown", str(exc))]
        except Exception as exc:  # noqa: BLE001 - a memory backend must never fail the run
            log.warning("memory %s failed: %s: %s", op, type(exc).__name__, exc)
            return [self._failed(e, op, tag, type(exc).__name__, str(exc))]
        return [e.caused(COMPLETED, {"op": op, "tag": tag, **body}, id=f"{e.id}:result")]

    @staticmethod
    def _failed(e: Event, op: str, tag: str, etype: str, message: str) -> Event:
        err = {"type": etype, "message": message}
        return e.caused(FAILED, {"op": op, "tag": tag, "error": err}, id=f"{e.id}:result")

    async def _execute(self, op: str, e: Event, scope: Scope, *, key: str) -> dict[str, Any]:
        d = e.data
        if op == "recall":
            as_of = datetime.fromisoformat(d["as_of"]) if d.get("as_of") else None
            items = await self._call(
                lambda: self.memory.search(
                    str(d.get("query", "")),
                    scope,
                    k=int(d.get("k", 5)),
                    filters=d.get("filters"),
                    as_of=as_of,
                )
            )
            return {"items": [m.model_dump(mode="json") for m in items]}
        if op == "remember":
            entries = [MemoryEntry.model_validate(x) for x in d.get("entries", [])]
            result = await self._write(e, key, lambda: self.memory.add(entries, scope, key=key))
            return {
                "ids": result.ids,
                "pending": result.pending,
                "usage": result.usage,
                "cost": result.cost,
            }
        if op == "forget":
            mem = self.memory
            if not isinstance(mem, Deletable):
                raise NotImplementedError(f"{type(mem).__name__} cannot delete by id")
            n = await self._call(lambda: mem.delete(list(d.get("ids", [])), scope))
            return {"deleted": n}
        raise ValueError(f"unknown memory op {op!r}")

    async def _call(self, fn: Callable[[], Any]) -> Any:
        return await retrying(
            lambda: wait_for(fn(), self.timeout_s), self.retry, what="memory call"
        )

    async def _write(self, e: Event, key: str, fn: Callable[[], Any]) -> AddResult:
        if self.delivery == "at_least_once":
            return await self._call(fn)
        assert self.engine is not None
        marker = await self.engine.emit(e.caused(STARTED, {"op": "remember"}, id=f"{e.id}:attempt"))
        if marker is not None and marker.dup:
            raise _OutcomeUnknown(
                f"an earlier attempt to write memory for {e.id} started but never reported; "
                f"{type(self.memory).__name__} cannot de-duplicate, so it was NOT retried"
            )
        return await fn()

    # -- write-back at run end --------------------------------------------------------------
    async def on_run_completed(self, e: Event) -> list[Event]:
        wb = self.write_back
        if wb is None or not e.run_id or e.data.get("$erased"):
            return []
        if wb.only_success and e.data.get("kind") != "success":
            return []
        assert self.engine is not None
        started = await self.engine._started_of(e.run_id)
        entries = list(wb.build(started, e))
        if not entries:
            return []
        req = e.caused(
            REQUESTED,
            {
                "op": "remember",
                "tag": "writeback",
                "entries": [m.model_dump(mode="json") for m in entries],
            },
            id=f"{e.run_id}:writeback",
        )
        # The run is finished, so the bus would drop a request event for it: do the write inline and
        # record the outcome. The deterministic ids make a repeat (redelivery) a no-op.
        return await self.handle(req)

    # -- helpers for erasure ------------------------------------------------------------------
    async def forget(self, scope: Scope) -> int:
        from aaa.memory.base import Forgettable

        if not isinstance(self.memory, Forgettable):
            raise NotImplementedError(f"{type(self.memory).__name__} does not support forget()")
        return await self.memory.forget(scope)


class _OutcomeUnknown(Exception):
    pass


__all__ = [
    "COMPLETED",
    "FAILED",
    "REQUESTED",
    "STARTED",
    "MemoryHandler",
    "WriteBack",
    "default_entries",
    "default_scope",
]
