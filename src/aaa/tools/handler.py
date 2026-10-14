from __future__ import annotations

import asyncio
from typing import Any

from aaa.events import Event
from aaa.runtime import Engine
from aaa.tools.tool import Tool, ToolArgsError, ToolContext, ToolRegistry

REQUESTED = "aaa.tool.requested"
COMPLETED = "aaa.tool.completed"
FAILED = "aaa.tool.failed"
APPROVAL_REQUESTED = "aaa.approval.requested"
APPROVAL_GRANTED = "aaa.approval.granted"
APPROVAL_DENIED = "aaa.approval.denied"


class ToolHandler:
    """Executes `aaa.tool.requested` {call_id, name, arguments}.

    - Idempotent per (run, call_id): redelivered requests never re-execute a side effect.
    - Tools with requires_approval wait for an external `aaa.approval.granted|denied`
      {call_id} event (e.g. RunHandle.signal) — enforced here, outside the model's control.
    - Executions run as tracked background tasks in bounded pools, so a tool that waits on a
      child run (delegation) can never starve the pool its child needs.
    - Failures (bad args, timeout, exception) become `aaa.tool.failed` facts the model sees."""

    def __init__(self, registry: ToolRegistry, *, pools: dict[str, int] | None = None) -> None:
        self.registry = registry
        limits = {"default": 16, "delegate": 8, **(pools or {})}
        self._sems = {k: asyncio.Semaphore(v) for k, v in limits.items()}
        self._started: set[tuple[str, str]] = set()
        self._tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self.engine: Engine | None = None

    def attach(self, engine: Engine) -> None:
        """One tool executor per engine: later handlers merge their tools into the first, so
        independent algorithms (e.g. a supervisor and its workers) share one namespace."""
        existing: ToolHandler | None = getattr(engine, "_tool_handler", None)
        if existing is not None:
            existing.registry.extend(self.registry)
            return
        engine._tool_handler = self  # type: ignore[attr-defined]
        self.engine = engine
        engine.subscribe(REQUESTED, self._on_requested, group="aaa.tool.request")
        engine.subscribe(APPROVAL_GRANTED, self._on_granted, group="aaa.tool.granted")
        engine.subscribe(APPROVAL_DENIED, self._on_denied, group="aaa.tool.denied")
        engine.subscribe("aaa.run.completed", self._on_run_done)

    @staticmethod
    def _fail(e: Event, call_id: str, name: str, etype: str, msg: str) -> Event:
        return e.caused(
            FAILED,
            {"call_id": call_id, "name": name, "error": {"type": etype, "message": msg}},
            id=f"{e.id}:result",
        )

    async def _already_resolved(self, req: Event) -> bool:
        """O(1): a request's result always has the deterministic id `<request id>:result`."""
        assert self.engine
        return await self.engine.store.has(req.stream, f"{req.id}:result")

    async def _on_requested(self, e: Event) -> list[Event]:
        call_id, name = e.data["call_id"], e.data["name"]
        t = self.registry.get(name)
        if t is None:
            return [
                self._fail(
                    e,
                    call_id,
                    name,
                    "UnknownTool",
                    f"no tool named {name!r}; available: {self.registry.names()}",
                )
            ]
        if await self._already_resolved(e):
            return []
        if t.requires_approval:
            return [
                e.caused(
                    APPROVAL_REQUESTED,
                    {"call_id": call_id, "name": name, "arguments": e.data.get("arguments", {})},
                )
            ]
        self._launch(e, t)
        return []

    async def _find_request(self, e: Event, call_id: str) -> Event | None:
        assert self.engine
        for x in await self.engine.store.read(e.stream):
            if x.type == REQUESTED and x.data.get("call_id") == call_id:
                return await self.engine.hydrate(x)
        return None

    async def _on_granted(self, e: Event) -> list[Event]:
        req = await self._find_request(e, e.data["call_id"])
        t = self.registry.get(req.data["name"]) if req else None
        if req and t and not await self._already_resolved(req):
            self._launch(req, t)
        return []

    async def _on_denied(self, e: Event) -> list[Event]:
        req = await self._find_request(e, e.data["call_id"])
        if not req or await self._already_resolved(req):
            return []
        return [
            self._fail(
                req,
                req.data["call_id"],
                req.data["name"],
                "ApprovalDenied",
                e.data.get("reason") or "denied by reviewer",
            )
        ]

    async def _on_run_done(self, e: Event) -> list[Event]:
        for t in self._tasks.pop(e.run_id or "", set()):
            t.cancel()
        return []

    def _launch(self, req: Event, t: Tool) -> None:
        assert self.engine
        key = (req.stream, req.data["call_id"])
        if key in self._started:
            return
        self._started.add(key)
        task = self.engine.spawn(self._execute(req, t))
        bucket = self._tasks.setdefault(req.run_id or "", set())
        bucket.add(task)
        task.add_done_callback(bucket.discard)

    async def _execute(self, req: Event, t: Tool) -> None:
        assert self.engine
        call_id = req.data["call_id"]
        sem = self._sems.get(t.pool, self._sems["default"])
        try:
            async with sem:
                ctx = ToolContext(self.engine, req.run_id or "", call_id)
                result = await asyncio.wait_for(
                    t.invoke(req.data.get("arguments") or {}, ctx), t.timeout_s
                )
            out = req.caused(
                COMPLETED,
                {"call_id": call_id, "name": t.name, "result": result},
                id=f"{req.id}:result",
            )
        except ToolArgsError as exc:
            out = self._fail(req, call_id, t.name, "InvalidArguments", str(exc))
        except TimeoutError:
            out = self._fail(req, call_id, t.name, "Timeout", f"timed out after {t.timeout_s}s")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any tool failure is reported to the model
            out = self._fail(req, call_id, t.name, type(exc).__name__, str(exc))
        await self.engine.emit(out)
