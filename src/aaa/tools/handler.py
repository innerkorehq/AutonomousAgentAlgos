from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aaa.core.retry import RetryPolicy
from aaa.events import Event
from aaa.events.bus import defer_ack
from aaa.events.guard import ExecutionGuard, LocalGuard
from aaa.runtime import Engine
from aaa.tools.guard import Delivery, GuardConfig, LostLease
from aaa.tools.tool import Tool, ToolArgsError, ToolContext, ToolRegistry

log = logging.getLogger("aaa.tools")

REQUESTED = "aaa.tool.requested"
STARTED = "aaa.tool.started"
COMPLETED = "aaa.tool.completed"
FAILED = "aaa.tool.failed"
FENCED = "aaa.tool.fenced"
STALE_RESULT = "aaa.tool.stale_result"
APPROVAL_REQUESTED = "aaa.approval.requested"
APPROVAL_GRANTED = "aaa.approval.granted"
APPROVAL_DENIED = "aaa.approval.denied"


class _Outcome:
    """Whether a launched call's outcome reached the log (False => do not acknowledge the message)."""

    recorded = True


class ToolHandler:
    """Executes `aaa.tool.requested` {call_id, name, arguments}.

    - Durable: the bus message stays unacknowledged until the tool finishes (`defer_ack`), so a
      worker that dies mid-call has the request redelivered to another worker.
    - Guarded (see aaa.tools.guard): an exclusive execution lease with heartbeat, numbered attempt
      records (the fencing token), a delivery policy (at-least-once with a cap, or at-most-once),
      stale-result rejection, and an idempotency key handed to the tool. All configurable globally
      via `guard=` and per tool.
    - Tools with requires_approval wait for an external `aaa.approval.granted|denied` {call_id}
      event (e.g. RunHandle.signal), enforced outside the model's control.
    - Executions run as tracked background tasks in bounded pools, so a tool that waits on a
      child run (delegation) can never starve the pool its child needs.
    - Failures (bad args, timeout, exception, lost attempt) become `aaa.tool.failed` facts the
      model sees."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        pools: dict[str, int] | None = None,
        guard: GuardConfig | dict[str, Any] | None = None,
        result_retry: RetryPolicy | None = None,
    ) -> None:
        self.registry = registry
        # A tool that already RAN must not lose its result to a store outage: by default keep retrying
        # (with backoff) until it is recorded. A bounded policy makes the handler give up and leave
        # the bus message unacknowledged instead, so the call is redelivered after the outage.
        self.result_retry = result_retry or RetryPolicy.forever()
        self.guard = guard if isinstance(guard, GuardConfig) else GuardConfig(**(guard or {}))
        limits = {"default": 16, "delegate": 8, **(pools or {})}
        self._sems = {k: asyncio.Semaphore(v) for k, v in limits.items()}
        self._started: set[tuple[str, str]] = set()
        self._tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self._exec_guard: ExecutionGuard = LocalGuard()
        self.worker = "local"
        self.engine: Engine | None = None

    def config_for(self, t: Tool) -> GuardConfig:
        """Effective guard configuration of a tool: handler default + the tool's overrides."""
        return self.guard.merged(t.guard)

    def _validate(self) -> None:
        for name in self.registry.names():
            tool = self.registry.get(name)
            assert tool is not None
            self.config_for(tool)  # raises on inconsistent overrides, at startup rather than at 3am

    def attach(self, engine: Engine) -> None:
        """One tool executor per engine: later handlers merge their tools into the first, so
        independent algorithms (e.g. a supervisor and its workers) share one namespace."""
        existing: ToolHandler | None = getattr(engine, "_tool_handler", None)
        if existing is not None:
            existing.registry.extend(self.registry)
            existing._validate()
            return
        self._validate()
        engine._tool_handler = self  # type: ignore[attr-defined]
        self.engine = engine
        if hasattr(engine.bus, "execution_guard"):
            self._exec_guard = engine.bus.execution_guard()
        self.worker = str(getattr(engine.bus, "member", "local"))
        engine.subscribe(REQUESTED, self._on_requested, group="aaa.tool.request")
        engine.subscribe(APPROVAL_GRANTED, self._on_granted, group="aaa.tool.granted")
        engine.subscribe(APPROVAL_DENIED, self._on_denied, group="aaa.tool.denied")
        engine.subscribe("aaa.run.completed", self._on_run_done)

    # -- helpers ----------------------------------------------------------------------------
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
        eng = self.engine
        return await eng.retry(lambda: eng.store.has(req.stream, f"{req.id}:result"), "read")

    async def _attempts_started(self, req: Event, limit: int) -> int:
        """How many numbered attempts of this call were durably recorded before this one."""
        assert self.engine
        n = 0
        eng = self.engine
        while n < limit:
            nxt = f"{req.id}:attempt:{n + 1}"
            if not await eng.retry(lambda nxt=nxt: eng.store.has(req.stream, nxt), "read"):
                break
            n += 1
        return n

    # -- bus handlers -----------------------------------------------------------------------
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
        eng = self.engine
        for x in await eng.retry(lambda: eng.store.read(e.stream), "read"):
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

    # -- execution --------------------------------------------------------------------------
    def _launch(self, req: Event, t: Tool) -> None:
        """Start the call in the background WITHOUT acknowledging the bus message: it is
        acknowledged only when the call is finished (or abandoned), so a crash redelivers it."""
        assert self.engine
        key = (req.stream, req.data["call_id"])
        if key in self._started:
            return
        self._started.add(key)
        deferral = defer_ack()
        outcome = _Outcome()
        task = self.engine.spawn(self._run(req, t, outcome))

        def done(_: asyncio.Task[Any]) -> None:
            # A done-callback, not a `finally` in the coroutine: it also fires when the task is
            # cancelled before its first step ever ran (a run that completes right after the call
            # was launched). A `finally` would never execute then, and the bus message would stay
            # unacknowledged until the handover timeout.
            self._started.discard(key)  # a later redelivery may legitimately run a new attempt
            if outcome.recorded:
                deferral.release()

        task.add_done_callback(done)
        bucket = self._tasks.setdefault(req.run_id or "", set())
        bucket.add(task)
        task.add_done_callback(bucket.discard)

    async def _run(self, req: Event, t: Tool, outcome: _Outcome) -> None:
        assert self.engine
        try:
            cfg = self.config_for(t)
            if cfg.enabled:
                await self._run_guarded(req, t, cfg)
            else:
                await self._run_attempt(req, t, cfg, attempt=1, lease_key=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.engine.is_transient(exc):
                outcome.recorded = False  # leave the bus message unacknowledged: redelivered later
                log.error(
                    "tool call %s (%s): its outcome could not be recorded (%s: %s). The bus message "
                    "is left unacknowledged so the call is redelivered once the store is back.",
                    req.data["call_id"],
                    t.name,
                    type(exc).__name__,
                    exc,
                )
            else:  # a bug in the tool machinery: tell the run instead of stranding it
                log.exception("tool call %s (%s) failed internally", req.data["call_id"], t.name)
                try:
                    await self.engine.emit(
                        self._fail(
                            req,
                            req.data["call_id"],
                            t.name,
                            "InternalError",
                            f"{type(exc).__name__}: {exc}",
                        ),
                        retry=self.result_retry,
                    )
                except Exception:
                    outcome.recorded = False
                    log.exception(
                        "could not record the internal failure of %s", req.data["call_id"]
                    )

    def _idempotency_key(self, req: Event, t: Tool) -> str:
        if t.idempotency_key is not None:
            return str(t.idempotency_key(req.data.get("arguments") or {}))
        return f"{req.stream}/{req.data['call_id']}"

    async def _run_guarded(self, req: Event, t: Tool, cfg: GuardConfig) -> None:
        assert self.engine
        eng, lease = self.engine, f"{req.stream}/{req.id}"
        while True:  # 1. EXCLUSION: wait until no other live worker is running this call
            if await self._already_resolved(req) or eng.is_done(req.run_id or ""):
                return
            if await self._exec_guard.acquire(lease, cfg.lease_ttl_s):
                break
            await asyncio.sleep(cfg.wait_poll_s)
        try:
            if await self._already_resolved(req):  # finished while we waited
                return
            prior = await self._attempts_started(req, max(cfg.max_attempts, 1))
            if (
                prior
            ):  # a previous attempt began and did not finish: it died, stalled, or was cancelled
                if cfg.delivery is Delivery.AT_MOST_ONCE:
                    await eng.emit(
                        self._fail(
                            req,
                            req.data["call_id"],
                            t.name,
                            "OutcomeUnknown",
                            f"attempt {prior} of this call started on another worker and never reported a result, "
                            "so it may already have taken effect. It was NOT retried because this tool is "
                            "at_most_once. Verify the real-world state before trying again.",
                        )
                    )
                    return
                if prior >= cfg.max_attempts:
                    await eng.emit(
                        self._fail(
                            req,
                            req.data["call_id"],
                            t.name,
                            "ExecutionAbandoned",
                            f"gave up after {prior} attempts that each started and never finished "
                            "(the call may be crashing workers).",
                        )
                    )
                    return
            attempt = prior + 1
            marker = await eng.emit(  # 2. FENCING: the attempt number is recorded BEFORE running
                req.caused(
                    STARTED,
                    {
                        "call_id": req.data["call_id"],
                        "name": t.name,
                        "attempt": attempt,
                        "worker": self.worker,
                        "idempotency_key": self._idempotency_key(req, t),
                    },
                    id=f"{req.id}:attempt:{attempt}",
                )
            )
            if (
                marker is not None and marker.dup
            ):  # another worker claimed this number: not ours to run
                return
            await self._run_attempt(req, t, cfg, attempt=attempt, lease_key=lease)
        finally:
            if (
                not eng.crashed
            ):  # a real crash releases nothing; the lease must expire / the session die
                await self._exec_guard.release(lease)

    async def _heartbeat(self, lease: str, cfg: GuardConfig, lost: asyncio.Event) -> None:
        last_ok = time.monotonic()
        while True:
            await asyncio.sleep(cfg.beat)
            try:
                if not await self._exec_guard.renew(lease, cfg.lease_ttl_s):
                    lost.set()
                    return
                last_ok = time.monotonic()
            except Exception:  # noqa: BLE001 - a backend blip is not loss until the lease could have expired
                if time.monotonic() - last_ok >= cfg.lease_ttl_s:
                    lost.set()
                    return

    async def _invoke(self, req: Event, t: Tool, attempt: int) -> Event:
        """Run the tool; always returns the event to record (completed or failed)."""
        assert self.engine
        call_id = req.data["call_id"]
        sem = self._sems.get(t.pool, self._sems["default"])
        try:
            async with sem:
                ctx = ToolContext(
                    self.engine,
                    req.run_id or "",
                    call_id,
                    attempt=attempt,
                    fence=attempt,
                    idempotency_key=self._idempotency_key(req, t),
                )
                result = await asyncio.wait_for(
                    t.invoke(req.data.get("arguments") or {}, ctx), t.timeout_s
                )
            return req.caused(
                COMPLETED,
                {"call_id": call_id, "name": t.name, "result": result},
                id=f"{req.id}:result",
            )
        except ToolArgsError as exc:
            return self._fail(req, call_id, t.name, "InvalidArguments", str(exc))
        except TimeoutError:
            return self._fail(req, call_id, t.name, "Timeout", f"timed out after {t.timeout_s}s")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any tool failure is reported to the model
            return self._fail(req, call_id, t.name, type(exc).__name__, str(exc))

    async def _run_attempt(
        self, req: Event, t: Tool, cfg: GuardConfig, *, attempt: int, lease_key: str | None
    ) -> None:
        assert self.engine
        eng = self.engine
        if lease_key is None:  # guard disabled: just run it
            await eng.emit(await self._invoke(req, t, attempt), retry=self.result_retry)
            return
        lost = asyncio.Event()
        beat = asyncio.create_task(self._heartbeat(lease_key, cfg, lost))
        work = asyncio.create_task(self._invoke(req, t, attempt))
        lost_wait = asyncio.create_task(lost.wait())
        try:
            await asyncio.wait({work, lost_wait}, return_when=asyncio.FIRST_COMPLETED)
            if lost.is_set() and not work.done() and cfg.on_lost_lease is LostLease.CANCEL:
                work.cancel()  # we are no longer the only runner: stop, and record nothing
                await asyncio.gather(work, return_exceptions=True)
                await eng.emit(
                    req.caused(
                        FENCED,
                        {
                            "call_id": req.data["call_id"],
                            "attempt": attempt,
                            "worker": self.worker,
                            "reason": "execution lease lost; attempt cancelled",
                        },
                        id=f"{req.id}:fenced:{attempt}",
                    )
                )
                return
            out = await work
            await self._commit(req, out, cfg, attempt)  # lease stays alive while we record it
        finally:
            for task in (beat, lost_wait, work):
                if not task.done():
                    task.cancel()
            await asyncio.gather(beat, lost_wait, work, return_exceptions=True)

    async def _commit(self, req: Event, out: Event, cfg: GuardConfig, attempt: int) -> None:
        eng = self.engine
        assert eng
        # 3. FENCED WRITE: a newer attempt supersedes this one, so its result must not be recorded
        if cfg.fence_results and await eng.retry(
            lambda: eng.store.has(req.stream, f"{req.id}:attempt:{attempt + 1}"), "read"
        ):
            await eng.emit(
                req.caused(
                    STALE_RESULT,
                    {
                        "call_id": req.data["call_id"],
                        "attempt": attempt,
                        "superseded_by": attempt + 1,
                        "worker": self.worker,
                        "outcome": out.type,
                        "data": out.data,
                    },
                    id=f"{req.id}:stale:{attempt}",
                ),
                retry=self.result_retry,
            )
            return
        await eng.emit(out, retry=self.result_retry)
