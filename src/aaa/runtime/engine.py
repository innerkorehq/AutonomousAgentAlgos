from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, TypeVar

from pydantic import BaseModel, Field

from aaa.core.ids import new_id
from aaa.core.outcome import Outcome, OutcomeKind
from aaa.data.blob import BlobStore
from aaa.events import (
    Event,
    EventBus,
    EventStore,
    Handler,
    InMemoryEventStore,
    InProcessBus,
    Projection,
    SchemaRegistry,
)
from aaa.runtime.budget import Budget, Ledger

log = logging.getLogger("aaa.engine")
S = TypeVar("S")

RUN_STARTED = "aaa.run.started"
RUN_COMPLETED = "aaa.run.completed"
RUN_CANCEL = "aaa.run.cancel"
CHILD_STARTED = "aaa.child.started"
CHILD_COMPLETED = "aaa.child.completed"
CLAIM_KEY = "$claim"
# Small, hot fields that stay on the stored event when its payload is claim-checked into a blob,
# so folds that only need them (spend, call ids, tags) never have to fetch the blob.
CLAIM_KEEP = ("usage", "cost", "tag", "call_id", "name")


class RunMeta(BaseModel):
    """Everything the engine needs to know about a run, as a pure fold of its log: spend,
    whether/how it finished, and which child runs are still open. Because it is derived, any
    process can rebuild it (from a snapshot + tail) and no process-local state is authoritative."""

    ledger: Ledger = Field(default_factory=Ledger)
    completed: dict[str, Any] | None = None
    children: dict[str, str | None] = Field(default_factory=dict)  # open child run id -> tag


def _apply_meta(m: RunMeta, e: Event) -> RunMeta:
    m.ledger = m.ledger.fold(e)
    if e.type == RUN_COMPLETED and m.completed is None:
        m.completed = e.data
    elif e.type == CHILD_STARTED:
        m.children[e.data["child_run_id"]] = e.data.get("tag")
    elif e.type == CHILD_COMPLETED:
        m.children.pop(e.data["child_run_id"], None)
    return m


RUNMETA = Projection(RunMeta, _apply_meta, name="aaa.runmeta", version=1, state_type=RunMeta)


@dataclass
class _Run:
    budget: Budget
    future: asyncio.Future[Outcome[Any]]
    trace_id: str
    algo: Any = None
    parent: str | None = None
    tag: str | None = None
    meta: RunMeta = field(default_factory=RunMeta)
    synced: int = -1  # last event seq folded into `meta`
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    done: bool = False
    tripped: bool = False
    timer: asyncio.Task[None] | None = None

    @property
    def ledger(self) -> Ledger:
        return self.meta.ledger


class Ctx:
    """Per-event context handed to algorithms. Algorithms hold no run state of their own:
    everything is derived from the run's event stream, so runs resume, replay and migrate
    between processes."""

    def __init__(self, engine: Engine, run_id: str | None) -> None:
        self.engine, self.run_id = engine, run_id

    async def history(self, from_seq: int = 0) -> list[Event]:
        """The run's events with any claim-checked payloads restored."""
        assert self.run_id
        raw = await self.engine.store.read(self.run_id, from_seq)
        return [await self.engine.hydrate(e) for e in raw]

    async def project(self, projection: Projection[S]) -> S:
        """Current state of `projection` for this run: latest snapshot + tail (O(tail))."""
        assert self.run_id
        return await self.engine.project(projection, self.run_id)

    async def started(self) -> Event:
        """The run's `aaa.run.started` event (input, budget, parent)."""
        assert self.run_id
        return await self.engine._started_of(self.run_id)

    def ledger(self) -> Ledger:
        run = self.engine._runs.get(self.run_id or "")
        return run.ledger if run else Ledger()


class Algorithm(Protocol):
    name: str

    def subscriptions(self) -> list[str]:
        """Event-type glob patterns this algorithm reacts to."""
        ...

    async def on_event(self, event: Event, ctx: Ctx) -> list[Event]: ...


class RunHandle:
    def __init__(self, engine: Engine, run_id: str) -> None:
        self._engine, self.run_id = engine, run_id

    async def result(self) -> Outcome[Any]:
        """Outcome of the run, wherever (whichever process) it was hosted or finished."""
        run = self._engine._runs.get(self.run_id)
        if run is not None and run.future.done():
            return run.future.result()
        waiters: list[asyncio.Future[Outcome[Any]]] = []
        if run is not None:
            waiters.append(asyncio.ensure_future(asyncio.shield(run.future)))
        waiters.append(asyncio.ensure_future(self._engine._await_completed(self.run_id)))
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            return next(iter(done)).result()
        finally:
            for w in waiters:
                if not w.done():
                    w.cancel()

    async def events(self, *, hydrate: bool = False) -> AsyncIterator[Event]:
        """Replay the run's stream from the start, then tail until the run completes."""
        async for e in self._engine.store.subscribe(self.run_id):
            yield await self._engine.hydrate(e) if hydrate else e
            if e.type == RUN_COMPLETED:
                return

    async def cancel(self, reason: str = "cancelled") -> None:
        await self.signal(RUN_CANCEL, {"reason": reason})

    async def signal(self, type: str, data: dict[str, Any] | None = None) -> Event | None:
        """Inject an external event (human approval, webhook, ...) into the run."""
        return await self._engine.emit(
            Event(
                type=type,
                data=data or {},
                stream=self.run_id,
                run_id=self.run_id,
                trace_id=self._engine._trace(self.run_id),
                source="external",
            )
        )


class Engine:
    """Owns store + bus + schemas (+ optional blob store). Everything persisted is appended to
    the log BEFORE it is delivered, so the log is the source of truth and delivery replayable.

    Cluster-ready: per-run state in `_runs` is only a cache of a fold of the log, rebuilt on
    demand ("adopted") by whichever process the bus routes a run's events to. With a
    distributed bus, several Engine processes sharing one store cooperate on the same runs."""

    def __init__(
        self,
        store: EventStore | None = None,
        bus: EventBus | None = None,
        schemas: SchemaRegistry | None = None,
        default_budget: Budget | None = None,
        blobs: BlobStore | None = None,
        claim_threshold: int = 16_384,
        sweep_interval_s: float | None = None,
    ) -> None:
        self.store: EventStore = store or InMemoryEventStore()
        self.bus = bus or InProcessBus()
        self.distributed: bool = getattr(self.bus, "distributed", False)
        self.schemas = schemas or SchemaRegistry()
        self.default_budget = default_budget or Budget()
        self.blobs, self.claim_threshold = blobs, claim_threshold
        self.sweep_interval_s = sweep_interval_s
        self._runs: dict[str, _Run] = {}
        self._algos: dict[str, Algorithm] = {}
        self._registered: set[int] = set()
        self._adopt_locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._sweeper: asyncio.Task[None] | None = None
        self.last_recovery: Any = None
        self.crashed = False  # set by abort(): models a process death
        self.subscribe(RUN_CANCEL, self._on_cancel, group="aaa.engine.cancel")

    async def __aenter__(self) -> Self:
        await self.bus.__aenter__()
        if self.sweep_interval_s:
            self._sweeper = asyncio.create_task(self._sweep_loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._sweeper:
            self._sweeper.cancel()
        await self.bus.__aexit__(*exc)
        for run in self._runs.values():
            if run.timer:
                run.timer.cancel()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def abort(self) -> None:
        """Hard stop without draining or completing anything — what a crash looks like to the
        log. Runs stay open in the store; another process (or `recover()`) continues them.
        Sets `crashed` so cleanup code (lease release, ...) behaves as if the process vanished."""
        self.crashed = True
        if self._sweeper:
            self._sweeper.cancel()
        await self.bus.abort()  # first: nothing may be acknowledged any more
        for run in self._runs.values():
            if run.timer:
                run.timer.cancel()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Run background work (e.g. a long tool call) tracked for clean shutdown."""
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    def handle(self, run_id: str) -> RunHandle:
        """A handle to any run by id (started here, in another process, or before a restart)."""
        return RunHandle(self, run_id)

    def _trace(self, run_id: str) -> str:
        run = self._runs.get(run_id)
        return run.trace_id if run else run_id

    async def project(self, projection: Projection[S], run_id: str) -> S:
        return (await projection.load(self.store, run_id, hydrate=self.hydrate)).state

    async def _started_of(self, run_id: str) -> Event:
        first = await self.store.read(run_id, 0, 1)
        if not first or first[0].type != RUN_STARTED:
            raise KeyError(f"no run {run_id!r}")
        return await self.hydrate(first[0])

    # -- blob claim-check -------------------------------------------------------------------
    async def _claim(self, event: Event) -> Event:
        if self.blobs is None or event.type == RUN_COMPLETED:
            return event
        raw = json.dumps(event.data, default=str).encode()
        if len(raw) <= self.claim_threshold:
            return event
        ref = await self.blobs.put(raw, "application/json")
        stub = {CLAIM_KEY: True, **{k: event.data[k] for k in CLAIM_KEEP if k in event.data}}
        return event.model_copy(update={"data": stub, "blobs": (*event.blobs, ref)})

    async def hydrate(self, event: Event) -> Event:
        """Restore a claim-checked payload (no-op for ordinary events)."""
        if self.blobs is None or event.data.get(CLAIM_KEY) is not True or not event.blobs:
            return event
        data = json.loads(await self.blobs.get(event.blobs[-1]))
        return event.model_copy(update={"data": data})

    # -- wiring -----------------------------------------------------------------------------
    def subscribe(
        self, pattern: str, handler: Handler, *, group: str | None = None, **kw: Any
    ) -> None:
        """Subscribe a handler whose returned events are persisted+published by the engine.
        Handlers receive hydrated events. Handler exceptions fail the run (as an event), never
        the bus. Events for runs that already finished are not delivered to handlers.

        `group` names a logical consumer. With a distributed bus, processes subscribing with the
        same group compete (each event is handled by exactly one of them, in order per run);
        `group=None` means broadcast to every process. Ignored by the in-process bus."""

        async def wrapped(e: Event) -> None:
            if e.run_id and self.is_done(e.run_id) and not e.type.startswith("aaa.run."):
                return
            try:
                e = await self.hydrate(e)
                for f in await handler(e) or ():
                    await self.emit(f)
            except Exception as exc:
                log.exception("handler failed for %s", e.type)
                if e.run_id:
                    await self.complete(
                        e.run_id, OutcomeKind.FAILED, reason=f"{type(exc).__name__}: {exc}"
                    )
            return

        if self.distributed:
            kw["group"] = group
        self.bus.subscribe(pattern, wrapped, **kw)

    def register(self, algo: Algorithm) -> None:
        """Subscribe an algorithm. It only ever sees events of runs it was started for, so
        several algorithms (and nested child runs) coexist on one bus. With a distributed bus,
        algorithm names must be unique and identical across processes."""
        if id(algo) in self._registered:
            return
        self._registered.add(id(algo))
        self._algos.setdefault(algo.name, algo)

        async def h(e: Event) -> list[Event]:
            rid = e.run_id or ""
            run = self._runs.get(rid) or await self._adopt(rid, algo)
            if run is None or run.algo is not algo:
                return []
            await self._sync(rid, run)  # absorb events other processes appended
            if run.done and not e.type.startswith("aaa.run."):
                return []
            return await algo.on_event(e, Ctx(self, rid))

        # ONE subscription (one queue, one worker): the algorithm sees its events strictly in
        # log order, so state folded from history is always consistent with what it handles.
        patterns = algo.subscriptions()
        self.subscribe(
            "*",
            h,
            group=f"aaa.algo.{algo.name}",
            predicate=lambda e: any(fnmatch.fnmatchcase(e.type, p) for p in patterns),
        )

    # -- run state: a cache of a fold of the log --------------------------------------------
    def is_done(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        return bool(run and run.done)

    async def _adopt(self, rid: str, algo: Algorithm | None) -> _Run | None:
        """Build local run state from the log (snapshot + tail). Returns None if `rid` is not a
        run, or belongs to a different algorithm than `algo`."""
        if run := self._runs.get(rid):
            return run
        async with self._adopt_locks.setdefault(rid, asyncio.Lock()):
            if run := self._runs.get(rid):
                return run
            first = await self.store.read(rid, 0, 1) if rid else []
            if not first or first[0].type != RUN_STARTED:
                return None
            started = await self.hydrate(first[0])
            if algo is not None and started.data.get("algo") != algo.name:
                return None
            loaded = await RUNMETA.load(self.store, rid, hydrate=self.hydrate)
            run = _Run(
                Budget(**started.data.get("budget", {})),
                asyncio.get_running_loop().create_future(),
                trace_id=started.trace_id or rid,
                algo=algo,
                parent=started.data.get("parent"),
                tag=started.data.get("run_tag"),
                meta=loaded.state,
                synced=loaded.seq,
            )
            if run.meta.completed is not None:
                self._mark_done(run, run.meta.completed)
            elif run.budget.deadline_s is not None:
                elapsed = time.time() - started.time.timestamp()
                run.timer = asyncio.create_task(
                    self._deadline(
                        rid, max(0.0, run.budget.deadline_s - elapsed), run.budget.deadline_s
                    )
                )
            self._runs[rid] = run
            return run

    @staticmethod
    def _mark_done(run: _Run, outcome: dict[str, Any]) -> None:
        run.done = True
        if run.timer:
            run.timer.cancel()
        if not run.future.done():
            run.future.set_result(Outcome(**outcome))

    async def _absorb(self, rid: str, run: _Run, stored: Event | None = None) -> None:
        """Fold events appended since `run.synced` into the run's meta (cheap if nothing new)."""
        async with run.lock:
            if stored is not None and stored.seq == run.synced + 1:
                _apply_meta(run.meta, stored)
                run.synced = stored.seq
            elif await self.store.last_seq(rid) > run.synced:
                for e in await self.store.read(rid, run.synced + 1):
                    _apply_meta(run.meta, e)
                    run.synced = e.seq
            if run.meta.completed is not None and not run.future.done():
                self._mark_done(run, run.meta.completed)

    async def _sync(self, rid: str, run: _Run) -> None:
        await self._absorb(rid, run)
        await self._check_budget(rid, run)

    async def _check_budget(self, rid: str, run: _Run) -> None:
        if run.done or run.tripped or not (why := run.ledger.exceeded(run.budget)):
            return
        run.tripped = True
        await self.emit(
            Event(
                id=f"{rid}:budget",
                type="aaa.budget.exceeded",
                stream=rid,
                run_id=rid,
                trace_id=run.trace_id,
                data={"reason": why},
            )
        )
        await self.complete(rid, OutcomeKind.BUDGET_EXCEEDED, reason=why)

    # -- emit -------------------------------------------------------------------------------
    async def emit(self, event: Event) -> Event | None:
        """Validate -> claim-check -> append (assign seq) -> account budget -> deliver.
        Returns the stored event, or None if dropped (duplicate terminal event)."""
        rid = event.run_id or ""
        run = self._runs.get(rid)
        if event.type == RUN_COMPLETED and run is not None:
            if run.done:
                return None
            run.done = True  # no await between check and set: atomic under asyncio
        event = self.schemas.validate(event)
        stored = (await self.store.append(event.stream, [await self._claim(event)]))[0]
        if stored.dup:  # already on the log (deterministic id): effect applied once, not twice
            if stored.type == RUN_COMPLETED:
                await self._after_completed(rid, stored)
            return stored
        if run is not None:
            await self._absorb(rid, run, stored)
        await self.bus.publish(stored)
        if stored.type == RUN_COMPLETED:
            await self._after_completed(rid, stored)
        elif run is not None:
            await self._check_budget(rid, run)
        return stored

    async def _after_completed(self, rid: str, done: Event) -> None:
        """Idempotent consequences of a run finishing, computed from the log alone: settle local
        state, cancel still-open children, announce the result to the parent."""
        run = self._runs.get(rid)
        if run is not None:
            self._mark_done(run, done.data)
        meta = (await RUNMETA.load(self.store, rid, hydrate=self.hydrate)).state
        for cid in list(meta.children):  # a finished parent leaves no orphan work
            await self.complete(cid, OutcomeKind.CANCELLED, reason="parent run finished")
        try:
            started = await self._started_of(rid)
        except KeyError:
            return
        if parent := started.data.get("parent"):
            await self.emit(
                self._child_completed(
                    parent, done, started.trace_id or rid, started.data.get("run_tag"), meta.ledger
                )
            )

    @staticmethod
    def _child_completed(
        parent_id: str, done: Event, trace_id: str, tag: str | None, ledger: Ledger
    ) -> Event:
        return Event(
            id=f"child-{done.run_id}",
            type=CHILD_COMPLETED,
            stream=parent_id,
            run_id=parent_id,
            trace_id=trace_id,
            source="engine",
            data={
                "child_run_id": done.run_id,
                "tag": tag,
                "outcome": done.data,
                "ledger": ledger.model_dump(),
            },
        )

    async def complete(
        self, run_id: str, kind: OutcomeKind, value: Any = None, reason: str | None = None
    ) -> None:
        await self.emit(
            Event(
                id=f"{run_id}:completed",
                type=RUN_COMPLETED,
                stream=run_id,
                run_id=run_id,
                trace_id=self._trace(run_id),
                data={"kind": kind.value, "value": value, "reason": reason},
            )
        )

    async def _on_cancel(self, e: Event) -> None:
        if e.run_id:
            await self.complete(e.run_id, OutcomeKind.CANCELLED, reason=e.data.get("reason"))

    async def _deadline(self, run_id: str, seconds: float, total: float | None = None) -> None:
        await asyncio.sleep(seconds)
        await self.emit(
            Event(
                type="aaa.timer.fired",
                stream=run_id,
                run_id=run_id,
                trace_id=self._trace(run_id),
                data={"kind": "deadline"},
            )
        )
        await self.complete(
            run_id, OutcomeKind.TIMEOUT, reason=f"deadline {total or seconds}s exceeded"
        )

    async def _await_completed(self, run_id: str) -> Outcome[Any]:
        async for e in self.store.subscribe(run_id):
            if e.type == RUN_COMPLETED:
                return Outcome(**e.data)
        raise AssertionError("unreachable")

    # -- runs -------------------------------------------------------------------------------
    async def start(
        self,
        algo: Algorithm,
        input: dict[str, Any],
        *,
        budget: Budget | None = None,
        run_id: str | None = None,
        parent: str | None = None,
        tag: str | None = None,
    ) -> RunHandle:
        """Start a run (idempotent for a given `run_id`). With `parent`, the child's spend rolls
        up into the parent's budget, its completion is announced on the parent stream, and it is
        cancelled if the parent finishes first."""
        self.register(algo)
        rid = run_id or new_id()
        if rid in self._runs:
            return RunHandle(self, rid)
        if await self.store.last_seq(rid) >= 0:  # exists already: finished, or open elsewhere
            await self._adopt(rid, None)
            return RunHandle(self, rid)
        parent_run = self._runs.get(parent) if parent else None
        run = _Run(
            budget or self.default_budget,
            asyncio.get_running_loop().create_future(),
            trace_id=parent_run.trace_id if parent_run else rid,
            algo=algo,
            parent=parent,
            tag=tag,
        )
        self._runs[rid] = run
        if parent:
            await self.emit(
                Event(
                    id=f"{rid}:childstarted",
                    type=CHILD_STARTED,
                    stream=parent,
                    run_id=parent,
                    trace_id=run.trace_id,
                    source="engine",
                    data={"child_run_id": rid, "tag": tag},
                )
            )
        if run.budget.deadline_s is not None:
            run.timer = asyncio.create_task(self._deadline(rid, run.budget.deadline_s))
        await self.emit(
            Event(
                id=f"{rid}:started",
                type=RUN_STARTED,
                stream=rid,
                run_id=rid,
                trace_id=run.trace_id,
                source=algo.name,
                data={
                    "algo": algo.name,
                    "input": input,
                    "parent": parent,
                    "run_tag": tag,
                    "budget": run.budget.model_dump(),
                },
            )
        )
        return RunHandle(self, rid)

    async def recover(self, algos: list[Algorithm]) -> dict[str, RunHandle]:
        """Continue every run left open in the store (single-process restart). See recovery.py."""
        from aaa.runtime.recovery import recover

        return await recover(self, algos)

    async def sweep(self) -> Any:
        """Stateless, idempotent cluster maintenance: expire deadlines of dead processes, cancel
        orphaned children, repair lost child announcements. Safe to run anywhere, any time."""
        from aaa.runtime.recovery import maintain

        return await maintain(self)

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval_s or 5)
            try:
                await self.sweep()
            except Exception:
                log.exception("sweep failed")

    async def run(self, algo: Algorithm, input: dict[str, Any], **kw: Any) -> Outcome[Any]:
        """Facade: start and await the outcome (no event knowledge needed)."""
        return await (await self.start(algo, input, **kw)).result()
