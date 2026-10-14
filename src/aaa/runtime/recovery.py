"""Recovery and cluster maintenance.

Two separate jobs:

* `maintain(engine)` — stateless, idempotent repairs that are safe in ANY process at ANY time:
  expire deadlines whose owner died, cancel orphaned children, re-announce lost child results.
  (Run periodically via `Engine(sweep_interval_s=...)` in a cluster.)

* `recover(engine, algos)` — single-process restart with an in-process bus: re-adopt every open
  run, then redeliver events nothing reacted to. With a distributed bus this is unnecessary:
  the bus log + per-group cursors redeliver unacknowledged events by themselves.

Why redelivery is safe: algorithms are level-triggered (they decide from folded state, not from
the event alone) and every derived event has a deterministic id, so a repeated reaction is
absorbed by the store's idempotent append. Capability work in flight at the crash (an LLM or tool
call with no result) is simply requested again: tools are at-least-once."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aaa.core.outcome import OutcomeKind

if TYPE_CHECKING:
    from aaa.runtime.engine import Algorithm, Engine, RunHandle

log = logging.getLogger("aaa.recovery")

# Events caused by a request that do NOT mean the request was handled: partial progress only.
# (A tool that STARTED but never finished, or an LLM call that streamed some deltas and died, must
# still be redelivered.)
PROGRESS_ONLY = frozenset({"aaa.tool.started", "aaa.llm.delta"})


@dataclass
class RecoveryReport:
    recovered: list[str] = field(default_factory=list)
    unknown_algo: list[str] = field(default_factory=list)
    orphans_cancelled: list[str] = field(default_factory=list)
    timed_out: list[str] = field(default_factory=list)
    announced: list[str] = field(default_factory=list)
    redelivered: int = 0


async def maintain(engine: Engine) -> RecoveryReport:
    from aaa.runtime.engine import RUNMETA

    report = RecoveryReport()
    index = await engine.store.run_index()
    open_ = {r.stream: r for r in index if r.completed is None}

    for sid, info in open_.items():  # 1. deadlines (a dead owner can't fire its own timer)
        started = await engine.hydrate(info.started)
        d = (started.data.get("budget") or {}).get("deadline_s")
        if d is not None and time.time() - started.time.timestamp() >= d:
            await engine.complete(sid, OutcomeKind.TIMEOUT, reason=f"deadline {d}s exceeded")
            report.timed_out.append(sid)

    for sid, info in open_.items():  # 2. open child whose parent is finished/missing: cancel
        parent = info.started.data.get("parent")
        if parent and (parent not in open_ or parent in report.timed_out):
            await engine.complete(sid, OutcomeKind.CANCELLED, reason="parent run finished")
            report.orphans_cancelled.append(sid)

    for r in index:  # 3. finished child whose result never reached its still-open parent
        parent = r.started.data.get("parent")
        if r.completed is None or not parent or parent not in open_:
            continue
        meta = (await RUNMETA.load(engine.store, parent, hydrate=engine.hydrate)).state
        if r.stream in meta.children:
            child_meta = (await RUNMETA.load(engine.store, r.stream, hydrate=engine.hydrate)).state
            await engine.emit(
                engine._child_completed(
                    parent,
                    r.completed,
                    r.started.trace_id or r.stream,
                    r.started.data.get("run_tag"),
                    child_meta.ledger,
                )
            )
            report.announced.append(r.stream)
    return report


async def recover(engine: Engine, algos: list[Algorithm]) -> dict[str, RunHandle]:
    from aaa.runtime.engine import RUN_COMPLETED, RunHandle

    by_name: dict[str, Algorithm] = {}
    for a in algos:
        if a.name in by_name and by_name[a.name] is not a:
            raise ValueError(
                f"ambiguous algorithm name {a.name!r}: give recoverable algorithms unique names"
            )
        by_name[a.name] = a

    index = await engine.store.run_index()
    handles: dict[str, RunHandle] = {}
    unknown: list[str] = []
    for r in index:
        if r.completed is not None or r.stream in engine._runs:
            continue
        started = await engine.hydrate(r.started)
        algo = by_name.get(started.data["algo"])
        if algo is None:
            unknown.append(r.stream)
            log.warning(
                "cannot recover run %s: no algorithm named %r supplied",
                r.stream,
                started.data["algo"],
            )
            continue
        engine.register(algo)
        if await engine._adopt(r.stream, algo) is not None:
            handles[r.stream] = RunHandle(engine, r.stream)

    report = await maintain(engine)
    report.recovered, report.unknown_algo = list(handles), unknown

    for sid in handles:  # redeliver events nothing reacted to (idempotent by construction)
        if engine.is_done(sid):
            continue
        events = await engine.store.read(sid)
        reacted = {e.causation_id for e in events if e.causation_id and e.type not in PROGRESS_ONLY}
        for e in events:
            if e.type != RUN_COMPLETED and e.id not in reacted:
                await engine.bus.publish(e)
                report.redelivered += 1
    log.info("recovered %d runs, redelivered %d events", len(handles), report.redelivered)
    engine.last_recovery = report
    return handles
