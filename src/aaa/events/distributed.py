"""Distributed event bus: a partitioned, durable log with consumer groups.

Model (Kafka-like, but the log is the store's own outbox, written in the SAME atomic step as the
event itself, so "stored but never published" cannot happen):

* An event goes to partition `crc32(stream) % P`. All events of one run share a partition, so
  they are consumed in order.
* A *group* is a logical consumer ("the LLM handler", "algorithm tool_loop"). Within a group each
  partition is owned by exactly one process at a time (a lease), which consumes it in order and
  records a cursor. Processes joining/leaving rebalance partitions (fair share).
* Delivery is at-least-once: a crash redelivers events after the last committed cursor. This is
  safe because handlers/algorithms are level-triggered and derived events have deterministic ids.
* `group=None` is a broadcast subscription: every process sees every event (from "now").

A lease holder that stalls past its lease can briefly overlap with the new owner (no fencing
token is enforced); idempotent appends absorb the duplicate work."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import math
import random
import time
import zlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, Self, runtime_checkable

from aaa.core.errors import Redeliver
from aaa.core.ids import new_id
from aaa.events.bus import DeliverySlot, Handler, current_delivery
from aaa.events.event import Event
from aaa.events.guard import ExecutionGuard, LeaseGuard

log = logging.getLogger("aaa.dbus")


def partition_of(stream: str, partitions: int) -> int:
    """Stable across processes and Python versions (unlike hash())."""
    return zlib.crc32(stream.encode()) % partitions


@runtime_checkable
class BusBackend(Protocol):
    partitions: int

    async def read(
        self, partition: int, after: int, limit: int, wait_s: float
    ) -> list[tuple[int, Event]]:
        """Entries with id > `after`, oldest first; block up to `wait_s` for new ones."""
        ...

    async def tail(self, partition: int) -> int:
        """Highest entry id in the partition (0 if empty)."""
        ...

    async def get_cursor(self, group: str, partition: int) -> int: ...
    async def set_cursor(self, group: str, partition: int, cursor: int) -> None: ...

    async def try_lease(self, group: str, partition: int, member: str, ttl_s: float) -> bool:
        """Acquire the lease, or renew it if `member` already holds it."""
        ...

    async def release_lease(self, group: str, partition: int, member: str) -> None: ...
    async def heartbeat(self, group: str, member: str, ttl_s: float) -> None: ...
    async def members(self, group: str) -> list[str]: ...
    async def leave(self, group: str, member: str) -> None: ...
    async def abandon(self) -> None:
        """Crash simulation hook: drop connections WITHOUT releasing anything."""
        ...

    async def close(self) -> None: ...


class MemoryBusBackend:
    """In-process reference backend (shared object = shared 'cluster'). Used by tests and as the
    executable specification the Postgres/Redis backends must match."""

    def __init__(self, partitions: int = 8, clock: Callable[[], float] = time.monotonic) -> None:
        self.partitions, self._clock = partitions, clock
        self._log: list[list[tuple[int, Event]]] = [[] for _ in range(partitions)]
        self._cursors: dict[tuple[str, int], int] = {}
        self._leases: dict[tuple[str, int], tuple[str, float]] = {}
        self._members: dict[str, dict[str, float]] = {}
        self._conds: dict[int, asyncio.Condition] = {}

    def _cond(self, p: int) -> asyncio.Condition:
        return self._conds.setdefault(p, asyncio.Condition())

    async def enqueue(self, event: Event) -> int:
        p = partition_of(event.stream, self.partitions)
        async with self._cond(p):
            eid = len(self._log[p]) + 1
            self._log[p].append((eid, event))
            self._cond(p).notify_all()
            return eid

    async def read(
        self, partition: int, after: int, limit: int, wait_s: float
    ) -> list[tuple[int, Event]]:
        cond = self._cond(partition)
        async with cond:
            if len(self._log[partition]) <= after and wait_s > 0:
                try:
                    await asyncio.wait_for(
                        cond.wait_for(lambda: len(self._log[partition]) > after), wait_s
                    )
                except TimeoutError:
                    return []
            return self._log[partition][after : after + limit]

    async def tail(self, partition: int) -> int:
        return len(self._log[partition])

    async def get_cursor(self, group: str, partition: int) -> int:
        return self._cursors.get((group, partition), 0)

    async def set_cursor(self, group: str, partition: int, cursor: int) -> None:
        self._cursors[(group, partition)] = max(cursor, self._cursors.get((group, partition), 0))

    async def try_lease(self, group: str, partition: int, member: str, ttl_s: float) -> bool:
        now = self._clock()
        holder = self._leases.get((group, partition))
        if holder is None or holder[0] == member or holder[1] <= now:
            self._leases[(group, partition)] = (member, now + ttl_s)
            return True
        return False

    async def release_lease(self, group: str, partition: int, member: str) -> None:
        if (h := self._leases.get((group, partition))) and h[0] == member:
            del self._leases[(group, partition)]

    async def heartbeat(self, group: str, member: str, ttl_s: float) -> None:
        self._members.setdefault(group, {})[member] = self._clock() + ttl_s

    async def members(self, group: str) -> list[str]:
        now = self._clock()
        return sorted(m for m, exp in self._members.get(group, {}).items() if exp > now)

    async def leave(self, group: str, member: str) -> None:
        self._members.get(group, {}).pop(member, None)

    async def abandon(self) -> None:  # leases/membership simply expire
        return None

    async def close(self) -> None:
        return None


@dataclass
class _Sub:
    pattern: str
    handler: Handler
    predicate: Callable[[Event], bool] | None
    concurrency: int
    group: str | None

    def matches(self, e: Event) -> bool:
        return fnmatch.fnmatchcase(e.type, self.pattern) and (
            self.predicate is None or self.predicate(e)
        )


@dataclass
class _Entry:
    """One delivered bus entry. `ok=False` means it was abandoned before completing (shutdown, lost
    lease): it must NEVER be acknowledged, so it is redelivered to whoever owns the partition next."""

    eid: int
    task: asyncio.Task[None] | None = None
    ok: bool = True


@dataclass
class _Consumer:
    group: str
    partition: int
    task: asyncio.Task[None] | None = None
    owned: bool = True
    pending: deque[_Entry] = field(default_factory=deque)
    acked: int = 0
    read_upto: int = 0


class DistributedBus:
    """EventBus over a BusBackend. `publish` is a no-op: the store already enqueued the event
    atomically with appending it (see PostgresEventStore / RedisEventStore / MemoryBusBackend)."""

    distributed = True

    def __init__(
        self,
        backend: BusBackend,
        *,
        member: str | None = None,
        batch: int = 64,
        poll_s: float = 0.5,
        lease_ttl_s: float = 6.0,
        tick_s: float = 1.0,
        handover_timeout_s: float = 30.0,
        redeliver_base_s: float = 0.2,
        redeliver_max_s: float = 5.0,
    ) -> None:
        self.handover_timeout_s = handover_timeout_s
        self.redeliver_base_s, self.redeliver_max_s = redeliver_base_s, redeliver_max_s
        self.backend, self.member = backend, member or new_id()
        self.batch, self.poll_s, self.lease_ttl_s, self.tick_s = batch, poll_s, lease_ttl_s, tick_s
        self._subs: list[_Sub] = []
        self._consumers: dict[tuple[str, int], _Consumer] = {}
        self._broadcast: list[asyncio.Task[None]] = []
        self._manager: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._running = False
        self._warned = False

    # -- API shared with InProcessBus ------------------------------------------------------
    def subscribe(
        self,
        pattern: str,
        handler: Handler,
        *,
        group: str | None = None,
        concurrency: int = 1,
        predicate: Callable[[Event], bool] | None = None,
    ) -> None:
        if group is not None and any(s.group == group for s in self._subs):
            raise ValueError(
                f"consumer group {group!r} is already subscribed in this process; with a distributed "
                "bus each group is one logical consumer (algorithm names must be unique)"
            )
        self._subs.append(_Sub(pattern, handler, predicate, concurrency, group))
        if self._running:
            if group is None:
                self._start_broadcast_for(self._subs[-1])
            self._wake.set()

    def execution_guard(self) -> ExecutionGuard:
        """Cross-process mutual exclusion for side-effecting work (same lease semantics as partitions)."""
        return LeaseGuard(self.backend, self.member)

    async def publish(self, event: Event) -> None:
        """No-op by design (the store enqueues atomically with the append)."""
        return

    async def __aenter__(self) -> Self:
        self._running = True
        self._manager = asyncio.create_task(self._manage())
        for s in self._subs:
            if s.group is None:
                self._start_broadcast_for(s)
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Graceful: stop consuming, let in-flight handlers finish, commit cursors, release leases."""
        self._running = False
        if self._manager:
            self._manager.cancel()
            await asyncio.gather(self._manager, return_exceptions=True)
        for t in self._broadcast:
            t.cancel()
        await asyncio.gather(*self._broadcast, return_exceptions=True)
        for c in list(self._consumers.values()):
            await self._stop_consumer(c, release=True)
        for g in {s.group for s in self._subs if s.group}:
            await self.backend.leave(g, self.member)

    async def abort(self) -> None:
        """Crash: cancel everything, release NOTHING (leases/membership must expire)."""
        self._running = False
        tasks = [t for t in [self._manager, *self._broadcast] if t]
        for c in self._consumers.values():
            c.owned = False
            if c.task:
                tasks.append(c.task)
            tasks += [en.task for en in c.pending if en.task]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.backend.abandon()

    async def drain(self, timeout: float = 15.0) -> None:
        """Wait until this process's consumers have handled everything currently in the log."""
        end = time.monotonic() + timeout
        quiet = 0
        while time.monotonic() < end:
            busy = False
            for c in self._consumers.values():
                if c.pending or c.read_upto < await self.backend.tail(c.partition):
                    busy = True
            quiet = 0 if busy else quiet + 1
            if quiet >= 3:
                return
            await asyncio.sleep(0.02)
        raise TimeoutError("bus did not drain")

    # -- partition ownership ----------------------------------------------------------------
    @property
    def owned_partitions(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for (g, p), c in self._consumers.items():
            if c.owned:
                out.setdefault(g, []).append(p)
        return {g: sorted(ps) for g, ps in out.items()}

    async def _manage(self) -> None:
        while True:
            try:
                await self._rebalance()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("bus rebalance failed")
            try:
                await asyncio.wait_for(self._wake.wait(), self.tick_s)
            except TimeoutError:
                pass
            self._wake.clear()

    async def _rebalance(self) -> None:
        P = self.backend.partitions
        for s in [s for s in self._subs if s.group is not None]:
            g = s.group
            assert g is not None
            await self.backend.heartbeat(g, self.member, self.lease_ttl_s)
            n = max(1, len(await self.backend.members(g)))
            target = math.ceil(P / n)
            mine = {p: c for (gg, p), c in self._consumers.items() if gg == g}
            for p, c in list(mine.items()):  # renew; a lost lease means stop consuming NOW
                if not await self.backend.try_lease(g, p, self.member, self.lease_ttl_s):
                    log.info("lost lease %s/%d", g, p)
                    await self._stop_consumer(c, release=False)
                    mine.pop(p)
            for p in sorted(mine, reverse=True)[: max(0, len(mine) - target)]:  # give back extras
                await self._stop_consumer(mine.pop(p), release=True)
            if len(mine) < target:
                start = zlib.crc32(self.member.encode()) % P
                for p in [(start + i) % P for i in range(P)]:
                    if p in mine:
                        continue
                    if await self.backend.try_lease(g, p, self.member, self.lease_ttl_s):
                        mine[p] = self._start_consumer(s, p)
                    if len(mine) >= target:
                        break

    def _start_consumer(self, sub: _Sub, p: int) -> _Consumer:
        assert sub.group is not None
        c = _Consumer(sub.group, p)
        self._consumers[(sub.group, p)] = c
        c.task = asyncio.create_task(self._consume(sub, c))
        return c

    async def _stop_consumer(self, c: _Consumer, *, release: bool) -> None:
        c.owned = False if not release else c.owned
        if c.task:
            c.task.cancel()
            await asyncio.gather(c.task, return_exceptions=True)
        if release:  # graceful: finish what is in flight (bounded), then commit and hand over
            inflight = [en.task for en in c.pending if en.task]
            if inflight:
                await asyncio.wait(inflight, timeout=self.handover_timeout_s)
            await self._commit(c)
            await self.backend.release_lease(c.group, c.partition, self.member)
            c.owned = False
            for en in c.pending:  # whatever is still unfinished is abandoned, not acknowledged
                if en.task and not en.task.done():
                    en.ok = False
                    en.task.cancel()
        else:  # lease lost: do NOT commit; the new owner resumes from the last committed cursor
            for en in c.pending:
                en.ok = False
                if en.task:
                    en.task.cancel()
        c.owned = False
        self._consumers.pop((c.group, c.partition), None)

    # -- consumption ------------------------------------------------------------------------
    async def _commit(self, c: _Consumer) -> None:
        advanced = False
        while (
            c.pending
            and c.pending[0].ok
            and (c.pending[0].task is None or c.pending[0].task.done())
        ):
            c.acked = c.pending.popleft().eid
            advanced = True
        if advanced and c.owned:
            await self.backend.set_cursor(c.group, c.partition, c.acked)

    def _redeliver_delay(self, attempt: int) -> float:
        d = min(self.redeliver_max_s, self.redeliver_base_s * 2 ** (attempt - 1))
        return d * (0.5 + random.random() / 2)

    async def _consume(self, sub: _Sub, c: _Consumer) -> None:
        while True:  # a backend outage must never kill the consumer while we still hold the lease
            try:
                c.acked = c.read_upto = await self.backend.get_cursor(c.group, c.partition)
                break
            except Exception:
                log.warning(
                    "consumer %s/%d: cannot read cursor; retrying",
                    c.group,
                    c.partition,
                    exc_info=True,
                )
                await asyncio.sleep(self._redeliver_delay(1))
        window = asyncio.Semaphore(sub.concurrency)

        async def run(entry: _Entry, e: Event) -> None:
            attempt = 0
            slot = DeliverySlot()
            try:
                while True:
                    slot = DeliverySlot()
                    token = current_delivery.set(slot)
                    try:
                        await sub.handler(e)
                        break
                    except (
                        Redeliver
                    ) as r:  # temporary problem: do NOT acknowledge; try again, in order
                        attempt += 1
                        if slot.deferral is not None:
                            slot.deferral.release()
                        if not (self._running and c.owned):
                            entry.ok = False  # abandoned: stays unacknowledged for the next owner
                            return
                        log.warning(
                            "group %s: redelivering %s (attempt %d): %s",
                            c.group,
                            e.type,
                            attempt,
                            r,
                        )
                        await asyncio.sleep(self._redeliver_delay(attempt))
                    except Exception:
                        log.exception("handler failed in group %s for %s", c.group, e.type)
                        break
                    finally:
                        current_delivery.reset(token)
            finally:
                window.release()  # a deferred event no longer occupies a concurrency slot...
            if slot.deferral is not None:  # ...but is not acknowledged until its work finishes
                await slot.deferral.wait()

        while True:
            try:
                entries = await self.backend.read(c.partition, c.read_upto, self.batch, self.poll_s)
                for eid, ev in entries:
                    entry = _Entry(eid)
                    if sub.matches(ev):
                        await window.acquire()  # blocks while `concurrency` handlers are in flight
                        entry.task = asyncio.create_task(run(entry, ev))
                    c.pending.append(entry)
                    c.read_upto = eid
                    await self._commit(c)
                await self._commit(c)
                if not entries:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning(
                    "consumer %s/%d: backend error; retrying", c.group, c.partition, exc_info=True
                )
                await asyncio.sleep(self._redeliver_delay(1))

    def _start_broadcast_for(self, sub: _Sub) -> None:
        async def run(p: int) -> None:
            while True:
                try:
                    cur = await self.backend.tail(p)  # broadcast = "from now on"
                    break
                except Exception:  # noqa: BLE001 - keep the reader alive through an outage
                    await asyncio.sleep(self._redeliver_delay(1))
            while True:
                try:
                    batch = await self.backend.read(p, cur, self.batch, self.poll_s)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.warning("broadcast reader %d: backend error; retrying", p, exc_info=True)
                    await asyncio.sleep(self._redeliver_delay(1))
                    continue
                for eid, ev in batch:
                    cur = eid
                    if sub.matches(ev):
                        try:
                            await sub.handler(ev)
                        except Exception:
                            log.exception("broadcast handler failed for %s", ev.type)

        self._broadcast += [asyncio.create_task(run(p)) for p in range(self.backend.partitions)]


__all__ = ["BusBackend", "DistributedBus", "MemoryBusBackend", "partition_of"]
