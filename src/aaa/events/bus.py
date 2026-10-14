from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, Self

from aaa.events.event import Event

log = logging.getLogger("aaa.bus")

# A handler reads an event and returns follow-up events (facts/commands) or None.
Handler = Callable[[Event], Awaitable[Iterable[Event] | None]]


class EventBus(Protocol):
    async def publish(self, event: Event) -> None: ...

    def subscribe(
        self,
        pattern: str,
        handler: Handler,
        *,
        group: str | None = None,
        concurrency: int = 1,
        predicate: Callable[[Event], bool] | None = None,
    ) -> None: ...


@dataclass
class _Sub:
    pattern: str
    handler: Handler
    predicate: Callable[[Event], bool] | None
    concurrency: int
    queue: asyncio.Queue[Event | None]
    workers: list[asyncio.Task[None]] = field(default_factory=list)


class InProcessBus:
    """Bounded-queue bus. Publishing blocks when a subscriber is full (backpressure).
    concurrency=1 preserves publish order per subscription. Handler exceptions are
    logged and isolated: one bad handler never kills the bus or other subscribers.
    Subscriptions sharing a `group` compete for events (one delivery per group)."""

    def __init__(self, queue_size: int = 1024) -> None:
        self._queue_size = queue_size
        self._subs: list[_Sub] = []
        self._groups: dict[str, list[_Sub]] = {}
        self._rr: dict[str, int] = {}
        self._running = False
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    def subscribe(
        self,
        pattern: str,
        handler: Handler,
        *,
        group: str | None = None,
        concurrency: int = 1,
        predicate: Callable[[Event], bool] | None = None,
    ) -> None:
        sub = _Sub(pattern, handler, predicate, concurrency, asyncio.Queue(self._queue_size))
        self._subs.append(sub)
        if group:
            self._groups.setdefault(f"{group}\0{pattern}", []).append(sub)
        if self._running:
            self._spawn(sub)

    def _spawn(self, sub: _Sub) -> None:
        sub.workers = [asyncio.create_task(self._work(sub)) for _ in range(sub.concurrency)]

    async def __aenter__(self) -> Self:
        self._running = True
        for s in self._subs:
            self._spawn(s)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.drain()
        for s in self._subs:
            for _ in s.workers:
                await s.queue.put(None)
        await asyncio.gather(*(w for s in self._subs for w in s.workers))
        self._running = False

    async def abort(self) -> None:
        """Hard stop (crash simulation / emergency shutdown): cancel workers, drop queues."""
        tasks = [w for s in self._subs for w in s.workers]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._running = False

    async def drain(self) -> None:
        """Wait until all queued and in-flight events (and their follow-ups) are handled."""
        await self._idle.wait()

    def _targets(self, event: Event) -> list[_Sub]:
        matched = [
            s
            for s in self._subs
            if fnmatch.fnmatchcase(event.type, s.pattern)
            and (s.predicate is None or s.predicate(event))
        ]
        out: list[_Sub] = []
        seen_groups: set[str] = set()
        for s in matched:
            key = next((k for k, v in self._groups.items() if s in v), None)
            if key is None:
                out.append(s)
            elif key not in seen_groups:
                seen_groups.add(key)
                members = [m for m in self._groups[key] if m in matched]
                i = self._rr.get(key, 0)
                self._rr[key] = i + 1
                out.append(members[i % len(members)])
        return out

    async def publish(self, event: Event) -> None:
        for s in self._targets(event):
            self._inflight += 1
            self._idle.clear()
            await s.queue.put(event)

    def _done(self) -> None:
        self._inflight -= 1
        if self._inflight == 0:
            self._idle.set()

    async def _work(self, sub: _Sub) -> None:
        while (event := await sub.queue.get()) is not None:
            try:
                for follow in await sub.handler(event) or ():
                    await self.publish(follow)
            except Exception:
                log.exception("handler failed for %s", event.type)
            finally:
                self._done()


class ClientBus:
    """For processes that only SUBMIT and OBSERVE runs (an API server, a CLI): it delivers and
    consumes nothing, so the process never becomes a worker. Pair it with a shared store; the
    workers' DistributedBus picks the events up from the store's outbox."""

    distributed = True

    def subscribe(self, pattern: str, handler: Handler, **kw: object) -> None:
        return None

    async def publish(self, event: Event) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def abort(self) -> None:
        return None

    async def drain(self, timeout: float = 0) -> None:
        return None
