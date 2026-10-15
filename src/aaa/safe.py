"""Observers must never fail a run.

`Engine.subscribe` treats an exception in a handler as a failed run, which is right for algorithms and wrong for
anything that merely watches (trace exporters, online evaluators, metrics). `safe_subscribe` gives such a watcher
its own bounded queue and worker task: the engine's handler only enqueues, so a slow or crashing watcher can neither
fail nor slow the run. Overflow drops the newest events and counts them instead of applying backpressure."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aaa.events import Event

if TYPE_CHECKING:
    from aaa.runtime import Engine

log = logging.getLogger("aaa.safe")


class SafeWorker:
    def __init__(
        self, handler: Callable[[Event], Awaitable[None]], *, name: str, maxsize: int = 10_000
    ) -> None:
        self.handler, self.name = handler, name
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0  # events lost to a full queue
        self.errors = 0  # handler exceptions swallowed
        self._task: asyncio.Task[None] | None = None

    def start(self, engine: Engine) -> None:
        self._task = engine.spawn(self._run())

    def offer(self, event: Event) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 1000 == 0:
                log.warning("%s: queue full, dropped %d event(s)", self.name, self.dropped)

    async def _run(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                await self.handler(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.errors += 1
                log.exception("%s: handler failed for %s (ignored)", self.name, event.type)
            finally:
                self.queue.task_done()

    async def drain(self, timeout_s: float = 10.0) -> bool:
        """Wait until everything queued so far has been handled. False if it did not finish in time."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout_s)
            return True
        except TimeoutError:
            return False

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None


def safe_subscribe(
    engine: Engine,
    pattern: str,
    handler: Callable[[Event], Awaitable[None]],
    *,
    name: str,
    group: str | None = None,
    maxsize: int = 10_000,
) -> SafeWorker:
    """Subscribe `handler` to `pattern` such that it can never fail or stall a run. With a distributed bus,
    processes using the same `group` compete (each event is seen by one of them)."""
    worker = SafeWorker(handler, name=name, maxsize=maxsize)

    async def intake(e: Event) -> None:
        worker.offer(e)

    engine.subscribe(pattern, intake, group=group)
    worker.start(engine)
    return worker
