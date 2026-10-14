"""Cancellation must always end a long-lived task, even when something below it swallows the cancel.

Background: on Python 3.11 the standard library's `asyncio.timeout` (used by redis-py) can convert a task's
cancellation into a `TimeoutError` when the two race, and the caller carries on. A bus manager that never
stops makes `await bus.abort()` / `Engine.__aexit__` hang forever (seen with Redis on 3.11)."""

import asyncio
import random

import pytest

from aaa.core import raise_if_cancelling, wait_event, wait_for
from aaa.events import DistributedBus, MemoryBusBackend


async def test_wait_for_returns_raises_and_propagates():
    async def ok():
        await asyncio.sleep(0.01)
        return 7

    async def boom():
        raise ValueError("inner")

    assert await wait_for(ok(), 1) == 7
    with pytest.raises(TimeoutError):
        await wait_for(asyncio.sleep(5), 0.02)
    with pytest.raises(ValueError, match="inner"):
        await wait_for(boom(), 1)
    ev = asyncio.Event()
    assert await wait_event(ev, 0.02) is False
    ev.set()
    assert await wait_event(ev, 0.02) is True


async def test_wait_for_never_loses_a_cancel_that_races_its_timeout():
    """Cancel each waiter right around the instant its timeout fires; every one must end."""
    random.seed(7)
    swallowed = 0
    for _ in range(150):
        t = asyncio.create_task(wait_for(asyncio.sleep(10), 0.004))
        await asyncio.sleep(0.004 + random.uniform(-0.0015, 0.0015))
        t.cancel()
        done, _ = await asyncio.wait({t}, timeout=2)
        assert done, "a waiter that was cancelled never finished (cancellation lost)"
        if not t.cancelled() and t.exception() is None:
            swallowed += 1  # finished NORMALLY after being cancelled
    assert swallowed == 0


async def test_raise_if_cancelling_ends_a_task_whose_cancel_was_swallowed_below_it():
    async def loop():
        while True:
            raise_if_cancelling()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:  # simulates a library that eats the cancellation
                pass

    t = asyncio.create_task(loop())
    await asyncio.sleep(0.05)
    t.cancel()
    done, _ = await asyncio.wait({t}, timeout=2)
    assert done and t.cancelled()


class SwallowingBackend:
    """A bus backend whose calls eat a cancellation once (as redis-py can on 3.11)."""

    def __init__(self, real):
        self._real, self.swallowed = real, 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def heartbeat(self, *a):
        try:
            await asyncio.sleep(0.3)  # long enough to be cancelled in the middle of
        except asyncio.CancelledError:
            self.swallowed += 1  # ...and swallow it
        return await self._real.heartbeat(*a)


async def test_a_bus_whose_manager_has_its_cancel_swallowed_still_shuts_down():
    backend = SwallowingBackend(MemoryBusBackend(4))
    bus = DistributedBus(backend, member="A", lease_ttl_s=1.0, tick_s=0.05, poll_s=0.05)

    async def h(e):
        return None

    bus.subscribe("t", h, group="g")
    await bus.__aenter__()
    await asyncio.sleep(0.15)  # manager is now inside the swallowing heartbeat
    aborting = asyncio.ensure_future(bus.abort())
    done, _ = await asyncio.wait({aborting}, timeout=5)  # fail fast: never wait on a stuck abort
    assert done, "bus.abort() never returned: the manager kept running after being cancelled"
    assert backend.swallowed >= 1
