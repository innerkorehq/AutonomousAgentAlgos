"""Cancellation-safe waiting.

Before Python 3.12, `asyncio.wait_for` can LOSE a cancellation that arrives at the same moment its timeout
fires (the cancel is swallowed and the task carries on). For a long-lived loop that is shut down by cancelling
it (a bus manager, a consumer, a poller) that means a task that never ends, and whoever awaits it hangs
forever. These helpers are built on `asyncio.wait`, which always propagates a cancellation, and behave
the same on every supported Python."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def raise_if_cancelling() -> None:
    """Re-raise a cancellation that something below us swallowed.

    On Python 3.11 the standard library's `asyncio.timeout` (used by redis-py and others) can turn a
    task's cancellation into a `TimeoutError` when the two race, and the caller then carries on as if
    nothing happened. Long-lived loops call this at the top of each iteration so a task that WAS
    cancelled always ends, instead of leaving whoever awaits it hanging forever."""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


async def wait_for(aw: Awaitable[T], timeout: float | None) -> T:
    """Like `asyncio.wait_for` (raises the builtin `TimeoutError`), but a cancellation of the caller is never lost."""
    fut = asyncio.ensure_future(aw)
    try:
        done, _ = await asyncio.wait({fut}, timeout=timeout)
        if done:
            return fut.result()
        raise TimeoutError
    finally:
        if not fut.done():
            fut.cancel()
            await asyncio.wait({fut})  # let the inner awaitable unwind before we leave


async def wait_event(event: asyncio.Event, timeout: float | None) -> bool:
    """Wait up to `timeout` for `event`; True if it was set. Cancellation-safe."""
    try:
        await wait_for(event.wait(), timeout)
        return True
    except TimeoutError:
        return False
