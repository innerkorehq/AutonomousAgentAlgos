"""Run a blocking client on one dedicated thread, so the event loop never waits on it and the client is never used concurrently."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class SyncRunner:
    def __init__(self, name: str = "aaa-vector") -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    async def run(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, lambda: fn(*args, **kw))

    def close(self) -> None:
        self._pool.shutdown(wait=True)
