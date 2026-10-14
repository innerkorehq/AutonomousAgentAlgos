"""A 'cluster' = a shared durable backend plus the ability to create independent nodes
(each its own store + bus-backend objects/connections, like separate processes would)."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aaa.events import DistributedBus, InMemoryEventStore, MemoryBusBackend

FAST = {"lease_ttl_s": 1.0, "tick_s": 0.1, "poll_s": 0.1}


@dataclass
class Node:
    store: Any
    backend: Any

    def bus(self, **kw: Any) -> DistributedBus:
        return DistributedBus(self.backend, **{**FAST, **kw})

    async def close(self) -> None:
        if hasattr(self.store, "aclose"):
            await self.store.aclose()
        await self.backend.close()


class Cluster:
    def __init__(
        self,
        kind: str,
        *,
        partitions: int = 8,
        admin_dsn: str | None = None,
        url: str | None = None,
        prefix: str = "aaa",
    ) -> None:
        self.kind, self.partitions, self.admin_dsn = kind, partitions, admin_dsn
        self.url, self.prefix = url, prefix
        self.dsn: str | None = None
        self._db: str | None = None
        self._nodes: list[Node] = []
        self._shared: Node | None = None

    async def setup(self) -> None:
        if self.kind == "postgres":  # a fresh database per test, on the throwaway server
            import asyncpg

            self._db = "t_" + uuid.uuid4().hex[:12]
            admin = await asyncpg.connect(self.admin_dsn)
            await admin.execute(f'CREATE DATABASE "{self._db}"')
            await admin.close()
            assert self.admin_dsn
            self.dsn = self.admin_dsn.rsplit("/", 1)[0] + "/" + self._db

    async def node(self) -> Node:
        if self.kind == "memory":
            if self._shared is None:
                backend = MemoryBusBackend(self.partitions)
                self._shared = Node(InMemoryEventStore(outbox=backend), backend)
            return self._shared
        if self.kind == "postgres":
            from aaa.events.postgres import PostgresEventStore

            assert self.dsn
            store = await PostgresEventStore.create(self.dsn, partitions=self.partitions)
        else:
            from aaa.events.redis import RedisEventStore

            assert self.url
            store = await RedisEventStore.create(
                self.url, partitions=self.partitions, prefix=self.prefix
            )
        n = Node(store, store.bus_backend())
        self._nodes.append(n)
        return n

    async def close(self) -> None:
        for n in self._nodes:
            await n.close()
        if self._db:
            import asyncpg

            admin = await asyncpg.connect(self.admin_dsn)
            await admin.execute(f'DROP DATABASE IF EXISTS "{self._db}" WITH (FORCE)')
            await admin.close()


async def until(pred, timeout: float = 10.0, every: float = 0.02) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = pred()
        if asyncio.iscoroutine(r):
            r = await r
        if r:
            return
        await asyncio.sleep(every)
    raise TimeoutError("condition not met in time")
