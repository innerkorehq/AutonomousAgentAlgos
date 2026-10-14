from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from aaa.core.errors import ConcurrencyError
from aaa.events.event import Event
from aaa.events.snapshot import StoredSnapshot

ANY_SEQ = -2  # expected_seq value meaning "don't check"
RUN_STARTED = "aaa.run.started"
RUN_COMPLETED = "aaa.run.completed"


@dataclass
class RunInfo:
    stream: str
    started: Event
    completed: Event | None


@runtime_checkable
class EventStore(Protocol):
    async def append(
        self, stream: str, events: Sequence[Event], expected_seq: int = ANY_SEQ
    ) -> list[Event]:
        """Atomically append; `expected_seq` is the stream's last seq (-1 = empty stream).
        Raises ConcurrencyError on mismatch. IDEMPOTENT: an event whose id is already in the
        stream is not appended again; the existing stored event is returned with `dup=True`."""
        ...

    async def read(
        self, stream: str, from_seq: int = 0, limit: int | None = None
    ) -> list[Event]: ...

    def subscribe(self, stream: str, from_seq: int = 0) -> AsyncIterator[Event]:
        """Replay from `from_seq`, then tail live appends."""
        ...

    async def last_seq(self, stream: str) -> int: ...

    async def has(self, stream: str, event_id: str) -> bool:
        """O(1) 'is this event id already in the stream?' (ids are unique per stream)."""
        ...

    async def run_index(self) -> list[RunInfo]:
        """Every run's started event and (if any) completed event, for crash recovery."""
        ...


class InMemoryEventStore:
    def __init__(self, outbox: Any = None) -> None:
        self.outbox = outbox  # optional MemoryBusBackend: appends are enqueued for a DistributedBus
        self._streams: dict[str, list[Event]] = defaultdict(list)
        self._ids: dict[str, dict[str, Event]] = defaultdict(dict)
        self._cond = asyncio.Condition()
        self._snaps: dict[tuple[str, str], StoredSnapshot] = {}

    async def load_snapshot(self, stream: str, name: str) -> StoredSnapshot | None:
        return self._snaps.get((stream, name))

    async def save_snapshot(self, snap: StoredSnapshot) -> None:
        cur = self._snaps.get((snap.stream, snap.name))
        if cur is None or snap.seq > cur.seq:
            self._snaps[(snap.stream, snap.name)] = snap

    async def last_seq(self, stream: str) -> int:
        return len(self._streams[stream]) - 1

    async def has(self, stream: str, event_id: str) -> bool:
        return event_id in self._ids[stream]

    async def append(
        self, stream: str, events: Sequence[Event], expected_seq: int = ANY_SEQ
    ) -> list[Event]:
        async with self._cond:
            log = self._streams[stream]
            last = len(log) - 1
            if expected_seq != ANY_SEQ and expected_seq != last:
                raise ConcurrencyError(f"{stream}: expected seq {expected_seq}, at {last}")
            out: list[Event] = []
            for e in events:
                if e.id in self._ids[stream]:
                    out.append(self._ids[stream][e.id].model_copy(update={"dup": True}))
                    continue
                stored = e.model_copy(update={"stream": stream, "seq": len(log)})
                log.append(stored)
                self._ids[stream][e.id] = stored
                out.append(stored)
                if self.outbox is not None:
                    await self.outbox.enqueue(stored)
            self._cond.notify_all()
            return out

    async def read(self, stream: str, from_seq: int = 0, limit: int | None = None) -> list[Event]:
        log = self._streams[stream][from_seq:]
        return log[:limit] if limit is not None else list(log)

    async def run_index(self) -> list[RunInfo]:
        out = []
        for stream, log in self._streams.items():
            started = next((e for e in log if e.type == RUN_STARTED), None)
            if started:
                done = next((e for e in reversed(log) if e.type == RUN_COMPLETED), None)
                out.append(RunInfo(stream, started, done))
        return out

    async def subscribe(self, stream: str, from_seq: int = 0) -> AsyncIterator[Event]:
        pos = from_seq
        while True:
            async with self._cond:
                await self._cond.wait_for(lambda p=pos: len(self._streams[stream]) > p)
                batch = self._streams[stream][pos:]
            for e in batch:
                yield e
            pos += len(batch)
