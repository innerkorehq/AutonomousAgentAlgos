from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter

from aaa.events.event import Event
from aaa.events.snapshot import SnapshotCapable, StoredSnapshot
from aaa.events.store import EventStore

S = TypeVar("S")
log = logging.getLogger("aaa.projection")


@dataclass
class Loaded(Generic[S]):
    state: S
    seq: int  # last event folded (-1 if none)
    from_snapshot: int = -1  # seq of the snapshot used, -1 if replayed from scratch
    replayed: int = 0  # number of events folded this load


class Projection(Generic[S]):
    """State = fold over a stream's events. `fold(state, event)` may mutate and return `state`
    (the state object is private to the load). With a `name` and `state_type`, the state is
    snapshotted every `every` events and later loads cost O(tail), not O(history).
    Bump `version` whenever the fold logic or state shape changes: stale snapshots are ignored."""

    def __init__(
        self,
        initial: Callable[[], S],
        fold: Callable[[S, Event], S],
        *,
        name: str = "",
        version: int = 1,
        state_type: Any = None,
        every: int = 64,
    ) -> None:
        self.initial, self.fold = initial, fold
        self.name, self.version, self.every = name, version, every
        self._ta: TypeAdapter[S] | None = TypeAdapter(state_type) if (name and state_type) else None

    def replay(self, events: Iterable[Event], state: S | None = None) -> S:
        s = self.initial() if state is None else state
        for e in events:
            s = self.fold(s, e)
        return s

    async def load(
        self,
        store: EventStore,
        stream: str,
        *,
        hydrate: Callable[[Event], Awaitable[Event]] | None = None,
        use_snapshots: bool = True,
    ) -> Loaded[S]:
        snapshots = use_snapshots and self._ta is not None and isinstance(store, SnapshotCapable)
        state: S | None = None
        snap_seq = -1
        if snapshots:
            assert self._ta is not None and isinstance(store, SnapshotCapable)
            try:
                snap = await store.load_snapshot(stream, self.name)
                if (
                    snap
                    and snap.version == self.version
                    and snap.seq <= await store.last_seq(stream)
                ):
                    state, snap_seq = self._ta.validate_json(snap.state), snap.seq
            except Exception:
                log.warning("ignoring unreadable snapshot %s/%s", stream, self.name, exc_info=True)
                state, snap_seq = None, -1
        events = await store.read(stream, snap_seq + 1)
        if hydrate:
            events = [await hydrate(e) for e in events]
        state = self.replay(events, state)
        seq = events[-1].seq if events else snap_seq
        if snapshots and len(events) >= self.every:
            assert self._ta is not None and isinstance(store, SnapshotCapable)
            try:
                await store.save_snapshot(
                    StoredSnapshot(
                        stream=stream,
                        name=self.name,
                        seq=seq,
                        version=self.version,
                        state=self._ta.dump_json(state).decode(),
                    )
                )
            except Exception:
                log.warning("could not save snapshot %s/%s", stream, self.name, exc_info=True)
        return Loaded(state, seq, snap_seq, len(events))
