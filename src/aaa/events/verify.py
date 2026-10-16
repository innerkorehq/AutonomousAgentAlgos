"""Consistency check for an event store, for after a restore (or before an upgrade): does every run's log still make sense?

Checks, per run stream: sequence numbers are 0..n-1 with no gap or repeat, the first event is `aaa.run.started`, event ids are
unique, at most one `aaa.run.completed` exists, every child run named by `aaa.child.started` exists, and every blob an event
refers to is present in the blob store (with `deep=True`, also that its bytes still match their sha256). It reads; it never writes.

It cannot tell you that a restore is *recent*: compare the newest event time it reports with what you expect."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aaa.events.event import Event
from aaa.events.store import EventStore


@dataclass
class VerifyReport:
    runs: int = 0
    events: int = 0
    open_runs: int = 0
    blobs_checked: int = 0
    newest_event: datetime | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


async def verify_store(
    store: EventStore, blobs: Any = None, *, deep: bool = False, max_problems: int = 50
) -> VerifyReport:
    rep = VerifyReport()
    index = await store.run_index()
    known = {r.stream for r in index}

    def problem(msg: str) -> None:
        if len(rep.problems) < max_problems:
            rep.problems.append(msg)

    for info in index:
        rep.runs += 1
        events: list[Event] = await store.read(info.stream)
        rep.events += len(events)
        if info.completed is None:
            rep.open_runs += 1
        if [e.seq for e in events] != list(range(len(events))):
            problem(
                f"{info.stream}: sequence numbers are not 0..{len(events) - 1} (a gap or a repeat)"
            )
        if not events or events[0].type != "aaa.run.started":
            problem(f"{info.stream}: does not start with aaa.run.started")
        if len({e.id for e in events}) != len(events):
            problem(f"{info.stream}: duplicate event ids")
        if sum(e.type == "aaa.run.completed" for e in events) > 1:
            problem(f"{info.stream}: more than one aaa.run.completed")
        for e in events:
            if rep.newest_event is None or e.time > rep.newest_event:
                rep.newest_event = e.time
            if e.type == "aaa.child.started" and e.data.get("child_run_id") not in known:
                problem(f"{info.stream}: child run {e.data.get('child_run_id')!r} is missing")
            if blobs is not None:
                for ref in e.blobs:
                    rep.blobs_checked += 1
                    try:
                        if not await blobs.exists(ref):
                            problem(
                                f"{info.stream}: blob {ref.sha256[:12]} (event {e.id}) is missing"
                            )
                        elif deep:
                            await blobs.get(ref)  # raises BlobIntegrityError on a mismatch
                    except Exception as exc:  # noqa: BLE001 - report, keep checking
                        problem(
                            f"{info.stream}: blob {ref.sha256[:12]} unreadable: {type(exc).__name__}"
                        )
    return rep
