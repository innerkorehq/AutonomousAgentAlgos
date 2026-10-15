"""The observer: turns finished (and, for sessions, still-open) runs into traces and hands them to sinks.

* **Export unit = a whole trace** (the root run, its child runs, their children...), ordered parents first. Several
  backends drop a span whose parent has not arrived (LangSmith does, silently), so spans are never sent child-first.
* **Triggered by the root run completing**, in a competing-consumer group, so in a cluster exactly one process exports a
  given trace. Child runs are exported with their root.
* **Long-lived runs** (sessions never complete until closed) are exported incrementally by `flush_open()`: closed
  spans not yet sent plus a provisional root span marked `aaa.partial`. Pass `flush_interval_s` to do it on a timer.
* **Idempotent**: span ids are deterministic and an in-memory record avoids resending within one process. After a restart
  a backend may see a span twice; sinks must tolerate that.
* **Never fails a run**: it reads the log from its own worker (`aaa.safe`), and a failing sink is logged and counted.

Times are event times, so a span includes queueing and any clock difference between processes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Self

from aaa.core.aio import wait_for
from aaa.events import Event
from aaa.observe.base import TraceSink
from aaa.runtime import Engine
from aaa.safe import SafeWorker, safe_subscribe
from aaa.trace import A, Span, children_of, spans_from_events

log = logging.getLogger("aaa.observe")
RUN_COMPLETED = "aaa.run.completed"


def _order(spans: list[Span]) -> list[Span]:
    """Parents before children, whatever the clocks say; siblings by start time."""
    ids = {s.span_id for s in spans}
    kids: dict[str | None, list[Span]] = {}
    for s in sorted(spans, key=lambda s: (s.start, s.span_id)):
        kids.setdefault(s.parent_id if s.parent_id in ids else None, []).append(s)
    out: list[Span] = []

    def walk(parent: str | None) -> None:
        for s in kids.get(parent, []):
            out.append(s)
            walk(s.span_id)

    walk(None)
    return out


class Observer:
    def __init__(
        self,
        engine: Engine,
        sinks: Sequence[TraceSink] | TraceSink,
        *,
        capture_content: bool = False,
        content_limit: int = 4000,
        sample: float = 1.0,
        tenant: str | None = None,
        settle_s: float = 0.2,
        flush_interval_s: float | None = None,
        sink_timeout_s: float = 30.0,
        max_tracked: int = 10_000,
        group: str = "aaa.observe",
    ) -> None:
        self.engine = engine
        self.sinks: list[TraceSink] = list(sinks) if isinstance(sinks, Sequence) else [sinks]
        self.capture_content, self.content_limit = capture_content, content_limit
        self.sample, self.tenant, self.settle_s = sample, tenant, settle_s
        self.flush_interval_s, self.sink_timeout_s, self.max_tracked, self.group = (
            flush_interval_s,
            sink_timeout_s,
            max_tracked,
            group,
        )
        # root run id -> {span id: was it sent as a partial (still-open) span}
        self._sent: OrderedDict[str, dict[str, bool]] = OrderedDict()
        self._worker: SafeWorker | None = None
        self._flusher: asyncio.Task[None] | None = None
        self.exported_spans = 0
        self.export_errors = 0

    # -- lifecycle ----------------------------------------------------------------------------
    def attach(self) -> Self:
        if self._worker is None:
            self._worker = safe_subscribe(
                self.engine, RUN_COMPLETED, self._on_completed, name="aaa.observe", group=self.group
            )
        if self.flush_interval_s and self._flusher is None:
            self._flusher = self.engine.spawn(self._flush_loop())
        return self

    async def __aenter__(self) -> Self:
        return self.attach()

    async def __aexit__(self, *exc: object) -> None:
        await self.drain()
        await self.stop()

    async def drain(self, timeout_s: float = 10.0) -> bool:
        """Wait until every trace for runs that have completed so far has been exported. (Events reach the observer through
        the bus, so the bus is flushed first; a distributed bus delivers on its own schedule and has nothing to flush.)"""
        if self._worker is None:
            return True
        try:
            await wait_for(self.engine.bus.drain(), timeout_s)
        except TimeoutError:
            return False
        return await self._worker.drain(timeout_s)

    async def stop(self) -> None:
        if self._flusher is not None:
            self._flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._flusher
            self._flusher = None
        if self._worker is not None:
            await self._worker.stop()
            self._worker = None

    # -- reading the log ----------------------------------------------------------------------
    async def _events(self, run_id: str) -> list[Event]:
        eng = self.engine
        raw = await eng.retry(lambda: eng.store.read(run_id), "read")
        return [await eng.hydrate(e) for e in raw]

    async def _collect(self, root_id: str) -> tuple[list[Span], dict[str, list[Event]]]:
        """Spans of the whole trace under `root_id`, plus the raw events by run."""
        by_run: dict[str, list[Event]] = {}
        spans: list[Span] = []

        async def visit(run_id: str, parent_events: list[Event] | None, depth: int) -> None:
            events = await self._events(run_id)
            by_run[run_id] = events
            spans.extend(
                spans_from_events(
                    events,
                    parent_events=parent_events,
                    capture_content=self.capture_content,
                    content_limit=self.content_limit,
                )
            )
            if depth < 16:
                for child in children_of(events):
                    if child not in by_run:
                        await visit(child, events, depth + 1)

        await visit(root_id, None, 0)
        return spans, by_run

    def _sampled(self, trace_id: str) -> bool:
        return self.sample >= 1.0 or (int(trace_id[:8], 16) % 10_000) < self.sample * 10_000

    # -- exporting ----------------------------------------------------------------------------
    async def export_trace(self, root_run_id: str, *, final: bool = True) -> int:
        """Export the trace rooted at `root_run_id`. `final=False` sends only closed spans not yet sent, plus a provisional root."""
        spans, by_run = await self._collect(root_run_id)
        if not spans:
            return 0
        root = next((s for s in spans if s.run_id == root_run_id and s.kind == "agent"), None)
        if (
            root is None
            or not self._sampled(root.trace_id)
            or (self.tenant is not None and root.tenant != self.tenant)
        ):
            return 0
        last = max((e.time for evs in by_run.values() for e in evs), default=root.start)
        end_of_trace = root.end or last
        sent = self._sent.setdefault(root_run_id, {})
        self._sent.move_to_end(root_run_id)
        while len(self._sent) > self.max_tracked:
            self._sent.popitem(last=False)
        batch: list[Span] = []
        for s in spans:
            sid = s.span_id
            if sid in sent and (not final or not sent[sid]):
                continue  # already out; a final export only replaces spans that went out as partial
            if s.end is None:
                if not final and s is not root:
                    continue  # still running: sent when it closes
                s = replace(
                    s,
                    end=end_of_trace,
                    status="incomplete",
                    attributes={**s.attributes, A.PARTIAL: True},
                )
            batch.append(s)
        if not batch:
            return 0
        await self._ship(_order(batch))
        sent.update({s.span_id: bool(s.attributes.get(A.PARTIAL)) for s in batch})
        return len(batch)

    async def _ship(self, spans: list[Span]) -> None:
        async def one(sink: TraceSink) -> None:
            try:
                await wait_for(sink.export(spans), self.sink_timeout_s)
                self.exported_spans += len(spans)
            except Exception:
                self.export_errors += 1
                log.warning(
                    "trace sink %s failed to export %d span(s)",
                    type(sink).__name__,
                    len(spans),
                    exc_info=True,
                )

        await asyncio.gather(*(one(s) for s in self.sinks))

    async def _on_completed(self, e: Event) -> None:
        run_id = e.run_id
        if not run_id:
            return
        started = next(iter(await self._events_head(run_id)), None)
        if started is None or started.data.get("parent"):
            return  # a child run: exported with its root
        if self.settle_s:
            await asyncio.sleep(self.settle_s)  # let cancelled children write their final events
        await self.export_trace(run_id, final=True)

    async def _events_head(self, run_id: str) -> list[Event]:
        eng = self.engine
        return await eng.retry(lambda: eng.store.read(run_id, 0, 1), "read")

    # -- long-lived runs and history ----------------------------------------------------------
    async def flush_open(self) -> int:
        """Export what is new in every root run that is still open (sessions, or runs hosted by a crashed process)."""
        n = 0
        for info in await self.engine.retry(self.engine.store.run_index, "read"):
            if info.completed is None and not info.started.data.get("parent"):
                try:
                    n += await self.export_trace(info.stream, final=False)
                except Exception:
                    log.warning("could not flush open run %s", info.stream, exc_info=True)
        return n

    async def _flush_loop(self) -> None:
        assert self.flush_interval_s
        while True:
            await asyncio.sleep(self.flush_interval_s)
            try:
                await self.flush_open()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("flush_open failed", exc_info=True)

    async def backfill(self, *, since: datetime | None = None, limit: int | None = None) -> int:
        """Export traces for finished root runs already in the log (to point a new backend at history)."""
        cutoff = since if since is None or since.tzinfo else since.replace(tzinfo=UTC)
        n = 0
        runs = [
            i
            for i in await self.engine.retry(self.engine.store.run_index, "read")
            if i.completed and not i.started.data.get("parent")
        ]
        for info in sorted(runs, key=lambda i: i.completed.time if i.completed else i.started.time):
            if cutoff and info.completed and info.completed.time < cutoff:
                continue
            n += await self.export_trace(info.stream, final=True)
            if limit is not None and n >= limit:
                break
        return n

    async def close_sinks(self) -> None:
        for s in self.sinks:
            with contextlib.suppress(Exception):
                await s.aclose()


def _unused(_: Any) -> None:  # keeps `Any` imported for subclasses that annotate with it
    return None
