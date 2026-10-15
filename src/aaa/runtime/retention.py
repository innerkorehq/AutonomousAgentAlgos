"""Erasure and retention.

The log is append-only, so "delete" is done the event-sourcing way: a finished run is either
**scrubbed** (every event keeps its id, type, seq, time, causal ids and tenant, but its payload,
blobs, meta and subject are gone; a `aaa.run.erased` marker records that it happened) or
**deleted** outright. Either way the run's snapshots, claim-checked blobs and the bus's transient
copies of its events are removed too, so nothing that held the data survives.

Only finished runs are erased (an open run is still being worked on; pass `force=True` to override).
Blobs are stored per stream (the claim-check payload is namespaced by its stream), so erasing one
run can never remove another run's data even if two payloads are textually identical."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from aaa.core.errors import RunNotFinished
from aaa.events import Event
from aaa.events.event import ERASED

if TYPE_CHECKING:
    from aaa.runtime.engine import Engine

log = logging.getLogger("aaa.retention")
RUN_ERASED = "aaa.run.erased"


@dataclass(frozen=True)
class RetentionPolicy:
    """Erase finished runs older than `older_than`. `tenant=None` matches runs of any tenant (and
    untenanted ones). With several policies the first one whose tenant matches a run applies, so list
    tenant-specific policies before a catch-all."""

    older_than: timedelta
    delete: bool = False  # False: scrub (keep the audit skeleton); True: remove the run entirely
    tenant: str | None = None
    interval_s: float = 3600.0  # how often `sweep()` re-evaluates the policy

    def matches(self, tenant: str | None) -> bool:
        return self.tenant is None or self.tenant == tenant


@dataclass
class ErasureReport:
    stream: str
    mode: str  # "scrub" | "delete"
    events: int = 0
    blobs: int = 0
    provider_state: int = 0  # conversation states deleted at the LLM provider


@dataclass
class MemoryReport:
    forgotten: dict[str, int] = field(default_factory=dict)  # backend class -> memories removed
    unsupported: list[str] = field(default_factory=list)  # backends that cannot forget


@dataclass
class PurgeReport:
    erased: list[ErasureReport] = field(default_factory=list)
    skipped_open: int = 0
    memory: MemoryReport | None = None  # set by TenantScope.erase_all

    @property
    def streams(self) -> list[str]:
        return [r.stream for r in self.erased]


async def erase_run(
    engine: Engine,
    run_id: str,
    *,
    delete: bool = False,
    reason: str = "erasure",
    force: bool = False,
) -> ErasureReport:
    """Erase one run. Idempotent: repeating it (e.g. after a crash half way) finishes the job."""
    from aaa.runtime.engine import RUNMETA

    first = await engine.retry(lambda: engine.store.read(run_id, 0, 1), "read")
    if not first:
        raise KeyError(f"no run {run_id!r}")
    meta = (await RUNMETA.load(engine.store, run_id, use_snapshots=False)).state
    if meta.completed is None and not force:
        raise RunNotFinished(f"run {run_id!r} is still open; cancel it first or pass force=True")
    events = await engine.retry(lambda: engine.store.read(run_id), "read")
    refs = {b.sha256: b for e in events for b in e.blobs}
    state_ids = [
        e.data["state_id"]
        for e in events
        if e.type == "aaa.llm.completed" and e.data.get("state_id")
    ]
    if engine.blobs is not None:  # first: if we crash after this, a rerun still finds the events
        for ref in refs.values():
            await engine.retry(lambda ref=ref: engine.blobs.delete(ref), "blob delete")  # type: ignore[union-attr]
    forgotten = 0
    if state_ids:  # first, like blobs: a rerun after a crash must still find the ids. A provider that stored the conversation (Responses API) holds the data too
        for holder in engine.state_holders:
            forgotten += await engine.retry(
                lambda holder=holder: holder.forget_state(state_ids), "provider state delete"
            )
    n = await engine.retry(
        lambda: engine.store.delete_stream(run_id) if delete else engine.store.scrub(run_id),
        "erase",
    )
    if not delete:  # the audit trail records that, and why, the payloads are gone
        marker = Event(
            id=f"{run_id}:erased",
            type=RUN_ERASED,
            stream=run_id,
            run_id=run_id,
            tenant=first[0].tenant,
            source="engine",
            data={
                "reason": reason,
                "events": n,
                "blobs": len(refs),
                **({"provider_state": forgotten} if forgotten else {}),
            },
        )
        await engine.retry(lambda: engine.store.append(run_id, [marker]), "append")
    engine._forget(run_id)
    log.info("erased run %s (%s): %d events, %d blobs", run_id, reason, n, len(refs))
    return ErasureReport(run_id, "delete" if delete else "scrub", n, len(refs), forgotten)


def _already_erased(completed: Event) -> bool:
    return completed.data.get(ERASED) is True


async def purge(
    engine: Engine,
    *,
    older_than: timedelta,
    tenant: str | None = None,
    delete: bool = False,
    reason: str = "retention",
    now: datetime | None = None,
) -> PurgeReport:
    """Erase every finished run (of `tenant`, if given) that completed more than `older_than` ago.
    Runs already scrubbed are left alone; with `delete=True` they are removed."""
    cutoff = (now or datetime.now(UTC)) - older_than
    report = PurgeReport()
    for info in await engine.retry(engine.store.run_index, "read"):
        if tenant is not None and info.started.tenant != tenant:
            continue
        if info.completed is None:
            report.skipped_open += 1
            continue
        if info.completed.time > cutoff or (_already_erased(info.completed) and not delete):
            continue
        report.erased.append(await erase_run(engine, info.stream, delete=delete, reason=reason))
    return report


async def apply_retention(engine: Engine, policies: Sequence[RetentionPolicy]) -> PurgeReport:
    """Evaluate `policies` (rate-limited per policy by its `interval_s`). Safe to call from any
    process at any time; `Engine.sweep()` calls it for you when `Engine(retention=...)` is set."""
    report = PurgeReport()
    due = [
        p
        for p in policies
        if time.monotonic() - engine._retention_seen.get(id(p), -1e18) >= p.interval_s
    ]
    if not due:
        return report
    for p in due:
        engine._retention_seen[id(p)] = time.monotonic()
    now = datetime.now(UTC)
    for info in await engine.retry(engine.store.run_index, "read"):
        if info.completed is None:
            continue
        policy = next((p for p in policies if p.matches(info.started.tenant)), None)
        if policy is None or policy not in due:
            continue
        if info.completed.time > now - policy.older_than:
            continue
        if _already_erased(info.completed) and not policy.delete:
            continue
        report.erased.append(
            await erase_run(engine, info.stream, delete=policy.delete, reason="retention")
        )
    return report
