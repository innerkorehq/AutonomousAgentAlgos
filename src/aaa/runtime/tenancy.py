"""Multi-tenancy: one engine and store serving many customers without letting them see each other.

A `TenantScope` is a view of an engine bound to one tenant. Everything it starts is tagged with the
tenant (inherited by every derived event, including child runs), run ids are namespaced
(`"<tenant>/<id>"`) so two tenants may both use the id `order-1`, and every lookup checks ownership:
asking for another tenant's run behaves exactly like asking for a run that does not exist."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from aaa.core.errors import TenantMismatch
from aaa.core.ids import new_id
from aaa.events import RunInfo
from aaa.runtime.budget import Budget
from aaa.runtime.retention import ErasureReport, MemoryReport, PurgeReport

if TYPE_CHECKING:
    from aaa.runtime.engine import Algorithm, Engine, RunHandle


def valid_tenant(tenant: str) -> str:
    if not tenant or "/" in tenant or tenant != tenant.strip():
        raise ValueError(f"invalid tenant {tenant!r}: must be non-empty and contain no '/'")
    return tenant


class TenantScope:
    def __init__(self, engine: Engine, tenant: str) -> None:
        self.engine, self.tenant = engine, valid_tenant(tenant)

    def run_id(self, run_id: str) -> str:
        """The namespaced stream id for one of this tenant's own run ids (idempotent)."""
        prefix = f"{self.tenant}/"
        return run_id if run_id.startswith(prefix) else prefix + run_id

    async def start(
        self,
        algo: Algorithm,
        input: dict[str, Any],
        *,
        run_id: str | None = None,
        budget: Budget | None = None,
        user: str | None = None,
        **kw: Any,
    ) -> RunHandle:
        rid = self.run_id(run_id or new_id())
        if user is not None:  # who the run is for: memory is scoped by it (never by model output)
            kw["meta"] = {**(kw.get("meta") or {}), "user": user}
        if kw.get("parent"):
            kw["parent"] = await self._owned(kw["parent"])
        handle = await self.engine.start(
            algo, input, run_id=rid, budget=budget, tenant=self.tenant, **kw
        )
        await self._owned(handle.run_id)  # an existing run id must also be ours
        return handle

    async def run(self, algo: Algorithm, input: dict[str, Any], **kw: Any) -> Any:
        return await (await self.start(algo, input, **kw)).result()

    async def _owned(self, run_id: str) -> str:
        rid = self.run_id(run_id)
        try:
            started = await self.engine._started_of(rid)
        except KeyError:
            raise KeyError(f"no run {run_id!r}") from None
        if started.tenant != self.tenant:
            raise KeyError(f"no run {run_id!r}") from TenantMismatch(rid)
        return rid

    async def handle(self, run_id: str) -> RunHandle:
        """A handle to one of this tenant's runs. KeyError if it does not exist OR belongs to someone else."""
        return self.engine.handle(await self._owned(run_id))

    async def runs(self, *, open_only: bool = False) -> list[RunInfo]:
        infos = await self.engine.retry(self.engine.store.run_index, "read")
        return [
            r
            for r in infos
            if r.started.tenant == self.tenant and not (open_only and r.completed is not None)
        ]

    async def erase(self, run_id: str, **kw: Any) -> ErasureReport:
        return await self.engine.erase(await self._owned(run_id), **kw)

    async def purge(self, *, older_than: timedelta, **kw: Any) -> PurgeReport:
        """Erase this tenant's finished runs older than `older_than` (never another tenant's)."""
        return await self.engine.purge(older_than=older_than, tenant=self.tenant, **kw)

    async def erase_all(self, *, memory: bool = True, **kw: Any) -> PurgeReport:
        """Right-to-erasure for the whole tenant: every FINISHED run now, and (unless `memory=False`)
        everything the engine's memory backends hold for the tenant. Open runs are reported in
        `skipped_open`; cancel them first, then call again."""
        report = await self.engine.purge(older_than=timedelta(0), tenant=self.tenant, **kw)
        if memory and self.engine.memories:
            report.memory = await self.forget_memory()
        return report

    async def forget_memory(
        self, *, user: str | None = None, agent: str | None = None
    ) -> MemoryReport:
        """Erase this tenant's long-term memories (or one user's / agent's) from every backend."""
        from aaa.memory.types import Scope

        return await self.engine.forget_memory(Scope(tenant=self.tenant, user=user, agent=agent))
