"""Mem0 adapter (pip install mem0ai). Works with the open-source `AsyncMemory` and the hosted
`AsyncMemoryClient`; written against mem0ai 2.x, where entity ids travel inside `filters`.

    from mem0 import AsyncMemory
    memory = Mem0Memory(AsyncMemory())

Mem0 extracts and consolidates facts with ITS OWN llm/embedder (configure them on the Mem0 object);
that spend is outside the engine's budget and gateway, and it cannot de-duplicate by idempotency key,
so AAA treats its writes as `at_most_once` (a worker that dies mid-write is reported, not repeated).

Scoping: Mem0 requires an entity id on every call. A scope with a user or agent maps to the single
entity `user_id=<scope slug>` (tenant-qualified, so tenants cannot collide). Wider scopes (a whole
tenant) use Mem0's wildcard entity filter plus the `aaa_tenant` metadata this adapter writes; that
path is the least exercised, so test `forget(Scope(tenant=...))` against your Mem0 version."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope

_META = ("aaa_tenant", "aaa_user", "aaa_agent", "aaa_kind")


def _dt(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(v) if isinstance(v, str) and v else None
    except ValueError:
        return None


class Mem0Memory:
    idempotent = (
        False  # Mem0 decides ADD/UPDATE/DELETE/NONE itself; repeating a write is not a no-op
    )

    def __init__(self, client: Any, *, infer: bool = True, threshold: float | None = None) -> None:
        self.client, self.infer, self.threshold = client, infer, threshold
        # The hosted client and the OSS object differ in a few signatures.
        self.hosted = type(client).__module__.startswith("mem0.client")

    # -- scope mapping ------------------------------------------------------------------------
    @staticmethod
    def _exact(scope: Scope) -> bool:
        return bool(scope.user or scope.agent or scope.namespace)

    def _filters(self, scope: Scope, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        f: dict[str, Any] = dict(extra or {})
        if self._exact(scope):
            f["user_id"] = scope.slug()
        else:  # tenant-wide: any entity, restricted to this tenant by metadata
            f["user_id"] = "*"
            f["aaa_tenant"] = scope.tenant or "-"
        return f

    # -- Memory -------------------------------------------------------------------------------
    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        messages = [{"role": e.role or "user", "content": e.text} for e in entries]
        meta: dict[str, Any] = {
            "aaa_tenant": scope.tenant or "-",
            **({"aaa_user": scope.user} if scope.user else {}),
            **({"aaa_agent": scope.agent} if scope.agent else {}),
            **({"aaa_kind": entries[0].kind} if len({e.kind for e in entries}) == 1 else {}),
        }
        for e in entries:
            meta.update(e.meta)
        if (
            self.hosted
        ):  # the hosted v3 API takes identity inside `filters`, not as top-level fields
            res = await self.client.add(
                messages, filters={"user_id": scope.slug()}, metadata=meta, infer=self.infer
            )
        else:
            res = await self.client.add(
                messages, user_id=scope.slug(), metadata=meta, infer=self.infer
            )
        results = res.get("results", []) if isinstance(res, dict) else []
        pending = isinstance(res, dict) and (
            str(res.get("status", "")).upper() == "PENDING" or "event_id" in res
        )
        return AddResult(
            ids=[str(r["id"]) for r in results if r.get("id")],
            pending=pending,
            detail={k: v for k, v in (res or {}).items() if k != "results"},
        )

    def _item(self, d: dict[str, Any]) -> MemoryItem:
        meta = {k: v for k, v in (d.get("metadata") or {}).items() if k not in _META}
        return MemoryItem(
            id=str(d["id"]),
            text=str(d.get("memory") or d.get("text") or ""),
            kind=(d.get("metadata") or {}).get("aaa_kind", "fact"),
            score=d.get("score"),
            created_at=_dt(d.get("created_at")),
            meta=meta,
            raw=d,
        )

    async def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        kw: dict[str, Any] = {"top_k": k, "filters": self._filters(scope, filters)}
        if self.threshold is not None:
            kw["threshold"] = self.threshold
        res = await self.client.search(query, **kw)
        rows = res.get("results", []) if isinstance(res, dict) else (res or [])
        return [self._item(r) for r in rows][
            :k
        ]  # Mem0 has no point-in-time recall: `as_of` ignored

    async def update(self, memory_id: str, text: str, scope: Scope) -> MemoryItem:
        owned = {m.id for m in await self.list(scope, limit=1000)}
        if memory_id not in owned:
            raise KeyError(memory_id)
        res = await self.client.update(memory_id, text=text)
        return MemoryItem(id=memory_id, text=text, raw=res)

    async def delete(self, ids: Sequence[str], scope: Scope) -> int:
        owned = {m.id for m in await self.list(scope, limit=1000)}
        n = 0
        for i in ids:
            if i in owned:
                await self.client.delete(i)
                n += 1
        return n

    async def forget(self, scope: Scope) -> int:
        ids = [m.id for m in await self.list(scope, limit=10_000)]
        if self._exact(scope):
            if self.hosted:
                await self.client.delete_all(filters={"user_id": scope.slug()})
            else:
                await self.client.delete_all(user_id=scope.slug())
        else:
            for i in ids:  # wider than one entity: delete what the wildcard filter found
                await self.client.delete(i)
        return len(ids)

    async def list(self, scope: Scope, *, limit: int = 100) -> list[MemoryItem]:
        size = {"page_size": limit} if self.hosted else {"top_k": limit}
        res = await self.client.get_all(filters=self._filters(scope), **size)
        rows = res.get("results", []) if isinstance(res, dict) else (res or [])
        return [self._item(r) for r in rows][:limit]

    async def aclose(self) -> None:
        return None  # the caller owns the Mem0 object
