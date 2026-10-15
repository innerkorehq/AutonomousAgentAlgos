"""Text in, text out: `EmbeddingIndex` makes any `VectorStore` an `aaa.data.Index`, so `RAGAgent`, `search_tool` and `NativeMemory`
work on a persistent store unchanged."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from aaa.data.capabilities import Embedder
from aaa.data.index import Doc, Hit
from aaa.vector.filters import Filter
from aaa.vector.types import (
    DEFAULT_NAMESPACE,
    DeleteWhere,
    DropNamespace,
    Match,
    Record,
    VectorError,
    VectorStore,
)

TENANT_KEY = "tenant"


class EmbeddingIndex:
    """`EmbeddingIndex(PgVectorStore(...), LLMEmbedder(client))`.

    `search(query, filter=...)` takes either a `Filter` / dict (evaluated by the store) or, for compatibility with `InMemoryIndex`,
    a Python predicate over the `Doc`. A predicate cannot be pushed down, so the index fetches more candidates and keeps asking until
    it has `k` survivors or the store is exhausted (bounded by `max_overfetch`); prefer a `Filter` on a large index."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        namespace: str | None = None,
        max_overfetch: int = 1000,
    ) -> None:
        self.store, self.embedder, self.namespace, self.max_overfetch = (
            store,
            embedder,
            namespace,
            max_overfetch,
        )

    async def upsert(self, docs: list[Doc]) -> None:
        if not docs:
            return
        need = [d for d in docs if d.embedding is None]
        if need:
            for d, v in zip(need, await self.embedder.embed([d.text for d in need]), strict=True):
                d.embedding = v
        await self.store.upsert(
            [Record(d.id, list(d.embedding or []), d.text, dict(d.meta)) for d in docs],
            namespace=self.namespace,
        )

    async def delete(self, ids: list[str]) -> None:
        await self.store.delete(ids, namespace=self.namespace)

    async def count(self) -> int:
        return await self.store.count(namespace=self.namespace)

    def _hit(self, m: Match) -> Hit:
        return Hit(
            doc=Doc(id=m.id, text=m.text, meta=m.meta),
            score=m.score,
            metric=self.store.metric,
            higher_is_better=self.store.higher_is_better,
        )

    async def search(
        self,
        query: str,
        k: int = 5,
        *,
        filter: Callable[[Doc], bool] | Filter | dict[str, Any] | None = None,
    ) -> list[Hit]:
        qv = (await self.embedder.embed([query]))[0]
        if filter is None or isinstance(filter, Filter | dict):
            ms = await self.store.query(
                qv, k, filter=Filter.parse(filter), namespace=self.namespace
            )
            return [self._hit(m) for m in ms]
        fetch = max(k * 3, k + 10)
        while True:
            ms = await self.store.query(qv, fetch, namespace=self.namespace)
            kept = [h for m in ms if filter((h := self._hit(m)).doc)]
            if len(kept) >= k or len(ms) < fetch or fetch >= self.max_overfetch:
                return kept[:k]
            fetch = min(fetch * 4, self.max_overfetch)

    async def aclose(self) -> None:
        await self.store.aclose()


class TenantVectorStore:
    """A view of a store for ONE tenant, safe to hand to code that handles that tenant's requests.

    Two independent walls: every call is pinned to the namespace `<prefix><tenant>` (a hard partition), AND every record carries
    `meta["tenant"]` and every query/delete is ANDed with `tenant == <tenant>` *outside* whatever filter the caller passes, so an `or` in
    the caller's filter cannot reach another tenant's rows. Passing a different namespace, or a record claiming another tenant, is an error."""

    def __init__(self, store: VectorStore, tenant: str, *, namespace_prefix: str = "") -> None:
        if not tenant:
            raise ValueError("tenant must be a non-empty string")
        self.store, self.tenant = store, tenant
        self.namespace = f"{namespace_prefix}{tenant}"
        self.metric, self.dimension = store.metric, store.dimension

    @property
    def higher_is_better(self) -> bool:
        return self.store.higher_is_better

    def _ns(self, namespace: str | None) -> str:
        if namespace not in (None, self.namespace):
            raise VectorError(
                f"tenant view of {self.tenant!r} cannot address namespace {namespace!r}"
            )
        return self.namespace

    def _filter(self, f: Filter | None) -> Filter:
        from aaa.vector.filters import And, Eq

        wall = Eq(TENANT_KEY, self.tenant)
        return wall if f is None else And((wall, f))

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        out = []
        for r in records:
            if r.meta.get(TENANT_KEY, self.tenant) != self.tenant:
                raise VectorError(f"record {r.id!r} claims tenant {r.meta[TENANT_KEY]!r}")
            out.append(Record(r.id, r.vector, r.text, {**r.meta, TENANT_KEY: self.tenant}))
        await self.store.upsert(out, namespace=self._ns(namespace))
        self.dimension = self.store.dimension

    async def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        return await self.store.query(
            vector, k, filter=self._filter(filter), namespace=self._ns(namespace)
        )

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        return [
            r
            for r in await self.store.get(ids, namespace=self._ns(namespace))
            if r.meta.get(TENANT_KEY) == self.tenant
        ]

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        mine = [
            r.id for r in await self.get(ids, namespace=namespace)
        ]  # never delete an id that is not this tenant's
        if mine:
            await self.store.delete(mine, namespace=self._ns(namespace))

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int:
        if not isinstance(self.store, DeleteWhere):
            raise VectorError(f"{type(self.store).__name__} cannot delete by filter")
        return await self.store.delete_where(self._filter(filter), namespace=self._ns(namespace))

    async def erase(self) -> None:
        """Remove everything of this tenant (offboarding / erasure)."""
        if isinstance(self.store, DropNamespace):
            await self.store.drop_namespace(self.namespace)
        else:
            await self.delete_where(self._filter(None))

    async def count(self, *, namespace: str | None = None) -> int:
        return await self.store.count(namespace=self._ns(namespace))

    async def aclose(self) -> None:
        return None  # the underlying store is shared; its owner closes it


__all__ = ["DEFAULT_NAMESPACE", "EmbeddingIndex", "TenantVectorStore"]
