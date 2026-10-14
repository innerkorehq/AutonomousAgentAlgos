"""LangGraph store adapter: any `langgraph.store.base.BaseStore` (InMemoryStore, the Postgres/Redis
stores, a hosted LangGraph store) becomes an AAA memory; this is also how LangMem is supported,
because LangMem keeps its memories in a `BaseStore`.

    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
    memory = LangGraphStoreMemory(store)

Namespaces are `(root, tenant, user, agent, *namespace)` built from `Scope.slug()` parts, so a
tenant-wide search is a prefix search. To let LangMem's own tools/managers read and write the same
memories, give them `memory.namespace_for(scope)`:

    create_search_memory_tool(namespace=memory.namespace_for(scope), store=store)

LangMem's records (`{"kind": "Memory", "content": {"content": ...}}`) are read transparently.
A store without an embedding index ranks by word overlap instead (no semantic search); configure the
store's `index=` for real semantic recall. Writes are idempotent (the key is derived from the text)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from aaa.core.ids import stable_id
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, MemoryKind, Scope, encode_part

_WORD = re.compile(r"\w+")
PAGE = 200


def _text_of(value: dict[str, Any]) -> str:
    """Our records store `text`; LangMem's store `content` (a string or `{"content": str}`)."""
    if isinstance(value.get("text"), str):
        return value["text"]
    c = value.get("content")
    if isinstance(c, dict):
        c = c.get("content")
    return c if isinstance(c, str) else ""


class LangGraphStoreMemory:
    idempotent = True

    def __init__(self, store: Any, *, root: str = "memories") -> None:
        self.store, self.root = store, root

    def namespace_for(self, scope: Scope) -> tuple[str, ...]:
        """The store namespace holding exactly this scope's memories (also what LangMem should use)."""
        parts: list[str] = [self.root, encode_part(scope.tenant) if scope.tenant else "_"]
        if scope.user or scope.agent or scope.namespace:
            parts.append(encode_part(scope.user) if scope.user else "_")
        if scope.agent or scope.namespace:
            parts.append(encode_part(scope.agent) if scope.agent else "_")
        parts += [encode_part(n) for n in scope.namespace]
        return tuple(parts)

    async def _items(
        self, scope: Scope, *, query: str | None = None, limit: int = 100
    ) -> list[Any]:
        """Everything under the scope's namespace prefix (paged), optionally ranked by `query`."""
        out: list[Any] = []
        offset = 0
        while len(out) < limit:
            page = await self.store.asearch(
                self.namespace_for(scope),
                query=query,
                limit=min(PAGE, limit - len(out)),
                offset=offset,
            )
            out += page
            if len(page) < PAGE:
                break
            offset += PAGE
        return out

    @staticmethod
    def _item(it: Any, score: float | None = None) -> MemoryItem:
        v = dict(it.value)
        return MemoryItem(
            id=str(it.key),
            text=_text_of(v),
            kind=_kind(v.get("kind")),
            score=score if score is not None else getattr(it, "score", None),
            created_at=getattr(it, "created_at", None),
            valid_at=_parse(v.get("valid_at")),
            invalid_at=_parse(v.get("invalid_at")),
            meta=dict(v.get("meta") or {}),
            raw=it,
        )

    # -- Memory -------------------------------------------------------------------------------
    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        ns = self.namespace_for(scope)
        ids: list[str] = []
        for e in entries:
            mid = e.id or stable_id("/".join(ns), " ".join(e.text.lower().split()))
            await self.store.aput(
                ns,
                mid,
                {
                    "text": e.text,
                    "kind": e.kind,
                    "role": e.role,
                    "meta": e.meta,
                    "valid_at": e.time.isoformat() if e.time else None,
                    "written_at": datetime.now(UTC).isoformat(),
                },
            )
            ids.append(mid)
        return AddResult(ids=ids)

    async def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        raw = await self._items(scope, query=query, limit=max(k * 4, 20))
        items = [self._item(it) for it in raw]
        if all(m.score is None for m in items):  # no embedding index: rank by word overlap instead
            q = {w.lower() for w in _WORD.findall(query)}
            scored = [(len(q & {w.lower() for w in _WORD.findall(m.text)}), m) for m in items]
            items = [
                m.model_copy(update={"score": float(n)})
                for n, m in sorted(scored, key=lambda t: -t[0])
                if n
            ]
        if as_of is not None:
            items = [m for m in items if m.valid_on(as_of)]
        if filters:
            items = [m for m in items if all(m.meta.get(f) == v for f, v in filters.items())]
        return items[:k]

    async def update(self, memory_id: str, text: str, scope: Scope) -> MemoryItem:
        for it in await self._items(scope, limit=10_000):
            if it.key == memory_id:
                await self.store.aput(tuple(it.namespace), memory_id, {**it.value, "text": text})
                return self._item(it).model_copy(update={"text": text})
        raise KeyError(memory_id)

    async def delete(self, ids: Sequence[str], scope: Scope) -> int:
        wanted = set(ids)
        n = 0
        for it in await self._items(scope, limit=10_000):
            if it.key in wanted:
                await self.store.adelete(tuple(it.namespace), it.key)
                n += 1
        return n

    async def forget(self, scope: Scope) -> int:
        n = 0
        for it in await self._items(scope, limit=1_000_000):
            await self.store.adelete(tuple(it.namespace), it.key)
            n += 1
        return n

    async def list(self, scope: Scope, *, limit: int = 100) -> list[MemoryItem]:
        return [self._item(it) for it in await self._items(scope, limit=limit)]

    async def aclose(self) -> None:
        return None  # the caller owns the store


def _parse(v: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(v) if isinstance(v, str) and v else None
    except ValueError:
        return None


def _kind(v: Any) -> MemoryKind:
    return v if v in ("fact", "episode", "profile", "procedure", "document") else "fact"
