"""A memory built on AAA's own `Index`: no external service, works offline, the reference
implementation the contract tests are written against.

    memory = NativeMemory()                                   # keyword search
    memory = NativeMemory(embedder=LLMEmbedder(client))       # hybrid keyword + vector
    memory = NativeMemory(extractor=llm_extractor(client))    # remember distilled facts, not raw chat

It is in-process (the ids it needs for `forget`/`list` are tracked in memory), so use it for
development, tests and single-process apps; use a service-backed adapter for durable, shared memory."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from aaa.core.ids import stable_id
from aaa.data.capabilities import Embedder
from aaa.data.index import Doc, Index, InMemoryIndex
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, Message
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope

Extractor = Callable[[Sequence[MemoryEntry], Scope], Awaitable[list[MemoryEntry]]]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class NativeMemory:
    idempotent = True  # ids are derived from content, so repeating a write changes nothing

    def __init__(
        self,
        index: Index | None = None,
        *,
        embedder: Embedder | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self.index: Index = index or InMemoryIndex(embedder)
        self.extractor = extractor
        self._store: dict[str, tuple[Scope, Doc]] = {}  # in-process: see the module docstring

    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        todo = await self.extractor(entries, scope) if self.extractor else list(entries)
        docs: list[Doc] = []
        for e in todo:
            if not e.text.strip():
                continue
            mid = e.id or stable_id(scope.key(), _norm(e.text))
            docs.append(
                Doc(
                    id=mid,
                    text=e.text,
                    meta={
                        **e.meta,
                        "kind": e.kind,
                        "created_at": (e.time or datetime.now(UTC)).isoformat(),
                        "valid_at": e.time.isoformat() if e.time else None,
                    },
                )
            )
            self._store[mid] = (scope, docs[-1])
        if docs:
            await self.index.upsert(docs)
        return AddResult(ids=[d.id for d in docs])

    def _item(self, doc: Doc, score: float | None = None) -> MemoryItem:
        m = doc.meta
        parse = lambda v: datetime.fromisoformat(v) if v else None
        return MemoryItem(
            id=doc.id,
            text=doc.text,
            kind=m.get("kind", "fact"),
            score=score,
            created_at=parse(m.get("created_at")),
            valid_at=parse(m.get("valid_at")),
            invalid_at=parse(m.get("invalid_at")),
            meta={
                k: v
                for k, v in m.items()
                if k not in ("kind", "created_at", "valid_at", "invalid_at")
            },
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
        def visible(d: Doc) -> bool:
            owner = self._store.get(d.id)
            if owner is None or not scope.contains(owner[0]):
                return False
            return not (filters and any(d.meta.get(f) != v for f, v in filters.items()))

        out: list[MemoryItem] = []
        for hit in await self.index.search(query, k=k * 4 if as_of else k, filter=visible):
            item = self._item(hit.doc, hit.score)
            if as_of is None or item.valid_on(as_of):
                out.append(item)
        return out[:k]

    async def update(self, memory_id: str, text: str, scope: Scope) -> MemoryItem:
        entry = self._store.get(memory_id)
        if entry is None or not scope.contains(entry[0]):
            raise KeyError(memory_id)
        doc = Doc(id=memory_id, text=text, meta=dict(entry[1].meta))
        await self.index.upsert([doc])
        self._store[memory_id] = (entry[0], doc)
        return self._item(doc)

    async def delete(self, ids: Sequence[str], scope: Scope) -> int:
        mine = [i for i in ids if i in self._store and scope.contains(self._store[i][0])]
        if mine:
            await self.index.delete(mine)
            for i in mine:
                del self._store[i]
        return len(mine)

    async def forget(self, scope: Scope) -> int:
        return await self.delete(
            [i for i, (o, _) in self._store.items() if scope.contains(o)], scope
        )

    async def list(self, scope: Scope, *, limit: int = 100) -> list[MemoryItem]:
        mine = [d for o, d in self._store.values() if scope.contains(o)]
        return [self._item(d) for d in mine[:limit]]

    async def aclose(self) -> None:
        return None


_EXTRACT_PROMPT = (
    "Extract durable facts worth remembering about the user from the conversation: preferences, "
    "identity, constraints, decisions, commitments. Ignore small talk and anything temporary. "
    'Reply with ONLY a JSON array of short, self-contained statements, e.g. ["Prefers vegetarian food"]. '
    "Reply [] if there is nothing worth remembering."
)


def llm_extractor(client: LLMClient, model: str | None = None) -> Extractor:
    """An `extractor` that distils a conversation into facts with one LLM call.

    Note this calls `client` directly: it is not routed through the engine's `LLMHandler`, so its
    tokens are not in the run's budget ledger. If that matters, do the extraction as an ordinary
    step of your algorithm and pass the resulting facts to `add`."""

    async def extract(entries: Sequence[MemoryEntry], scope: Scope) -> list[MemoryEntry]:
        convo = "\n".join(f"{e.role or 'note'}: {e.text}" for e in entries)
        resp = await client.complete(
            LLMRequest(
                model=model,
                messages=[
                    Message(role="system", content=_EXTRACT_PROMPT),
                    Message(role="user", content=convo),
                ],
            )
        )
        raw = resp.message.content or "[]"
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        try:
            facts = json.loads(m.group(0)) if m else []
        except ValueError:
            facts = []
        return [MemoryEntry(text=str(f)) for f in facts if isinstance(f, str) and f.strip()]

    return extract
