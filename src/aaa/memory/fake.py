"""An in-process memory with every capability, for tests and offline examples (the analogue of
`aaa.llm.fake`). Retrieval is word overlap, so tests are deterministic and need no model."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from aaa.core.ids import stable_id
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope

_WORD = re.compile(r"\w+")


def _words(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s)}


class FakeMemory:
    """Scoped, idempotent, fully featured. `calls` records every operation for assertions."""

    idempotent = True

    def __init__(self) -> None:
        self._items: dict[str, tuple[Scope, MemoryItem]] = {}
        self._keys: dict[str, AddResult] = {}
        self.calls: list[tuple[str, Any]] = []

    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        self.calls.append(("add", [e.text for e in entries]))
        if key is not None and key in self._keys:
            return self._keys[key]
        ids: list[str] = []
        for e in entries:
            mid = e.id or stable_id(scope.key(), " ".join(e.text.lower().split()))
            self._items.setdefault(
                mid,
                (
                    scope,
                    MemoryItem(
                        id=mid,
                        text=e.text,
                        kind=e.kind,
                        created_at=e.time or datetime.now(UTC),
                        valid_at=e.time,
                        meta=dict(e.meta),
                    ),
                ),
            )
            ids.append(mid)
        result = AddResult(ids=ids)
        if key is not None:
            self._keys[key] = result
        return result

    def _visible(self, scope: Scope) -> list[MemoryItem]:
        return [item for owner, item in self._items.values() if scope.contains(owner)]

    async def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        self.calls.append(("search", query))
        q = _words(query)
        scored: list[MemoryItem] = []
        for item in self._visible(scope):
            if as_of is not None and not item.valid_on(as_of):
                continue
            if filters and any(item.meta.get(f) != v for f, v in filters.items()):
                continue
            overlap = len(q & _words(item.text))
            if overlap:
                scored.append(item.model_copy(update={"score": float(overlap)}))
        scored.sort(key=lambda m: m.score or 0.0, reverse=True)
        return scored[:k]

    async def update(self, memory_id: str, text: str, scope: Scope) -> MemoryItem:
        self.calls.append(("update", memory_id))
        owner, item = self._items[memory_id]
        if not scope.contains(owner):
            raise KeyError(memory_id)
        item = item.model_copy(update={"text": text})
        self._items[memory_id] = (owner, item)
        return item

    async def delete(self, ids: Sequence[str], scope: Scope) -> int:
        self.calls.append(("delete", list(ids)))
        gone = [i for i in ids if i in self._items and scope.contains(self._items[i][0])]
        for i in gone:
            del self._items[i]
        return len(gone)

    async def forget(self, scope: Scope) -> int:
        self.calls.append(("forget", scope.key()))
        gone = [i for i, (owner, _) in self._items.items() if scope.contains(owner)]
        for i in gone:
            del self._items[i]
        return len(gone)

    async def list(self, scope: Scope, *, limit: int = 100) -> list[MemoryItem]:
        return self._visible(scope)[:limit]

    async def profile(self, scope: Scope) -> dict[str, Any]:
        return {"facts": [m.text for m in self._visible(scope)]}

    async def aclose(self) -> None:
        return None
