"""Wrap ANY memory library by handing over its functions.

    memory = CallableMemory(
        add=lambda entries, scope: my_lib.store([e.text for e in entries], user=scope.user),
        search=lambda query, scope, k: my_lib.find(query, user=scope.user, limit=k),
    )

Functions may be sync or async. Returns are forgiving: `add` may return nothing, a list of ids or an
`AddResult`; `search` may return `MemoryItem`s, dicts (`text`/`memory`/`fact`/`content`, optional
`id`/`score`) or plain strings. Optional functions (`update`, `delete`, `forget`, `profile`, `list`)
switch the matching capability on, so `isinstance(memory, Forgettable)` is true exactly when you
supplied `forget`."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any

from aaa.core.ids import stable_id
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope

_TEXT_KEYS = ("text", "memory", "fact", "content", "value")


async def _call(fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
    r = fn(*args, **kw)
    return await r if inspect.isawaitable(r) else r


def coerce_item(x: Any, position: int = 0) -> MemoryItem:
    if isinstance(x, MemoryItem):
        return x
    if isinstance(x, str):
        return MemoryItem(id=stable_id("callable", x), text=x)
    if isinstance(x, dict):
        text = next((str(x[k]) for k in _TEXT_KEYS if x.get(k) is not None), None)
        if text is None:
            raise ValueError(f"cannot find the text of a memory in {sorted(x)}")
        rest = {k: v for k, v in x.items() if k not in (*_TEXT_KEYS, "id", "score")}
        return MemoryItem(
            id=str(x.get("id") or stable_id("callable", text)),
            text=text,
            score=x.get("score"),
            meta=rest,
            raw=x,
        )
    raise TypeError(
        f"a search function must return MemoryItem | dict | str, got {type(x).__name__}"
    )


class CallableMemory:
    def __init__(
        self,
        add: Callable[..., Any],
        search: Callable[..., Any],
        *,
        update: Callable[..., Any] | None = None,
        delete: Callable[..., Any] | None = None,
        forget: Callable[..., Any] | None = None,
        profile: Callable[..., Any] | None = None,
        list: Callable[..., Any] | None = None,
        aclose: Callable[[], Awaitable[None] | None] | None = None,
        idempotent: bool = False,
    ) -> None:
        self._add, self._search, self._aclose = add, search, aclose
        self.idempotent = idempotent  # set True only if your `add` honours the `key` it is given
        self._accepts_key = "key" in inspect.signature(add).parameters
        if update:
            self.update = self._wrap_update(update)  # type: ignore[method-assign]
        if delete:
            self.delete = self._wrap_delete(delete)  # type: ignore[method-assign]
        if forget:
            self.forget = self._wrap_forget(forget)  # type: ignore[method-assign]
        if profile:
            self.profile = self._wrap_profile(profile)  # type: ignore[method-assign]
        if list:
            self.list = self._wrap_list(list)  # type: ignore[method-assign]

    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        kw = {"key": key} if self._accepts_key else {}
        r = await _call(self._add, list(entries), scope, **kw)
        if isinstance(r, AddResult):
            return r
        if r is None:
            return AddResult()
        return AddResult(ids=[str(i) for i in r] if isinstance(r, list | tuple) else [str(r)])

    async def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        params = inspect.signature(self._search).parameters
        extra = {n: v for n, v in (("filters", filters), ("as_of", as_of)) if n in params}
        r = await _call(self._search, query, scope, k, **extra)
        items = [coerce_item(x, i) for i, x in enumerate(r or [])]
        if as_of is not None and "as_of" not in params:
            items = [m for m in items if m.valid_on(as_of)]
        return items[:k]

    async def aclose(self) -> None:
        if self._aclose is not None:
            await _call(self._aclose)

    @staticmethod
    def _wrap_update(fn: Callable[..., Any]) -> Callable[..., Awaitable[MemoryItem]]:
        async def update(memory_id: str, text: str, scope: Scope) -> MemoryItem:
            r = await _call(fn, memory_id, text, scope)
            return coerce_item(r) if r is not None else MemoryItem(id=memory_id, text=text)

        return update

    @staticmethod
    def _wrap_delete(fn: Callable[..., Any]) -> Callable[..., Awaitable[int]]:
        async def delete(ids: Sequence[str], scope: Scope) -> int:
            r = await _call(fn, list(ids), scope)
            return len(ids) if r is None or r is True else int(r)

        return delete

    @staticmethod
    def _wrap_forget(fn: Callable[..., Any]) -> Callable[..., Awaitable[int]]:
        async def forget(scope: Scope) -> int:
            r = await _call(fn, scope)
            return 0 if r is None or isinstance(r, bool) else int(r)

        return forget

    @staticmethod
    def _wrap_profile(fn: Callable[..., Any]) -> Callable[..., Awaitable[dict[str, Any]]]:
        async def profile(scope: Scope) -> dict[str, Any]:
            r = await _call(fn, scope)
            return r if isinstance(r, dict) else {"summary": str(r)}

        return profile

    @staticmethod
    def _wrap_list(fn: Callable[..., Any]) -> Callable[..., Awaitable[list[MemoryItem]]]:
        async def list_(scope: Scope, *, limit: int = 100) -> list[MemoryItem]:
            r = await _call(fn, scope, limit)
            return [coerce_item(x) for x in (r or [])][:limit]

        return list_
