"""Supermemory adapter (hosted or self-hosted), over plain HTTP: no SDK dependency, because the SDK's
method names have changed between releases while the HTTP API is documented by an OpenAPI spec.

    memory = SupermemoryMemory(api_key=os.environ["SUPERMEMORY_API_KEY"])
    memory = SupermemoryMemory(api_key="...", base_url="http://localhost:6767")   # self-hosted

Two ways to write, chosen with `ingest=`:

* `"documents"` (default): `POST /v3/documents`. Supermemory extracts memories from the content with its
  own models, asynchronously (`AddResult.pending=True` until it is `done`; pass `wait=True` to poll).
  `customId` is derived from the idempotency key, so a repeated write replaces instead of duplicating.
* `"memories"`: `POST /v4/memories` stores the text verbatim as a memory, synchronously.

Scoping: one `containerTag` per (tenant, user, agent, namespace) (`Scope.slug()`, shortened to the
100-character limit). Tenant-wide `forget` deletes every container tag with the tenant's prefix, which
the API restricts to organisation owners/admins.

API versions: v3/v4 are announced to shut down on 2026-12-31 in favour of v5. Every path lives in
`PATHS`, so moving to v5 is one dict, not a rewrite."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

import httpx

from aaa.core.ids import stable_id
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope

PATHS = {
    "add_document": "/v3/documents",
    "get_document": "/v3/documents/{id}",
    "delete_document": "/v3/documents/{id}",
    "add_memories": "/v4/memories",
    "forget_memory": "/v4/memories",
    "update_memory": "/v4/memories",
    "list_memories": "/v4/memories/list",
    "search": "/v4/search",
    "profile": "/v4/profile",
    "container_tags": "/v3/container-tags/list",
    "delete_container": "/v3/container-tags/{tag}",
}
TAG_MAX = 100  # containerTag limit; allowed characters are [A-Za-z0-9_:-]


def _dt(v: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(v) if isinstance(v, str) else None
    except ValueError:
        return None


class SupermemoryMemory:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.supermemory.ai",
        ingest: Literal["documents", "memories"] = "documents",
        search_mode: Literal["memories", "hybrid", "documents"] = "memories",
        wait: bool = False,
        wait_timeout_s: float = 60.0,
        threshold: float = 0.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.http = client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0),
        )
        self._own = client is None
        self.ingest, self.search_mode, self.wait = ingest, search_mode, wait
        self.wait_timeout_s, self.threshold = wait_timeout_s, threshold
        # `customId` makes document ingestion idempotent; direct memory creation has no such key.
        self.idempotent = ingest == "documents"

    @staticmethod
    def tag(scope: Scope) -> str:
        return scope.slug(max_len=TAG_MAX)

    async def _tags(self, scope: Scope) -> list[str]:
        """The container tags a scope covers: its own, plus (for tenant- or user-wide scopes) every
        more specific tag that starts with it."""
        own = self.tag(scope)
        if scope.agent or scope.namespace:
            return [own]
        res = await self._call("GET", PATHS["container_tags"])
        listed = res if isinstance(res, list) else (res or {}).get("containerTags", [])
        names = [t if isinstance(t, str) else t.get("containerTag", "") for t in listed]
        prefix = scope.slug()
        return list(
            dict.fromkeys([own, *[t for t in names if t == prefix or t.startswith(prefix + "--")]])
        )

    async def _call(self, method: str, path: str, **kw: Any) -> Any:
        r = await self.http.request(method, path, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    # -- Memory -------------------------------------------------------------------------------
    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        tag = self.tag(scope)
        if self.ingest == "memories":
            body = {
                "containerTag": tag,
                "memories": [
                    {
                        "content": e.text,
                        "metadata": _meta(e),
                        **({"isStatic": True} if e.kind == "profile" else {}),
                    }
                    for e in entries
                ],
            }
            res = await self._call("POST", PATHS["add_memories"], json=body)
            return AddResult(ids=[m["id"] for m in (res or {}).get("memories", [])])
        ids: list[str] = []
        pending = False
        for i, e in enumerate(entries):
            custom = e.id or stable_id(tag, key or "", str(i), " ".join(e.text.lower().split()))
            content = f"{e.role}: {e.text}" if e.role else e.text
            res = await self._call(
                "POST",
                PATHS["add_document"],
                json={
                    "content": content,
                    "containerTag": tag,
                    "customId": custom,
                    "metadata": _meta(e),
                    **({"documentDate": e.time.isoformat()} if e.time else {}),
                },
            )
            doc_id = str(res["id"])
            ids.append(doc_id)
            if self.wait:
                await self._wait_done(doc_id)
            else:
                pending = True
        return AddResult(ids=ids, pending=pending)

    async def _wait_done(self, doc_id: str) -> None:
        async def poll() -> None:
            while True:
                doc = await self._call("GET", PATHS["get_document"].format(id=doc_id))
                status = (doc or {}).get("status")
                if status == "done":
                    return
                if status == "failed":
                    raise RuntimeError(f"supermemory failed to process document {doc_id}")
                await asyncio.sleep(0.5)

        await asyncio.wait_for(poll(), self.wait_timeout_s)

    @staticmethod
    def _item(r: dict[str, Any]) -> MemoryItem:
        meta = dict(r.get("metadata") or {})
        return MemoryItem(
            id=str(r["id"]),
            text=str(r.get("memory") or r.get("chunk") or ""),
            kind="document" if r.get("chunk") and not r.get("memory") else "fact",
            score=r.get("similarity"),
            created_at=_dt(r.get("createdAt")),
            valid_at=None,
            meta={**meta, **({"version": r["version"]} if r.get("version") is not None else {})},
            raw=r,
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
        tags = await self._tags(scope)
        body: dict[str, Any] = {
            "q": query,
            **({"containerTag": tags[0]} if len(tags) == 1 else {"containerTags": tags}),
            "limit": max(1, min(k, 100)),
            "searchMode": self.search_mode,
            "threshold": self.threshold,
        }
        if (
            filters
        ):  # filtered client-side: exact semantics without depending on QueryFilter's schema
            body["limit"] = 100
        res = await self._call("POST", PATHS["search"], json=body)
        items = [self._item(r) for r in (res or {}).get("results", [])]
        if filters:
            items = [m for m in items if all(m.meta.get(f) == v for f, v in filters.items())]
        return items[:k]

    async def update(self, memory_id: str, text: str, scope: Scope) -> MemoryItem:
        try:
            res = await self._call(
                "PATCH",
                PATHS["update_memory"],
                json={"id": memory_id, "containerTag": self.tag(scope), "newContent": text},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:  # not in this scope's container
                raise KeyError(memory_id) from e
            raise
        return MemoryItem(id=str((res or {}).get("id", memory_id)), text=text, raw=res)

    async def delete(self, ids: Sequence[str], scope: Scope) -> int:
        n = 0
        for i in ids:
            try:
                await self._call(
                    "DELETE",
                    PATHS["forget_memory"],
                    json={"id": i, "containerTag": self.tag(scope)},
                )
                n += 1
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in (404, 409):  # missing / already forgotten
                    raise
        return n

    async def forget(self, scope: Scope) -> int:
        removed = 0
        for tag in await self._tags(scope):
            try:
                out = await self._call("DELETE", PATHS["delete_container"].format(tag=tag))
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    continue
                raise
            removed += int((out or {}).get("deletedDocumentsCount", 0)) + int(
                (out or {}).get("deletedMemoriesCount", 0)
            )
        return removed

    async def list(self, scope: Scope, *, limit: int = 100) -> list[MemoryItem]:
        res = await self._call(
            "POST",
            PATHS["list_memories"],
            json={"containerTags": await self._tags(scope), "limit": limit},
        )
        return [self._item(r) for r in (res or {}).get("memoryEntries", [])][:limit]

    async def profile(self, scope: Scope) -> dict[str, Any]:
        res = await self._call("POST", PATHS["profile"], json={"containerTag": self.tag(scope)})
        return dict((res or {}).get("profile", {}))

    async def aclose(self) -> None:
        if self._own:
            await self.http.aclose()


def _meta(e: MemoryEntry) -> dict[str, Any]:
    """Supermemory metadata values may only be string, number, boolean or list of strings."""
    out: dict[str, Any] = {"aaa_kind": e.kind}
    for k, v in e.meta.items():
        out[k] = v if isinstance(v, str | int | float | bool) else str(v)
    return out
