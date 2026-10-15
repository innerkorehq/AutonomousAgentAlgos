# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Pinecone (serverless or pod indexes) through the official async client.

    store = PineconeStore(api_key="...", index_name="docs", dimension=1536)                 # looks the index host up
    store = PineconeStore(api_key="...", index_host="docs-abc.svc.pinecone.io", dimension=1536)   # no control-plane call
    await store.setup()                                  # create_index(...) if it does not exist (optional; needs `create=True`)

Namespaces are Pinecone's own (a hard partition inside the index). Pinecone metadata is flat, may not contain null, and is limited to 40 KB per record
(the text is stored in it too), so nested objects are **flattened to dotted keys for filtering** and the exact original is kept as JSON, which is what
`query`/`get` return; a record over the limit is rejected up front with a clear error.

Refused with `UnsupportedFilter`: ranges over strings, equality with `null`, negated ranges (Pinecone's filters have no NOT). Not provided: `delete_where`
(Pinecone serverless cannot delete by metadata filter) -- `drop_namespace` (delete everything in a namespace) is.

Writes in Pinecone serverless are eventually consistent. With `consistent=True` (the default) `upsert` and `delete` wait until the change is visible to `fetch`,
so a following query sees it (bounded by `consistency_timeout_s`); set it to False for bulk loads.

Scores: cosine -> similarity, dotproduct -> inner product, euclidean -> the distance Pinecone reports (Pinecone Local reports it *squared*; lower is better).
Verified against Pinecone Local (the official emulator), not a live Pinecone account."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

from aaa.vector._flat import flatten
from aaa.vector._flat import to_where as _to_where
from aaa.vector.filters import Filter
from aaa.vector.types import DEFAULT_NAMESPACE, Match, Metric, Record, VectorError, check_vector

META_KEY, TEXT_KEY = "_aaa_meta", "_aaa_text"
NEVER = {"_aaa_never": {"$exists": True}}  # no record has this key
MAX_METADATA_BYTES = 38_000  # Pinecone's limit is 40 KB per record; keep headroom for keys


def to_filter(f: Filter) -> dict[str, Any] | None:
    return _to_where(f, store="Pinecone", supports_exists=True, never=NEVER)


class PineconeStore:
    def __init__(
        self,
        *,
        dimension: int,
        api_key: str | None = None,
        index_name: str | None = None,
        index_host: str | None = None,
        metric: Metric = "cosine",
        create: bool = False,
        cloud: str = "aws",
        region: str = "us-east-1",
        consistent: bool = True,
        consistency_timeout_s: float = 30.0,
        control_host: str | None = None,
        client: Any = None,
    ) -> None:
        if not (index_name or index_host):
            raise ValueError("pass index_name= or index_host=")
        self.dimension: int | None = dimension
        self.metric: Metric = metric
        self.api_key, self.index_name, self.index_host = api_key, index_name, index_host
        self.create, self.cloud, self.region = create, cloud, region
        self.consistent, self.consistency_timeout_s, self.control_host = (
            consistent,
            consistency_timeout_s,
            control_host,
        )
        self._pc: Any = client
        self._own = client is None
        self._ix: Any = None

    @property
    def higher_is_better(self) -> bool:
        return self.metric != "euclidean"

    async def _client(self) -> Any:
        if self._pc is None:
            from pinecone import PineconeAsyncio

            kw: dict[str, Any] = {"api_key": self.api_key}
            if self.control_host:
                kw["host"] = self.control_host
            self._pc = PineconeAsyncio(**kw)
        return self._pc

    async def setup(self) -> None:
        """Create the index if `create=True` and it does not exist yet (needs the control plane and a matching SDK/API version)."""
        pc = await self._client()
        if self.create and self.index_name and not await pc.has_index(self.index_name):
            from pinecone import ServerlessSpec

            pm = {"cosine": "cosine", "dot": "dotproduct", "euclidean": "euclidean"}[self.metric]
            await pc.create_index(
                name=self.index_name,
                dimension=int(self.dimension or 0),
                metric=pm,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

    async def _index(self) -> Any:
        if self._ix is None:
            pc = await self._client()
            host = self.index_host
            if host is None:
                host = (await pc.describe_index(self.index_name)).host
            self._ix = pc.IndexAsyncio(host=host)
        return self._ix

    @staticmethod
    def _ns(namespace: str | None) -> str:
        return namespace or DEFAULT_NAMESPACE

    async def _await_visible(self, ids: Sequence[str], namespace: str, present: bool) -> None:
        ix = await self._index()
        deadline = time.monotonic() + self.consistency_timeout_s
        delay = 0.05
        while True:
            got = (await ix.fetch(ids=list(ids), namespace=namespace)).vectors
            if (len(got) == len(set(ids))) if present else (len(got) == 0):
                return
            if time.monotonic() > deadline:
                raise VectorError(
                    f"Pinecone did not make the write visible within {self.consistency_timeout_s}s (eventual consistency)"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        for r in records:
            check_vector(self.dimension, r.vector)
        if not records:
            return
        vectors = []
        for r in records:
            meta = {**flatten(r.meta), META_KEY: json.dumps(r.meta), TEXT_KEY: r.text}
            if len(json.dumps(meta).encode()) > MAX_METADATA_BYTES:
                raise VectorError(
                    f"record {r.id!r}: text plus metadata exceed Pinecone's 40 KB metadata limit"
                )
            vectors.append({"id": r.id, "values": [float(x) for x in r.vector], "metadata": meta})
        ns = self._ns(namespace)
        ix = await self._index()
        await ix.upsert(vectors=vectors, namespace=ns, show_progress=False)
        if self.consistent:
            await self._await_visible([r.id for r in records], ns, present=True)

    @staticmethod
    def _unpack(meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return meta.get(TEXT_KEY, ""), json.loads(meta.get(META_KEY, "{}"))

    async def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        check_vector(self.dimension, vector, "query vector")
        flt = (
            to_filter(filter) if filter is not None else None
        )  # compile first: refusals happen before any I/O
        ix = await self._index()
        r = await ix.query(
            vector=[float(x) for x in vector],
            top_k=min(int(k), 10_000),
            namespace=self._ns(namespace),
            filter=flt,
            include_metadata=True,
        )
        out = []
        for m in r.matches:
            text, meta = self._unpack(dict(m.metadata or {}))
            out.append(Match(m.id, float(m.score), text, meta))
        return out

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        if not ids:
            return []
        ix = await self._index()
        got = (await ix.fetch(ids=list(ids), namespace=self._ns(namespace))).vectors
        out = []
        for i in ids:
            if i in got:
                text, meta = self._unpack(dict(got[i].metadata or {}))
                out.append(Record(i, [float(x) for x in got[i].values], text, meta))
        return out

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        if not ids:
            return
        ns = self._ns(namespace)
        ix = await self._index()
        await ix.delete(ids=list(ids), namespace=ns)
        if self.consistent:
            await self._await_visible(list(ids), ns, present=False)

    async def drop_namespace(self, namespace: str) -> None:
        ix = await self._index()
        await ix.delete(delete_all=True, namespace=self._ns(namespace))

    async def count(self, *, namespace: str | None = None) -> int:
        ix = await self._index()
        stats = await ix.describe_index_stats()
        ns = (stats.namespaces or {}).get(self._ns(namespace))
        return int(ns.vector_count) if ns is not None else 0

    async def aclose(self) -> None:
        if self._ix is not None:
            await self._ix.close()
            self._ix = None
        if self._pc is not None and self._own:
            await self._pc.close()
            self._pc = None
