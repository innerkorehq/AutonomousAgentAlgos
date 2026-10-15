# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Qdrant through `AsyncQdrantClient` (server, cloud, or the in-process `:memory:` / local-path modes).

    store = QdrantStore(url="http://localhost:6333", dimension=1536, collection="docs")
    store = QdrantStore(location=":memory:", dimension=1536)          # tests, notebooks

One collection holds every namespace; the namespace is a keyword payload field (indexed with `is_tenant=True`, Qdrant's
recommendation for multi-tenant data) that every query filters on. Record ids are mapped to UUIDs (Qdrant ids are UUIDs or integers);
the original id, text and metadata live in the payload. Metadata sits under `meta.` so it can never collide with the store's own fields.

Not expressible natively, so refused with `UnsupportedFilter` rather than approximated: ranges over **strings** (Qdrant ranges are numeric
or datetime). Scores: cosine and dot are similarities, euclidean is a distance (lower is better).

Qdrant stores cosine-collection vectors **normalised to unit length**, so `get` returns the direction you sent, not its magnitude."""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Sequence
from typing import Any

from aaa.vector.filters import And, Eq, Exists, Filter, In, Ne, Not, Or, Range, UnsupportedFilter
from aaa.vector.types import DEFAULT_NAMESPACE, Match, Metric, Record, check_vector

_UUID_NS = uuid.UUID("6f1c1e0e-5a53-4a0e-9d3c-0a4f5b1a7a11")
NS_KEY, ID_KEY, TEXT_KEY, META = "_ns", "_id", "_text", "meta"


def point_id(namespace: str, id: str) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{namespace}\x00{id}"))


def to_qdrant(f: Filter) -> Any:
    from qdrant_client import models as m

    def key(field: str) -> str:
        return f"{META}.{field}"

    def num(v: Any) -> bool:
        return isinstance(v, int | float) and not isinstance(v, bool)

    if isinstance(f, Eq):
        if f.value is None:
            return m.Filter(must=[m.IsNullCondition(is_null=m.PayloadField(key=key(f.field)))])
        return m.Filter(
            must=[m.FieldCondition(key=key(f.field), match=m.MatchValue(value=f.value))]
        )
    if isinstance(f, Ne):
        return m.Filter(must_not=[to_qdrant(Eq(f.field, f.value))])
    if isinstance(f, In):
        vals = list(f.values)
        if (
            vals
            and all(isinstance(v, str) for v in vals)
            or vals
            and all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
        ):
            return m.Filter(must=[m.FieldCondition(key=key(f.field), match=m.MatchAny(any=vals))])
        return (
            m.Filter(should=[to_qdrant(Eq(f.field, v)) for v in vals])
            if vals
            else m.Filter(must=[m.HasIdCondition(has_id=[])])
        )
    if isinstance(f, Exists):
        return m.Filter(must_not=[m.IsEmptyCondition(is_empty=m.PayloadField(key=key(f.field)))])
    if isinstance(f, Range):
        bounds = {"gt": f.gt, "gte": f.gte, "lt": f.lt, "lte": f.lte}
        used = {k: v for k, v in bounds.items() if v is not None}
        if not all(num(v) for v in used.values()):
            raise UnsupportedFilter(
                "Qdrant ranges are numeric: a range over strings (or booleans) cannot be evaluated by the store"
            )
        return m.Filter(
            must=[
                m.FieldCondition(
                    key=key(f.field), range=m.Range(**{k: float(v) for k, v in used.items()})
                )
            ]
        )
    if isinstance(f, And):
        return m.Filter(must=[to_qdrant(i) for i in f.items])
    if isinstance(f, Or):
        return (
            m.Filter(should=[to_qdrant(i) for i in f.items])
            if f.items
            else m.Filter(must=[m.HasIdCondition(has_id=[])])
        )
    if isinstance(f, Not):
        return m.Filter(must_not=[to_qdrant(f.item)])
    raise UnsupportedFilter(f"Qdrant adapter cannot express {type(f).__name__}")


class QdrantStore:
    def __init__(
        self,
        *,
        dimension: int,
        collection: str = "aaa_vectors",
        metric: Metric = "cosine",
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        self.dimension: int | None = dimension
        self.metric: Metric = metric
        self.collection = collection
        self._client = client
        self._own = client is None
        self._kw = client_kwargs
        self._ready = False

    @property
    def higher_is_better(self) -> bool:
        return self.metric != "euclidean"

    async def _c(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(**self._kw)
        return self._client

    async def setup(self) -> None:
        from qdrant_client import models as m

        c = await self._c()
        if not await c.collection_exists(self.collection):
            dist = {
                "cosine": m.Distance.COSINE,
                "dot": m.Distance.DOT,
                "euclidean": m.Distance.EUCLID,
            }[self.metric]
            await c.create_collection(
                self.collection,
                vectors_config=m.VectorParams(size=int(self.dimension or 0), distance=dist),
            )
        with (
            warnings.catch_warnings()
        ):  # the local (in-process) client says payload indexes have no effect: true, and harmless
            warnings.simplefilter("ignore", UserWarning)
            await c.create_payload_index(
                self.collection,
                NS_KEY,
                field_schema=m.KeywordIndexParams(type="keyword", is_tenant=True),
            )
        self._ready = True

    async def _client_ready(self) -> Any:
        if not self._ready:
            await self.setup()
        return await self._c()

    def _scope(self, namespace: str | None, f: Filter | None = None) -> Any:
        from qdrant_client import models as m

        must: list[Any] = [
            m.FieldCondition(key=NS_KEY, match=m.MatchValue(value=namespace or DEFAULT_NAMESPACE))
        ]
        if f is not None:
            must.append(to_qdrant(f))
        return m.Filter(must=must)

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        from qdrant_client import models as m

        for r in records:
            check_vector(self.dimension, r.vector)
        ns = namespace or DEFAULT_NAMESPACE
        c = await self._client_ready()
        await c.upsert(
            self.collection,
            points=[
                m.PointStruct(
                    id=point_id(ns, r.id),
                    vector=[float(x) for x in r.vector],
                    payload={NS_KEY: ns, ID_KEY: r.id, TEXT_KEY: r.text, META: r.meta},
                )
                for r in records
            ],
        )

    async def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        check_vector(self.dimension, vector, "query vector")
        scope = self._scope(
            namespace, filter
        )  # translate first: an inexpressible filter must fail before any I/O
        c = await self._client_ready()
        res = await c.query_points(
            self.collection,
            query=[float(x) for x in vector],
            query_filter=scope,
            limit=int(k),
            with_payload=True,
        )
        return [
            Match(
                p.payload[ID_KEY],
                float(p.score),
                p.payload.get(TEXT_KEY, ""),
                dict(p.payload.get(META) or {}),
            )
            for p in res.points
        ]

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        if not ids:
            return []
        ns = namespace or DEFAULT_NAMESPACE
        c = await self._client_ready()
        pts = await c.retrieve(
            self.collection,
            ids=[point_id(ns, i) for i in ids],
            with_payload=True,
            with_vectors=True,
        )
        by = {
            p.payload[ID_KEY]: Record(
                p.payload[ID_KEY],
                list(p.vector),
                p.payload.get(TEXT_KEY, ""),
                dict(p.payload.get(META) or {}),
            )
            for p in pts
        }
        return [by[i] for i in ids if i in by]

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        from qdrant_client import models as m

        if ids:
            ns = namespace or DEFAULT_NAMESPACE
            c = await self._client_ready()
            await c.delete(
                self.collection,
                points_selector=m.PointIdsList(points=[point_id(ns, i) for i in ids]),
            )

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int:
        from qdrant_client import models as m

        scope = self._scope(namespace, filter)
        c = await self._client_ready()
        n = (await c.count(self.collection, count_filter=scope, exact=True)).count
        await c.delete(self.collection, points_selector=m.FilterSelector(filter=scope))
        return int(n)

    async def drop_namespace(self, namespace: str) -> None:
        from qdrant_client import models as m

        c = await self._client_ready()
        await c.delete(
            self.collection, points_selector=m.FilterSelector(filter=self._scope(namespace))
        )

    async def count(self, *, namespace: str | None = None) -> int:
        c = await self._client_ready()
        return int(
            (await c.count(self.collection, count_filter=self._scope(namespace), exact=True)).count
        )

    async def aclose(self) -> None:
        if self._client is not None and self._own:
            await self._client.close()
            self._client = None
