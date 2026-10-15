# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Chroma (embedded, persistent on disk, or a server through its HTTP client).

    store = ChromaStore(path="./chroma", dimension=384)                  # embedded and persistent
    store = ChromaStore(host="chroma.internal", port=8000, dimension=384)   # server
    store = ChromaStore(dimension=384)                                    # in memory

Chroma has no namespaces, so each namespace is its own collection (`<prefix>.<hex of the namespace>`). Chroma's metadata is flat and cannot hold
`null`, so nested objects are **flattened to dotted keys for filtering** and the exact original metadata is also kept as JSON (`_aaa_meta`), which is what
`query`/`get` return. Chroma's own client is synchronous; calls run on one dedicated thread.

Filters Chroma cannot express are refused with `UnsupportedFilter` (never approximated): ranges over strings, equality with `null`, `exists`, and negations
of ranges. Scores: cosine -> similarity, dot -> inner product (Chroma's `ip` distance is 1 - dot), euclidean -> L2 distance (Chroma reports it squared;
the adapter takes the root)."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Sequence
from typing import Any

from aaa.vector._sync import SyncRunner
from aaa.vector.filters import And, Eq, Exists, Filter, In, Ne, Not, Or, Range, UnsupportedFilter
from aaa.vector.types import DEFAULT_NAMESPACE, Match, Metric, Record, check_vector

META_KEY = "_aaa_meta"
TEXT_KEY = "_aaa_text"
NEVER = {"_aaa_never": True}  # matches nothing: no record has this key


def flatten(meta: dict[str, Any]) -> dict[str, Any]:
    """Scalar leaves of the metadata under dotted keys (what Chroma can filter on)."""
    out: dict[str, Any] = {}

    def walk(prefix: str, v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x)
        elif isinstance(v, str | int | float | bool):
            out[prefix] = v

    walk("", meta)
    return out


def _num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def to_where(f: Filter, negate: bool = False) -> dict[str, Any] | None:
    """Compile to a Chroma `where`; None means 'no condition'. Negation is pushed to the leaves (De Morgan)."""
    if isinstance(f, Not):
        return to_where(f.item, not negate)
    if isinstance(f, Eq):
        if f.value is None:
            raise UnsupportedFilter("Chroma cannot store null, so it cannot filter for it")
        return {f.field: {"$ne": f.value}} if negate else {f.field: f.value}
    if isinstance(f, Ne):
        return to_where(Eq(f.field, f.value), not negate)
    if isinstance(f, In):
        if not f.values:
            return None if negate else NEVER
        homogeneous = len({type(v) for v in f.values}) == 1
        if homogeneous:
            return {f.field: {"$nin" if negate else "$in": list(f.values)}}
        return to_where(
            And(tuple(Ne(f.field, v) for v in f.values))
            if negate
            else Or(tuple(Eq(f.field, v) for v in f.values))
        )
    if isinstance(f, Exists):
        raise UnsupportedFilter("Chroma cannot test whether a metadata field exists")
    if isinstance(f, Range):
        if negate:
            raise UnsupportedFilter(
                "Chroma cannot negate a range (that would include records where the field is missing or not a number)"
            )
        ops = {"$gt": f.gt, "$gte": f.gte, "$lt": f.lt, "$lte": f.lte}
        used = {k: v for k, v in ops.items() if v is not None}
        if not all(_num(v) for v in used.values()):
            raise UnsupportedFilter(
                "Chroma ranges are numeric: a range over strings cannot be evaluated by the store"
            )
        conds = [{f.field: {k: v}} for k, v in used.items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}
    if isinstance(f, And | Or):
        is_and = isinstance(f, And) != negate  # De Morgan: not(and) = or(not)
        parts = []
        for item in f.items:
            w = to_where(item, negate)
            if w is None:
                if is_and:
                    continue  # 'no condition' inside an AND adds nothing
                return None  # inside an OR it makes the whole OR unconditional
            parts.append(w)
        if not parts:
            return None if is_and else NEVER
        if len(parts) == 1:
            return parts[0]
        return {"$and" if is_and else "$or": parts}
    raise UnsupportedFilter(f"Chroma adapter cannot express {type(f).__name__}")


def _coll_name(prefix: str, namespace: str) -> str:
    return f"{prefix}.{namespace.encode().hex()}"


class ChromaStore:
    def __init__(
        self,
        *,
        dimension: int,
        metric: Metric = "cosine",
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        client: Any = None,
        prefix: str = "aaa",
    ) -> None:
        self.dimension: int | None = dimension
        self.metric: Metric = metric
        self.prefix = prefix
        self._args = {"path": path, "host": host, "port": port}
        self._client = client
        self._own = client is None
        self._run = SyncRunner("aaa-chroma")
        self._colls: dict[str, Any] = {}

    @property
    def higher_is_better(self) -> bool:
        return self.metric != "euclidean"

    def _c(self) -> Any:
        if self._client is None:
            import chromadb

            a = self._args
            if a["host"]:
                self._client = chromadb.HttpClient(host=a["host"], port=a["port"])
            elif a["path"]:
                self._client = chromadb.PersistentClient(path=a["path"])
            else:
                self._client = chromadb.EphemeralClient()
        return self._client

    def _coll(self, namespace: str | None, create: bool) -> Any | None:
        ns = namespace or DEFAULT_NAMESPACE
        name = _coll_name(self.prefix, ns)
        if name in self._colls:
            return self._colls[name]
        space = {"cosine": "cosine", "dot": "ip", "euclidean": "l2"}[self.metric]
        if create:
            col = self._c().get_or_create_collection(name, configuration={"hnsw": {"space": space}})
        else:
            try:
                col = self._c().get_collection(name)
            except Exception:  # noqa: BLE001 - chroma raises different errors by version when a collection is absent
                return None
        self._colls[name] = col
        return col

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        for r in records:
            check_vector(self.dimension, r.vector)
        if not records:
            return

        def go() -> None:
            col = self._coll(namespace, True)
            assert col is not None  # create=True always yields a collection
            col.upsert(
                ids=[r.id for r in records],
                embeddings=[[float(x) for x in r.vector] for r in records],
                metadatas=[
                    {**flatten(r.meta), META_KEY: json.dumps(r.meta), TEXT_KEY: r.text}
                    for r in records
                ],
            )

        await self._run.run(go)

    def _match(self, id_: str, dist: float | None, meta: dict[str, Any]) -> Match:
        score = 0.0
        if dist is not None:
            score = {
                "cosine": 1.0 - dist,
                "dot": 1.0 - dist,
                "euclidean": math.sqrt(max(dist, 0.0)),
            }[self.metric]
        return Match(
            id_, float(score), meta.get(TEXT_KEY, ""), json.loads(meta.get(META_KEY, "{}"))
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
        where = (
            to_where(filter) if filter is not None else None
        )  # compile first: refusals happen before any I/O

        def go() -> list[Match]:
            col = self._coll(namespace, False)
            if col is None:
                return []
            n = col.count()
            if n == 0:
                return []
            kw: dict[str, Any] = {"where": where} if where else {}
            r = col.query(
                query_embeddings=[[float(x) for x in vector]],
                n_results=min(int(k), n),
                include=["metadatas", "distances"],
                **kw,
            )
            return [
                self._match(i, d, m)
                for i, d, m in zip(r["ids"][0], r["distances"][0], r["metadatas"][0], strict=True)
            ]

        return await self._run.run(go)

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        if not ids:
            return []

        def go() -> list[Record]:
            col = self._coll(namespace, False)
            if col is None:
                return []
            r = col.get(ids=list(ids), include=["embeddings", "metadatas"])
            by = {
                i: Record(
                    i, [float(x) for x in e], m.get(TEXT_KEY, ""), json.loads(m.get(META_KEY, "{}"))
                )
                for i, e, m in zip(r["ids"], r["embeddings"], r["metadatas"], strict=True)
            }
            return [by[i] for i in ids if i in by]

        return await self._run.run(go)

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        def go() -> None:
            col = self._coll(namespace, False)
            if col is not None and ids:
                col.delete(ids=list(ids))

        await self._run.run(go)

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int:
        where = to_where(filter)

        def go() -> int:
            col = self._coll(namespace, False)
            if col is None:
                return 0
            found = col.get(where=where, include=[]) if where else col.get(include=[])
            if found["ids"]:
                col.delete(ids=found["ids"])
            return len(found["ids"])

        return int(await self._run.run(go))

    async def drop_namespace(self, namespace: str) -> None:
        def go() -> None:
            name = _coll_name(self.prefix, namespace)
            with contextlib.suppress(Exception):  # an absent collection: nothing to drop
                self._c().delete_collection(name)
            self._colls.pop(name, None)

        await self._run.run(go)

    async def count(self, *, namespace: str | None = None) -> int:
        def go() -> int:
            col = self._coll(namespace, False)
            return int(col.count()) if col is not None else 0

        return int(await self._run.run(go))

    async def aclose(self) -> None:
        self._run.close()
        self._colls.clear()
        if self._client is not None and self._own:
            close = getattr(self._client, "close", None)
            if close:
                close()
            self._client = None
