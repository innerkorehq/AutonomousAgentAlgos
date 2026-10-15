# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""LanceDB (embedded; a directory, or object storage through LanceDB's `storage_options`).

    store = LanceDBStore("./lance", dimension=384)

One table holds every namespace; the namespace (hex encoded) is a column that every query pre-filters on, and rows are keyed by a hash of
`(namespace, id)`, so no user value is ever interpolated into the SQL-like expressions LanceDB takes.

**Metadata filters are exact but not pushed down.** Metadata is stored as JSON, which LanceDB cannot query, so a filtered search asks for candidates in
growing batches and applies the filter here until `k` survive or the namespace is exhausted: always correct (the same semantics as every other store, tested by the
shared contract suite), but a very selective filter scans a lot. If you need pushed-down metadata filtering at scale, choose pgvector or Qdrant, or put the field you filter
on in the namespace. The sync client runs on one dedicated thread.

Scores: cosine -> similarity, dot -> inner product (LanceDB reports 1 - dot), euclidean -> L2 distance (LanceDB reports it squared; the adapter takes the root)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from aaa.vector._sync import SyncRunner
from aaa.vector.filters import Filter, matches
from aaa.vector.types import DEFAULT_NAMESPACE, Match, Metric, Record, check_vector


def _rid(ns: str, id: str) -> str:
    return hashlib.sha256(f"{ns}\x00{id}".encode()).hexdigest()


def _nsk(ns: str) -> str:
    return ns.encode().hex()


class LanceDBStore:
    def __init__(
        self,
        uri: str = "./lancedb",
        *,
        dimension: int,
        metric: Metric = "cosine",
        table: str = "aaa_vectors",
        storage_options: dict[str, str] | None = None,
        initial_batch: int = 64,
    ) -> None:
        self.uri, self.table_name, self.storage_options = uri, table, storage_options
        self.dimension: int | None = dimension
        self.metric: Metric = metric
        self.initial_batch = initial_batch
        self._run = SyncRunner("aaa-lancedb")
        self._tbl: Any = None
        self._db: Any = None

    @property
    def higher_is_better(self) -> bool:
        return self.metric != "euclidean"

    def _table(self) -> Any:
        if self._tbl is None:
            import lancedb
            import pyarrow as pa

            self._db = lancedb.connect(self.uri, storage_options=self.storage_options)
            schema = pa.schema(
                [
                    pa.field("rid", pa.string()),
                    pa.field("nsk", pa.string()),
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), int(self.dimension or 0))),
                    pa.field("text", pa.string()),
                    pa.field("meta", pa.string()),
                ]
            )
            self._tbl = self._db.create_table(self.table_name, schema=schema, exist_ok=True)
        return self._tbl

    @staticmethod
    def _in(ids: Sequence[str], ns: str) -> str:
        return "rid IN (" + ",".join(f"'{_rid(ns, i)}'" for i in ids) + ")"

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        for r in records:
            check_vector(self.dimension, r.vector)
        if not records:
            return
        ns = namespace or DEFAULT_NAMESPACE
        rows = [
            {
                "rid": _rid(ns, r.id),
                "nsk": _nsk(ns),
                "id": r.id,
                "vector": [float(x) for x in r.vector],
                "text": r.text,
                "meta": json.dumps(r.meta),
            }
            for r in records
        ]

        def go() -> None:
            self._table().merge_insert(
                "rid"
            ).when_matched_update_all().when_not_matched_insert_all().execute(rows)

        await self._run.run(go)

    def _score(self, d: float) -> float:
        if self.metric == "euclidean":
            return math.sqrt(max(d, 0.0))
        return 1.0 - d

    async def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        check_vector(self.dimension, vector, "query vector")
        ns = namespace or DEFAULT_NAMESPACE
        qv = [float(x) for x in vector]

        def fetch(n: int) -> list[dict[str, Any]]:
            q = (
                self._table()
                .search(qv)
                .metric(self.metric if self.metric != "euclidean" else "l2")
                .where(f"nsk = '{_nsk(ns)}'", prefilter=True)
            )
            return q.limit(n).to_list()

        def go() -> list[Match]:
            total = self._table().count_rows(f"nsk = '{_nsk(ns)}'")
            if total == 0:
                return []
            n = k if filter is None else max(k, self.initial_batch)
            while True:
                n = min(n, total)
                rows = fetch(n)
                out = []
                for r in rows:
                    meta = json.loads(r["meta"])
                    if filter is None or matches(filter, meta):
                        out.append(
                            Match(r["id"], float(self._score(r["_distance"])), r["text"], meta)
                        )
                if filter is None or len(out) >= k or n >= total:
                    return out[:k]
                n *= 4  # not enough survivors yet: look further down the ranking

        return await self._run.run(go)

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        if not ids:
            return []
        ns = namespace or DEFAULT_NAMESPACE

        def go() -> list[dict[str, Any]]:
            return self._table().search().where(self._in(ids, ns)).limit(len(ids)).to_list()

        by = {
            r["id"]: Record(
                r["id"], [float(x) for x in r["vector"]], r["text"], json.loads(r["meta"])
            )
            for r in await self._run.run(go)
        }
        return [by[i] for i in ids if i in by]

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        if ids:
            ns = namespace or DEFAULT_NAMESPACE
            await self._run.run(lambda: self._table().delete(self._in(ids, ns)))

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int:
        ns = namespace or DEFAULT_NAMESPACE

        def go() -> int:
            t = self._table()
            rows = (
                t.search()
                .where(f"nsk = '{_nsk(ns)}'")
                .select(["rid", "meta"])
                .limit(t.count_rows() or 1)
                .to_list()
            )
            doomed = [r["rid"] for r in rows if matches(filter, json.loads(r["meta"]))]
            for i in range(0, len(doomed), 500):
                t.delete("rid IN (" + ",".join(f"'{x}'" for x in doomed[i : i + 500]) + ")")
            return len(doomed)

        return int(await self._run.run(go))

    async def drop_namespace(self, namespace: str) -> None:
        await self._run.run(lambda: self._table().delete(f"nsk = '{_nsk(namespace)}'"))

    async def count(self, *, namespace: str | None = None) -> int:
        ns = namespace or DEFAULT_NAMESPACE
        return int(await self._run.run(lambda: self._table().count_rows(f"nsk = '{_nsk(ns)}'")))

    async def aclose(self) -> None:
        self._run.close()
        self._tbl = self._db = None
