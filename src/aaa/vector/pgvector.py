# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""pgvector on PostgreSQL, through asyncpg (already a dependency of the Postgres event store).

    store = PgVectorStore("postgresql://user:pw@host/db", dimension=1536, table="doc_vectors")
    await store.setup()        # CREATE EXTENSION vector; table; HNSW index (needs the privileges for that, once)

One table, one row per `(namespace, id)`: `embedding vector(<dimension>)`, `text`, `meta jsonb`. Vectors travel as text literals cast
inside the SQL (`$1::text::vector`), so no asyncpg codec has to be registered on the connection. Filters become parameterised
jsonb conditions; table and index names are validated and quoted, never interpolated from user input.

Scores: `cosine` -> similarity (1 - cosine distance), `dot` -> inner product, `euclidean` -> L2 distance (lower is better).

Scaling notes: one HNSW index serves the whole table, and the namespace/filter conditions are applied to its candidates, so a very
selective filter on a huge table can return fewer than `k` rows unless you raise `hnsw.ef_search` (`ef_search=` below, default 100);
for hard tenant partitions at scale use one table per tenant (`table=`) or partition the table yourself."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from aaa.vector import hybrid
from aaa.vector.filters import And, Eq, Exists, Filter, In, Ne, Not, Or, Range, UnsupportedFilter
from aaa.vector.types import DEFAULT_NAMESPACE, Match, Metric, Record, check_vector

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_OPS = {
    "cosine": ("<=>", "vector_cosine_ops"),
    "dot": ("<#>", "vector_ip_ops"),
    "euclidean": ("<->", "vector_l2_ops"),
}


def _vec(v: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _parse_vec(s: str) -> list[float]:
    return [float(x) for x in s.strip("[]").split(",")] if s and s != "[]" else []


class _Sql:
    """Collects positional parameters while a filter is compiled to SQL."""

    def __init__(self, start: int) -> None:
        self.params: list[Any] = []
        self.start = start

    def p(self, value: Any, cast: str = "") -> str:
        self.params.append(value)
        return f"${self.start + len(self.params) - 1}{cast}"


def compile_filter(f: Filter, q: _Sql, col: str = "meta") -> str:
    def path(field: str) -> str:
        return q.p(field.split("."), "::text[]")

    def num_or_str(field: str, bound: Any, op: str) -> str:
        pth = path(field)
        if isinstance(bound, bool):
            return "false"
        if isinstance(bound, int | float):
            return f"(CASE WHEN jsonb_typeof({col} #> {pth}) = 'number' THEN ({col} #>> {pth})::numeric {op} {q.p(bound, '::numeric')} ELSE false END)"
        return f"(CASE WHEN jsonb_typeof({col} #> {pth}) = 'string' THEN ({col} #>> {pth}) {op} {q.p(bound, '::text')} ELSE false END)"

    if isinstance(f, Eq):
        if f.value is None:
            return f"({col} #> {path(f.field)} = 'null'::jsonb)"
        return f"({col} #> {path(f.field)} = {q.p(json.dumps(f.value), '::jsonb')})"
    if isinstance(f, Ne):
        return f"(NOT COALESCE({compile_filter(Eq(f.field, f.value), q, col)}, false))"
    if isinstance(f, In):
        if not f.values:
            return "false"
        return f"({col} #> {path(f.field)} = ANY({q.p([json.dumps(v) for v in f.values], '::jsonb[]')}))"
    if isinstance(f, Exists):
        return f"(({col} #> {path(f.field)}) IS NOT NULL AND ({col} #> {path(f.field)}) <> 'null'::jsonb)"
    if isinstance(f, Range):
        parts = [
            num_or_str(f.field, b, op)
            for b, op in ((f.gt, ">"), (f.gte, ">="), (f.lt, "<"), (f.lte, "<="))
            if b is not None
        ]
        return "(" + " AND ".join(parts) + ")" if parts else "false"
    if isinstance(f, And):
        return (
            "(" + " AND ".join(compile_filter(i, q, col) for i in f.items) + ")"
            if f.items
            else "true"
        )
    if isinstance(f, Or):
        return (
            "(" + " OR ".join(compile_filter(i, q, col) for i in f.items) + ")"
            if f.items
            else "false"
        )
    if isinstance(f, Not):
        return f"(NOT COALESCE({compile_filter(f.item, q, col)}, false))"
    raise UnsupportedFilter(f"pgvector cannot express {type(f).__name__}")


class PgVectorStore:
    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool: Any = None,
        dimension: int,
        metric: Metric = "cosine",
        table: str = "aaa_vectors",
        hnsw: bool = True,
        ef_search: int = 100,
        text_search_config: str = "english",
    ) -> None:
        if not _IDENT.match(table):
            raise ValueError(f"invalid table name {table!r}: letters, digits and underscores only")
        if (dsn is None) == (pool is None):
            raise ValueError("pass exactly one of dsn= or pool=")
        self.dsn, self._pool, self._own_pool = dsn, pool, pool is None
        self.dimension: int | None = dimension
        self.metric: Metric = metric
        self.table, self.hnsw, self.ef_search = table, hnsw, ef_search
        if not _IDENT.match(text_search_config):
            raise ValueError(f"invalid text search configuration {text_search_config!r}")
        self.text_search_config = text_search_config  # the PostgreSQL text-search configuration (language) used for hybrid search
        self._ready = False

    @property
    def higher_is_better(self) -> bool:
        return self.metric != "euclidean"

    @property
    def _t(self) -> str:
        return f'"{self.table}"'

    async def _db(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
        return self._pool

    async def setup(self) -> None:
        pool = await self._db()
        op, opclass = _OPS[self.metric]
        async with pool.acquire() as c:
            await c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await c.execute(
                f"CREATE TABLE IF NOT EXISTS {self._t} (namespace text NOT NULL, id text NOT NULL, "
                f"embedding vector({int(self.dimension or 0)}) NOT NULL, text text NOT NULL DEFAULT '', "
                f"meta jsonb NOT NULL DEFAULT '{{}}'::jsonb, PRIMARY KEY (namespace, id))"
            )
            if self.hnsw:
                await c.execute(
                    f'CREATE INDEX IF NOT EXISTS "{self.table}_hnsw" ON {self._t} USING hnsw (embedding {opclass})'
                )
            await c.execute(
                f'CREATE INDEX IF NOT EXISTS "{self.table}_meta" ON {self._t} USING gin (meta jsonb_path_ops)'
            )
            # keyword search for hybrid queries: a stored generated column, so every write path keeps it current (adds the column to tables from 0.2)
            await c.execute(
                f"ALTER TABLE {self._t} ADD COLUMN IF NOT EXISTS tsv tsvector "
                f"GENERATED ALWAYS AS (to_tsvector('{self.text_search_config}'::regconfig, text)) STORED"
            )
            await c.execute(
                f'CREATE INDEX IF NOT EXISTS "{self.table}_tsv" ON {self._t} USING gin (tsv)'
            )
        self._ready = True
        del op

    async def _acquire(self) -> Any:
        if not self._ready:
            await self.setup()
        return (await self._db()).acquire()

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        for r in records:
            check_vector(self.dimension, r.vector)
        ns = namespace or DEFAULT_NAMESPACE
        async with await self._acquire() as c:
            await c.executemany(
                f"INSERT INTO {self._t} (namespace, id, embedding, text, meta) VALUES ($1, $2, $3::text::vector, $4, $5::jsonb) "
                "ON CONFLICT (namespace, id) DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text, meta = EXCLUDED.meta",
                [(ns, r.id, _vec(r.vector), r.text, json.dumps(r.meta)) for r in records],
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
        op, _ = _OPS[self.metric]
        q = _Sql(3)
        where = " AND " + compile_filter(filter, q) if filter is not None else ""
        dist = f"embedding {op} $1::text::vector"
        score = {"cosine": f"1 - ({dist})", "dot": f"-({dist})", "euclidean": dist}[self.metric]
        sql = (
            f"SELECT id, text, meta, {score} AS score FROM {self._t} WHERE namespace = $2{where} "
            f"ORDER BY {dist} LIMIT {int(k)}"
        )
        async with await self._acquire() as c, c.transaction():
            await c.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}")
            rows = await c.fetch(sql, _vec(vector), namespace or DEFAULT_NAMESPACE, *q.params)
        return [Match(r["id"], float(r["score"]), r["text"], json.loads(r["meta"])) for r in rows]

    async def query_hybrid(
        self,
        vector: Sequence[float],
        text: str,
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        """Dense similarity fused with PostgreSQL full-text relevance (`ts_rank_cd`) by reciprocal rank. Any word of the query may match."""
        n = hybrid.pool(k)
        dense = await self.query(vector, n, filter=filter, namespace=namespace)
        words = hybrid.tokens(text)
        if not words:
            return dense[:k]
        q = _Sql(4)
        where = " AND " + compile_filter(filter, q) if filter is not None else ""
        sql = (
            f"SELECT id, text, meta FROM {self._t}, to_tsquery('{self.text_search_config}'::regconfig, $1) AS tq "
            f"WHERE namespace = $2 AND tsv @@ tq{where} ORDER BY ts_rank_cd(tsv, tq) DESC, id LIMIT $3"
        )
        async with await self._acquire() as c:
            rows = await c.fetch(
                sql, " | ".join(words), namespace or DEFAULT_NAMESPACE, n, *q.params
            )
        keyword = [Match(r["id"], 0.0, r["text"], json.loads(r["meta"])) for r in rows]
        return hybrid.fuse(dense, keyword, k)

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        if not ids:
            return []
        async with await self._acquire() as c:
            rows = await c.fetch(
                f"SELECT id, embedding::text AS v, text, meta FROM {self._t} WHERE namespace = $1 AND id = ANY($2::text[])",
                namespace or DEFAULT_NAMESPACE,
                list(ids),
            )
        by = {
            r["id"]: Record(r["id"], _parse_vec(r["v"]), r["text"], json.loads(r["meta"]))
            for r in rows
        }
        return [by[i] for i in ids if i in by]

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        if ids:
            async with await self._acquire() as c:
                await c.execute(
                    f"DELETE FROM {self._t} WHERE namespace = $1 AND id = ANY($2::text[])",
                    namespace or DEFAULT_NAMESPACE,
                    list(ids),
                )

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int:
        q = _Sql(2)
        cond = compile_filter(filter, q)
        async with await self._acquire() as c:
            tag = await c.execute(
                f"DELETE FROM {self._t} WHERE namespace = $1 AND {cond}",
                namespace or DEFAULT_NAMESPACE,
                *q.params,
            )
        return int(tag.split()[-1])

    async def drop_namespace(self, namespace: str) -> None:
        async with await self._acquire() as c:
            await c.execute(f"DELETE FROM {self._t} WHERE namespace = $1", namespace)

    async def count(self, *, namespace: str | None = None) -> int:
        async with await self._acquire() as c:
            return int(
                await c.fetchval(
                    f"SELECT count(*) FROM {self._t} WHERE namespace = $1",
                    namespace or DEFAULT_NAMESPACE,
                )
            )

    async def aclose(self) -> None:
        if self._pool is not None and self._own_pool:
            await self._pool.close()
            self._pool = None
