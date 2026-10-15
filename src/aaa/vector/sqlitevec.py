# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""SQLite with the sqlite-vec extension: a vector store in one file, for local tools, tests and small deployments.

    store = SqliteVecStore("vectors.db", dimension=384)          # or ":memory:"

sqlite-vec is brute force (as is this store): every query scans the matching rows, which is fine up to roughly a few hundred thousand
vectors and exact at any size. Rows live in an ordinary table (`ns`, `id`, `vec` blob, `text`, `meta` JSON); the extension supplies the distance
functions. Because the scan is a plain SQL query, **every filter is evaluated in SQL before the top-k cut**, using SQLite's JSON functions, with the same
semantics as the other stores. sqlite-vec is pre-1.0: pin its version.

Needs a Python whose `sqlite3` can load extensions (`enable_load_extension`); the macOS system Python and some minimal builds cannot. Scores:
cosine -> similarity, dot -> inner product (a Python function, slower), euclidean -> L2 distance (lower is better)."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Sequence
from typing import Any

from aaa.vector._sync import SyncRunner
from aaa.vector.filters import And, Eq, Exists, Filter, In, Ne, Not, Or, Range, UnsupportedFilter
from aaa.vector.types import DEFAULT_NAMESPACE, Match, Metric, Record, VectorError, check_vector


def _pack(v: Sequence[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _unpack(b: bytes) -> list[float]:
    return list(struct.unpack(f"{len(b) // 4}f", b))


def _dot(a: bytes, b: bytes) -> float:
    return sum(x * y for x, y in zip(_unpack(a), _unpack(b), strict=True))


def _path(field: str) -> str:
    parts = field.split(".")
    if any('"' in p or not p for p in parts):
        raise UnsupportedFilter(f"field name {field!r} cannot be used in a JSON path")
    return "$." + ".".join(f'"{p}"' for p in parts)


def compile_filter(f: Filter, params: list[Any]) -> str:
    def add(v: Any) -> str:
        params.append(v)
        return "?"

    def typ(field: str) -> str:
        return f"json_type(meta, {add(_path(field))})"

    def val(field: str) -> str:
        return f"json_extract(meta, {add(_path(field))})"

    def eq(field: str, v: Any) -> str:
        if v is None:
            return f"({typ(field)} = 'null')"
        if isinstance(v, bool):
            return f"({typ(field)} = {add('true' if v else 'false')})"
        if isinstance(v, int | float):
            return f"({typ(field)} IN ('integer','real') AND {val(field)} = {add(v)})"
        return f"({typ(field)} = 'text' AND {val(field)} = {add(v)})"

    if isinstance(f, Eq):
        return eq(f.field, f.value)
    if isinstance(f, Ne):
        return f"(NOT COALESCE({eq(f.field, f.value)}, 0))"
    if isinstance(f, In):
        return "(" + " OR ".join(eq(f.field, v) for v in f.values) + ")" if f.values else "0"
    if isinstance(f, Exists):
        return f"(COALESCE({typ(f.field)} <> 'null', 0))"
    if isinstance(f, Range):
        parts = []
        for bound, op in ((f.gt, ">"), (f.gte, ">="), (f.lt, "<"), (f.lte, "<=")):
            if bound is None:
                continue
            if isinstance(bound, bool):
                return "0"
            kinds = "('integer','real')" if isinstance(bound, int | float) else "('text')"
            parts.append(f"({typ(f.field)} IN {kinds} AND {val(f.field)} {op} {add(bound)})")
        return "(" + " AND ".join(parts) + ")" if parts else "0"
    if isinstance(f, And):
        return (
            "(" + " AND ".join(compile_filter(i, params) for i in f.items) + ")" if f.items else "1"
        )
    if isinstance(f, Or):
        return (
            "(" + " OR ".join(compile_filter(i, params) for i in f.items) + ")" if f.items else "0"
        )
    if isinstance(f, Not):
        return f"(NOT COALESCE({compile_filter(f.item, params)}, 0))"
    raise UnsupportedFilter(f"sqlite-vec adapter cannot express {type(f).__name__}")


class SqliteVecStore:
    def __init__(
        self,
        path: str = ":memory:",
        *,
        dimension: int,
        metric: Metric = "cosine",
        table: str = "aaa_vectors",
    ) -> None:
        import re

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$", table):
            raise ValueError(f"invalid table name {table!r}")
        self.path, self.table = path, table
        self.dimension: int | None = dimension
        self.metric: Metric = metric
        self._run = SyncRunner("aaa-sqlitevec")
        self._db: sqlite3.Connection | None = None

    @property
    def higher_is_better(self) -> bool:
        return self.metric != "euclidean"

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            import sqlite_vec

            db = sqlite3.connect(self.path, check_same_thread=False)
            if not hasattr(db, "enable_load_extension"):
                raise VectorError(
                    "this Python's sqlite3 cannot load extensions; use a build with enable_load_extension (e.g. python.org or uv's)"
                )
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            db.create_function("aaa_dot", 2, _dot, deterministic=True)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                f'CREATE TABLE IF NOT EXISTS "{self.table}" (ns TEXT NOT NULL, id TEXT NOT NULL, vec BLOB NOT NULL, '
                "text TEXT NOT NULL DEFAULT '', meta TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (ns, id))"
            )
            self._db = db
        return self._db

    @property
    def _t(self) -> str:
        return f'"{self.table}"'

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None:
        for r in records:
            check_vector(self.dimension, r.vector)
        ns = namespace or DEFAULT_NAMESPACE
        rows = [(ns, r.id, _pack(r.vector), r.text, json.dumps(r.meta)) for r in records]

        def go() -> None:
            db = self._conn()
            with db:
                db.executemany(
                    f"INSERT OR REPLACE INTO {self._t} (ns, id, vec, text, meta) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )

        await self._run.run(go)

    async def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]:
        check_vector(self.dimension, vector, "query vector")
        # Anonymous placeholders only, so parameters bind in textual order: the vector (in the distance), the namespace, then the filter's.
        params: list[Any] = [_pack(vector), namespace or DEFAULT_NAMESPACE]
        cond = ""
        if filter is not None:
            cond = " AND " + compile_filter(filter, params)
        dist = {
            "cosine": "vec_distance_cosine(vec, ?)",
            "euclidean": "vec_distance_L2(vec, ?)",
            "dot": "aaa_dot(vec, ?)",
        }[self.metric]
        order = "DESC" if self.metric == "dot" else "ASC"
        sql = f"SELECT id, text, meta, {dist} AS d FROM {self._t} WHERE ns = ?{cond} ORDER BY d {order} LIMIT {int(k)}"

        def go() -> list[tuple[Any, ...]]:
            return self._conn().execute(sql, params).fetchall()

        rows = await self._run.run(go)
        out = []
        for id_, text, meta, d in rows:
            score = 1.0 - d if self.metric == "cosine" else d
            out.append(Match(id_, float(score), text, json.loads(meta)))
        return out

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]:
        if not ids:
            return []
        ns = namespace or DEFAULT_NAMESPACE

        def go() -> list[tuple[Any, ...]]:
            marks = ",".join("?" * len(ids))
            return (
                self._conn()
                .execute(
                    f"SELECT id, vec, text, meta FROM {self._t} WHERE ns = ? AND id IN ({marks})",
                    [ns, *ids],
                )
                .fetchall()
            )

        by = {
            r[0]: Record(r[0], _unpack(r[1]), r[2], json.loads(r[3]))
            for r in await self._run.run(go)
        }
        return [by[i] for i in ids if i in by]

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None:
        if not ids:
            return
        ns = namespace or DEFAULT_NAMESPACE

        def go() -> None:
            db = self._conn()
            with db:
                db.execute(
                    f"DELETE FROM {self._t} WHERE ns = ? AND id IN ({','.join('?' * len(ids))})",
                    [ns, *ids],
                )

        await self._run.run(go)

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int:
        params: list[Any] = []
        cond = compile_filter(filter, params)

        def go() -> int:
            db = self._conn()
            with db:
                return db.execute(
                    f"DELETE FROM {self._t} WHERE ns = ? AND {cond}",
                    [namespace or DEFAULT_NAMESPACE, *params],
                ).rowcount

        return int(await self._run.run(go))

    async def drop_namespace(self, namespace: str) -> None:
        def go() -> None:
            db = self._conn()
            with db:
                db.execute(f"DELETE FROM {self._t} WHERE ns = ?", [namespace])

        await self._run.run(go)

    async def count(self, *, namespace: str | None = None) -> int:
        def go() -> int:
            return (
                self._conn()
                .execute(
                    f"SELECT count(*) FROM {self._t} WHERE ns = ?", [namespace or DEFAULT_NAMESPACE]
                )
                .fetchone()[0]
            )

        return int(await self._run.run(go))

    async def aclose(self) -> None:
        def go() -> None:
            if self._db is not None:
                self._db.close()
                self._db = None

        await self._run.run(go)
        self._run.close()
