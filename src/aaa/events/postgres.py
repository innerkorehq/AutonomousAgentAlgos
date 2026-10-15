"""PostgreSQL EventStore + bus backend (asyncpg).

* Events live in `aaa_events` (PK stream,seq; UNIQUE stream,id => idempotent append).
* The bus is a transactional OUTBOX (`aaa_outbox`) written in the SAME transaction as the event.
  Each append takes a row lock on its partition's counter, so (a) appends within a partition are
  serialized, (b) outbox ids are gap-free and commit-ordered, which makes cursor-based tailing
  correct (no "late commit with a lower id" problem), and (c) per-stream seq/OCC need no extra lock.
* Wake-ups use LISTEN/NOTIFY; polling is the fallback, so a dropped listener degrades latency,
  never correctness.
* Partition leases are session-level advisory locks on a dedicated connection: when a process dies
  its connection dies and Postgres frees its leases immediately (no TTL guesswork).
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from collections.abc import AsyncIterator, Sequence
from typing import Any

import asyncpg

from aaa.core.aio import wait_event
from aaa.core.errors import ConcurrencyError
from aaa.events import migrate
from aaa.events.distributed import partition_of
from aaa.events.event import Event, scrubbed
from aaa.events.snapshot import StoredSnapshot
from aaa.events.store import ANY_SEQ, RUN_COMPLETED, RUN_STARTED, RunInfo

log = logging.getLogger("aaa.postgres")
CHANNEL = "aaa_bus"

MIGRATIONS: dict[
    int, list[str]
] = {}  # n -> SQL statements migrating layout n-1 to n (empty: version 1 is the 0.1/0.2 layout)

_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS aaa_meta (key text PRIMARY KEY, value text NOT NULL)",
    """CREATE TABLE IF NOT EXISTS aaa_events (
        stream text NOT NULL, seq integer NOT NULL, id text NOT NULL, type text NOT NULL, body text NOT NULL,
        PRIMARY KEY (stream, seq), UNIQUE (stream, id))""",
    "CREATE INDEX IF NOT EXISTS aaa_events_type ON aaa_events (type)",
    "CREATE TABLE IF NOT EXISTS aaa_partitions (partition integer PRIMARY KEY, last bigint NOT NULL DEFAULT 0)",
    """CREATE TABLE IF NOT EXISTS aaa_outbox (
        partition integer NOT NULL, id bigint NOT NULL, body text NOT NULL, PRIMARY KEY (partition, id))""",
    """CREATE TABLE IF NOT EXISTS aaa_cursors (
        grp text NOT NULL, partition integer NOT NULL, cursor bigint NOT NULL, PRIMARY KEY (grp, partition))""",
    """CREATE TABLE IF NOT EXISTS aaa_members (
        grp text NOT NULL, member text NOT NULL, expires timestamptz NOT NULL, PRIMARY KEY (grp, member))""",
    """CREATE TABLE IF NOT EXISTS aaa_snapshots (
        stream text NOT NULL, name text NOT NULL, seq integer NOT NULL, version integer NOT NULL, state text NOT NULL,
        PRIMARY KEY (stream, name))""",
]


def _signed32(n: int) -> int:
    n &= 0xFFFFFFFF
    return n - (1 << 32) if n >= (1 << 31) else n


class PostgresEventStore:
    def __init__(self, pool: asyncpg.Pool, dsn: str, partitions: int, outbox: bool) -> None:
        self.pool, self.dsn, self.partitions, self.outbox = pool, dsn, partitions, outbox
        self._signals: dict[int, asyncio.Event] = {}
        self._listener: asyncpg.Connection | None = None
        self._backend: PostgresBusBackend | None = None

    @classmethod
    async def create(
        cls, dsn: str, *, partitions: int = 16, outbox: bool = True, max_pool: int = 10
    ) -> PostgresEventStore:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=max_pool)
        self = cls(pool, dsn, partitions, outbox)
        try:
            async with pool.acquire() as c:
                await c.execute(
                    "SELECT pg_advisory_lock(727001)"
                )  # serialize concurrent first-time setup
                try:
                    for stmt in _SCHEMA:
                        await c.execute(stmt)
                    await c.execute(
                        "INSERT INTO aaa_meta VALUES ('partitions', $1) ON CONFLICT DO NOTHING",
                        str(partitions),
                    )
                    have = int(
                        await c.fetchval("SELECT value FROM aaa_meta WHERE key='partitions'")
                    )
                    if have != partitions:
                        raise ValueError(
                            f"this database was created with {have} partitions, not {partitions}; "
                            "the partition count is fixed for the lifetime of a cluster"
                        )
                    await c.execute(
                        "INSERT INTO aaa_partitions (partition) SELECT g FROM generate_series(0, $1 - 1) g ON CONFLICT DO NOTHING",
                        partitions,
                    )
                    await self._migrate(c)
                finally:
                    await c.execute("SELECT pg_advisory_unlock(727001)")
        except BaseException:
            await pool.close()
            raise
        await self._start_listener()
        return self

    async def _migrate(self, c: Any) -> None:
        """Called with the setup advisory lock held. A database without a version row was created by 0.1/0.2: layout 1."""
        await c.execute(
            "INSERT INTO aaa_meta VALUES ('schema_version', $1) ON CONFLICT DO NOTHING",
            str(migrate.SCHEMA_VERSION),
        )
        found = int(await c.fetchval("SELECT value FROM aaa_meta WHERE key='schema_version'"))
        for v in migrate.pending(found, migrate.SCHEMA_VERSION, MIGRATIONS):
            async with c.transaction():
                for stmt in MIGRATIONS[v]:
                    await c.execute(stmt)
                await c.execute("UPDATE aaa_meta SET value=$1 WHERE key='schema_version'", str(v))

    async def schema_version(self) -> int:
        async with self.pool.acquire() as c:
            return int(
                str(await c.fetchval("SELECT value FROM aaa_meta WHERE key='schema_version'"))
            )

    async def _start_listener(self) -> None:
        try:
            conn = await asyncpg.connect(self.dsn)
            self._listener = conn
            await conn.add_listener(CHANNEL, self._on_notify)
        except Exception:
            log.warning("LISTEN unavailable; falling back to polling", exc_info=True)
            self._listener = None

    def _on_notify(self, _c: Any, _pid: int, _chan: str, payload: str) -> None:
        self._signal(int(payload)).set()

    def _signal(self, p: int) -> asyncio.Event:
        return self._signals.setdefault(p, asyncio.Event())

    async def _wait(self, p: int, timeout: float) -> None:
        await wait_event(self._signal(p), timeout)

    def bus_backend(self) -> PostgresBusBackend:
        if self._backend is None:
            self._backend = PostgresBusBackend(self)
        return self._backend

    # -- EventStore -------------------------------------------------------------------------
    @staticmethod
    def _event(seq: int, body: str) -> Event:
        return Event.model_validate_json(body).model_copy(update={"seq": seq})

    async def append(
        self, stream: str, events: Sequence[Event], expected_seq: int = ANY_SEQ
    ) -> list[Event]:
        p = partition_of(stream, self.partitions)
        async with self.pool.acquire() as c, c.transaction():
            last_id = await c.fetchval(
                "SELECT last FROM aaa_partitions WHERE partition=$1 FOR UPDATE", p
            )
            cur = await c.fetchval(
                "SELECT COALESCE(MAX(seq), -1) FROM aaa_events WHERE stream=$1", stream
            )
            if expected_seq != ANY_SEQ and expected_seq != cur:
                raise ConcurrencyError(f"{stream}: expected seq {expected_seq}, at {cur}")
            out: list[Event] = []
            nxt, n_new = cur + 1, 0
            for e in events:
                row = await c.fetchrow(
                    "SELECT seq, body FROM aaa_events WHERE stream=$1 AND id=$2", stream, e.id
                )
                if row:
                    out.append(
                        self._event(row["seq"], row["body"]).model_copy(update={"dup": True})
                    )
                    continue
                stored = e.model_copy(update={"stream": stream, "seq": nxt})
                body = stored.model_dump_json()
                await c.execute(
                    "INSERT INTO aaa_events (stream, seq, id, type, body) VALUES ($1,$2,$3,$4,$5)",
                    stream,
                    nxt,
                    stored.id,
                    stored.type,
                    body,
                )
                n_new += 1
                if self.outbox:
                    await c.execute(
                        "INSERT INTO aaa_outbox (partition, id, body) VALUES ($1,$2,$3)",
                        p,
                        last_id + n_new,
                        body,
                    )
                out.append(stored)
                nxt += 1
            if n_new:
                await c.execute(
                    "UPDATE aaa_partitions SET last = last + $2 WHERE partition=$1", p, n_new
                )
                await c.execute("SELECT pg_notify($1, $2)", CHANNEL, str(p))
            return out

    async def read(self, stream: str, from_seq: int = 0, limit: int | None = None) -> list[Event]:
        rows = await self.pool.fetch(
            "SELECT seq, body FROM aaa_events WHERE stream=$1 AND seq>=$2 ORDER BY seq LIMIT $3",
            stream,
            from_seq,
            limit if limit is not None else 2**31 - 1,
        )
        return [self._event(r["seq"], r["body"]) for r in rows]

    async def last_seq(self, stream: str) -> int:
        return int(
            await self.pool.fetchval(
                "SELECT COALESCE(MAX(seq), -1) FROM aaa_events WHERE stream=$1", stream
            )
        )

    async def has(self, stream: str, event_id: str) -> bool:
        return bool(
            await self.pool.fetchval(
                "SELECT 1 FROM aaa_events WHERE stream=$1 AND id=$2", stream, event_id
            )
        )

    async def subscribe(self, stream: str, from_seq: int = 0) -> AsyncIterator[Event]:
        pos, p = from_seq, partition_of(stream, self.partitions)
        while True:
            sig = self._signal(p)
            sig.clear()
            batch = await self.read(stream, pos)
            if not batch:
                await self._wait(p, 0.5)
                continue
            for e in batch:
                yield e
            pos = batch[-1].seq + 1

    async def run_index(self) -> list[RunInfo]:
        started = await self.pool.fetch(
            "SELECT DISTINCT ON (stream) stream, seq, body FROM aaa_events WHERE type=$1 ORDER BY stream, seq",
            RUN_STARTED,
        )
        done = {
            r["stream"]: r
            for r in await self.pool.fetch(
                "SELECT DISTINCT ON (stream) stream, seq, body FROM aaa_events WHERE type=$1 ORDER BY stream, seq",
                RUN_COMPLETED,
            )
        }
        return [
            RunInfo(
                r["stream"],
                self._event(r["seq"], r["body"]),
                self._event(done[r["stream"]]["seq"], done[r["stream"]]["body"])
                if r["stream"] in done
                else None,
            )
            for r in started
        ]

    # -- erasure ----------------------------------------------------------------------------
    async def _erase(self, stream: str, delete: bool) -> int:
        p = partition_of(stream, self.partitions)
        async with self.pool.acquire() as c, c.transaction():
            await c.execute("SELECT 1 FROM aaa_partitions WHERE partition=$1 FOR UPDATE", p)
            rows = await c.fetch("SELECT seq, body FROM aaa_events WHERE stream=$1", stream)
            if delete:
                await c.execute("DELETE FROM aaa_events WHERE stream=$1", stream)
            else:
                for r in rows:
                    await c.execute(
                        "UPDATE aaa_events SET body=$3 WHERE stream=$1 AND seq=$2",
                        stream,
                        r["seq"],
                        scrubbed(self._event(r["seq"], r["body"])).model_dump_json(),
                    )
            # bus copies of the events (a bounded, transient outbox): the payloads must go too
            await c.execute(
                "DELETE FROM aaa_outbox WHERE partition=$1 AND (body::jsonb)->>'stream' = $2",
                p,
                stream,
            )
            await c.execute("DELETE FROM aaa_snapshots WHERE stream=$1", stream)
            return len(rows)

    async def scrub(self, stream: str) -> int:
        return await self._erase(stream, False)

    async def delete_stream(self, stream: str) -> int:
        return await self._erase(stream, True)

    # -- snapshots --------------------------------------------------------------------------
    async def load_snapshot(self, stream: str, name: str) -> StoredSnapshot | None:
        r = await self.pool.fetchrow(
            "SELECT seq, version, state FROM aaa_snapshots WHERE stream=$1 AND name=$2",
            stream,
            name,
        )
        return (
            StoredSnapshot(
                stream=stream, name=name, seq=r["seq"], version=r["version"], state=r["state"]
            )
            if r
            else None
        )

    async def save_snapshot(self, snap: StoredSnapshot) -> None:
        await self.pool.execute(
            """INSERT INTO aaa_snapshots (stream, name, seq, version, state) VALUES ($1,$2,$3,$4,$5)
               ON CONFLICT (stream, name) DO UPDATE SET seq=EXCLUDED.seq, version=EXCLUDED.version, state=EXCLUDED.state
               WHERE EXCLUDED.seq > aaa_snapshots.seq""",
            snap.stream,
            snap.name,
            snap.seq,
            snap.version,
            snap.state,
        )

    # -- maintenance ------------------------------------------------------------------------
    async def trim_outbox(self, keep_last: int = 100_000) -> int:
        """Delete bus entries older than the newest `keep_last` per partition. Consumers that fall
        further behind than this lose the gap; size it to your worst-case outage."""
        res = await self.pool.execute(
            "DELETE FROM aaa_outbox o USING aaa_partitions p WHERE o.partition = p.partition AND o.id <= p.last - $1",
            keep_last,
        )
        return int(res.split()[-1])

    async def aclose(self) -> None:
        if self._backend:
            await self._backend.close()
        if self._listener:
            await self._listener.close()
        await self.pool.close()


class PostgresBusBackend:
    def __init__(self, store: PostgresEventStore) -> None:
        self.store, self.partitions = store, store.partitions
        self._lease_conn: asyncpg.Connection | None = None
        self._lease_lock = asyncio.Lock()
        self._held: set[tuple[str, int]] = set()

    async def read(
        self, partition: int, after: int, limit: int, wait_s: float
    ) -> list[tuple[int, Event]]:
        q = "SELECT id, body FROM aaa_outbox WHERE partition=$1 AND id>$2 ORDER BY id LIMIT $3"
        s = self.store
        sig = s._signal(partition)
        sig.clear()
        rows = await s.pool.fetch(q, partition, after, limit)
        if not rows and wait_s > 0:
            await s._wait(partition, wait_s)
            rows = await s.pool.fetch(q, partition, after, limit)
        return [(r["id"], Event.model_validate_json(r["body"])) for r in rows]

    async def tail(self, partition: int) -> int:
        return int(
            await self.store.pool.fetchval(
                "SELECT last FROM aaa_partitions WHERE partition=$1", partition
            )
        )

    async def get_cursor(self, group: str, partition: int) -> int:
        v = await self.store.pool.fetchval(
            "SELECT cursor FROM aaa_cursors WHERE grp=$1 AND partition=$2", group, partition
        )
        return int(v or 0)

    async def set_cursor(self, group: str, partition: int, cursor: int) -> None:
        await self.store.pool.execute(
            """INSERT INTO aaa_cursors (grp, partition, cursor) VALUES ($1,$2,$3)
               ON CONFLICT (grp, partition) DO UPDATE SET cursor = GREATEST(aaa_cursors.cursor, EXCLUDED.cursor)""",
            group,
            partition,
            cursor,
        )

    # leases: session advisory locks on one dedicated connection (freed automatically on death)
    @staticmethod
    def _key(group: str, partition: int) -> tuple[int, int]:
        return _signed32(zlib.crc32(f"aaa.lease.{group}".encode())), partition

    async def _conn(self) -> asyncpg.Connection:
        if self._lease_conn is None or self._lease_conn.is_closed():
            self._held.clear()  # a new session holds nothing
            self._lease_conn = await asyncpg.connect(self.store.dsn)
        return self._lease_conn  # type: ignore[return-value]

    async def try_lease(self, group: str, partition: int, member: str, ttl_s: float) -> bool:
        async with self._lease_lock:
            try:
                c = await self._conn()
                if (group, partition) in self._held:
                    await c.execute("SELECT 1")  # connection still alive => lock still held
                    return True
                k1, k2 = self._key(group, partition)
                if await c.fetchval("SELECT pg_try_advisory_lock($1, $2)", k1, k2):
                    self._held.add((group, partition))
                    return True
                return False
            except (asyncpg.PostgresError, OSError, asyncpg.InterfaceError):
                self._held.clear()
                if self._lease_conn:
                    self._lease_conn.terminate()
                    self._lease_conn = None
                return False

    async def release_lease(self, group: str, partition: int, member: str) -> None:
        async with self._lease_lock:
            if (
                (group, partition) in self._held
                and self._lease_conn
                and not self._lease_conn.is_closed()
            ):
                k1, k2 = self._key(group, partition)
                await self._lease_conn.execute("SELECT pg_advisory_unlock($1, $2)", k1, k2)
            self._held.discard((group, partition))

    async def heartbeat(self, group: str, member: str, ttl_s: float) -> None:
        await self.store.pool.execute(
            """INSERT INTO aaa_members (grp, member, expires) VALUES ($1,$2, now() + make_interval(secs => $3))
               ON CONFLICT (grp, member) DO UPDATE SET expires = EXCLUDED.expires""",
            group,
            member,
            ttl_s,
        )
        await self.store.pool.execute(
            "DELETE FROM aaa_members WHERE grp=$1 AND expires < now()", group
        )

    async def members(self, group: str) -> list[str]:
        rows = await self.store.pool.fetch(
            "SELECT member FROM aaa_members WHERE grp=$1 AND expires > now() ORDER BY member", group
        )
        return [r["member"] for r in rows]

    async def leave(self, group: str, member: str) -> None:
        await self.store.pool.execute(
            "DELETE FROM aaa_members WHERE grp=$1 AND member=$2", group, member
        )

    async def abandon(self) -> None:
        """Simulated crash: kill the lease session. Postgres frees its advisory locks at once."""
        if self._lease_conn:
            self._lease_conn.terminate()
            self._lease_conn = None
        self._held.clear()

    async def close(self) -> None:
        if self._lease_conn and not self._lease_conn.is_closed():
            await self._lease_conn.close()
        self._lease_conn = None
        self._held.clear()
