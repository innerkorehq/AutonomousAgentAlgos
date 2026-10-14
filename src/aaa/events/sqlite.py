from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from aaa.core.aio import wait_for
from aaa.core.errors import ConcurrencyError
from aaa.events.event import Event, scrubbed
from aaa.events.snapshot import StoredSnapshot
from aaa.events.store import ANY_SEQ, RUN_COMPLETED, RUN_STARTED, RunInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    stream TEXT NOT NULL,
    seq    INTEGER NOT NULL,
    id     TEXT NOT NULL,
    type   TEXT NOT NULL,
    body   TEXT NOT NULL,
    PRIMARY KEY (stream, seq),
    UNIQUE (stream, id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS events_type ON events (type);
CREATE TABLE IF NOT EXISTS snapshots (
    stream TEXT NOT NULL, name TEXT NOT NULL, seq INTEGER NOT NULL, version INTEGER NOT NULL, state TEXT NOT NULL,
    PRIMARY KEY (stream, name)
) WITHOUT ROWID;
"""


class SQLiteEventStore:
    """Durable EventStore on stdlib sqlite3 (no extra dependency).

    - One dedicated thread owns the connection, so the event loop never blocks on disk.
    - WAL + synchronous=NORMAL: crash-safe (no corruption); a power loss may drop the last
      few commits, never reorder or tear them. Use synchronous="FULL" for strict durability.
    - Append is one IMMEDIATE transaction: the OCC check and the inserts are atomic.
    - Append is idempotent on (stream, id), which is what makes crash recovery safe.
    - `subscribe` wakes instantly for same-process appends and polls for other processes."""

    def __init__(
        self, path: str | Path, *, synchronous: str = "NORMAL", poll_s: float = 0.2
    ) -> None:
        self.path, self._poll = str(path), poll_s
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aaa-sqlite")
        self._cond: asyncio.Condition | None = None
        self._pool.submit(self._open, synchronous).result()

    def _open(self, synchronous: str) -> None:
        self._conn = sqlite3.connect(self.path, isolation_level=None)  # explicit transactions
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA synchronous={synchronous}")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    async def _run(self, fn: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @staticmethod
    def _parse(seq: int, body: str) -> Event:
        return Event.model_validate_json(body).model_copy(update={"seq": seq})

    def _append(self, stream: str, events: list[Event], expected: int) -> list[Event]:
        c = self._conn
        c.execute("BEGIN IMMEDIATE")
        try:
            last = c.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM events WHERE stream=?", (stream,)
            ).fetchone()[0]
            if expected != ANY_SEQ and expected != last:
                raise ConcurrencyError(f"{stream}: expected seq {expected}, at {last}")
            out: list[Event] = []
            nxt = last + 1
            for e in events:
                row = c.execute(
                    "SELECT seq, body FROM events WHERE stream=? AND id=?", (stream, e.id)
                ).fetchone()
                if row:
                    out.append(self._parse(row[0], row[1]).model_copy(update={"dup": True}))
                    continue
                stored = e.model_copy(update={"stream": stream, "seq": nxt})
                c.execute(
                    "INSERT INTO events (stream, seq, id, type, body) VALUES (?,?,?,?,?)",
                    (stream, nxt, stored.id, stored.type, stored.model_dump_json()),
                )
                nxt += 1
                out.append(stored)
            c.execute("COMMIT")
            return out
        except BaseException:
            c.execute("ROLLBACK")
            raise

    async def append(
        self, stream: str, events: Sequence[Event], expected_seq: int = ANY_SEQ
    ) -> list[Event]:
        out = await self._run(self._append, stream, list(events), expected_seq)
        async with self._condition():
            self._condition().notify_all()
        return out

    def _read(self, stream: str, from_seq: int, limit: int | None) -> list[Event]:
        q = "SELECT seq, body FROM events WHERE stream=? AND seq>=? ORDER BY seq"
        args: tuple[Any, ...] = (stream, from_seq)
        if limit is not None:
            q, args = q + " LIMIT ?", (*args, limit)
        return [self._parse(s, b) for s, b in self._conn.execute(q, args).fetchall()]

    async def read(self, stream: str, from_seq: int = 0, limit: int | None = None) -> list[Event]:
        return await self._run(self._read, stream, from_seq, limit)

    async def last_seq(self, stream: str) -> int:
        return await self._run(
            lambda: self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM events WHERE stream=?", (stream,)
            ).fetchone()[0]
        )

    async def has(self, stream: str, event_id: str) -> bool:
        return bool(
            await self._run(
                lambda: self._conn.execute(
                    "SELECT 1 FROM events WHERE stream=? AND id=?", (stream, event_id)
                ).fetchone()
            )
        )

    def _run_index(self) -> list[RunInfo]:
        c = self._conn
        started = {
            s: self._parse(q, b)
            for s, q, b in c.execute(
                "SELECT stream, seq, body FROM events WHERE type=? AND seq=(SELECT MIN(seq) FROM events e2 WHERE e2.stream=events.stream AND e2.type=?)",
                (RUN_STARTED, RUN_STARTED),
            )
        }
        done = {
            s: self._parse(q, b)
            for s, q, b in c.execute(
                "SELECT stream, seq, body FROM events WHERE type=?", (RUN_COMPLETED,)
            )
        }
        return [RunInfo(s, e, done.get(s)) for s, e in started.items()]

    async def run_index(self) -> list[RunInfo]:
        return await self._run(self._run_index)

    async def subscribe(self, stream: str, from_seq: int = 0) -> AsyncIterator[Event]:
        pos = from_seq
        while True:
            batch = await self.read(stream, pos)
            if not batch:
                try:
                    async with self._condition():
                        await wait_for(self._condition().wait(), self._poll)
                except TimeoutError:
                    pass
                continue
            for e in batch:
                yield e
            pos = batch[-1].seq + 1

    async def load_snapshot(self, stream: str, name: str) -> StoredSnapshot | None:
        row = await self._run(
            lambda: self._conn.execute(
                "SELECT seq, version, state FROM snapshots WHERE stream=? AND name=?",
                (stream, name),
            ).fetchone()
        )
        return (
            StoredSnapshot(stream=stream, name=name, seq=row[0], version=row[1], state=row[2])
            if row
            else None
        )

    async def save_snapshot(self, snap: StoredSnapshot) -> None:
        await self._run(
            lambda: self._conn.execute(
                "INSERT INTO snapshots (stream, name, seq, version, state) VALUES (?,?,?,?,?) "
                "ON CONFLICT (stream, name) DO UPDATE SET seq=excluded.seq, version=excluded.version, "
                "state=excluded.state WHERE excluded.seq > snapshots.seq",
                (snap.stream, snap.name, snap.seq, snap.version, snap.state),
            )
        )

    def _scrub(self, stream: str, delete: bool) -> int:
        c = self._conn
        c.execute("BEGIN IMMEDIATE")
        try:
            rows = c.execute("SELECT seq, body FROM events WHERE stream=?", (stream,)).fetchall()
            if delete:
                c.execute("DELETE FROM events WHERE stream=?", (stream,))
            else:
                for seq, body in rows:
                    c.execute(
                        "UPDATE events SET body=? WHERE stream=? AND seq=?",
                        (scrubbed(self._parse(seq, body)).model_dump_json(), stream, seq),
                    )
            c.execute("DELETE FROM snapshots WHERE stream=?", (stream,))
            c.execute("COMMIT")
            return len(rows)
        except BaseException:
            c.execute("ROLLBACK")
            raise

    async def scrub(self, stream: str) -> int:
        return await self._run(self._scrub, stream, False)

    async def delete_stream(self, stream: str) -> int:
        return await self._run(self._scrub, stream, True)

    async def aclose(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, self._conn.close)
        self._pool.shutdown(wait=True)
