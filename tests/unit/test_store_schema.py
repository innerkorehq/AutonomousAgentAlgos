"""Layout versioning of the durable stores: new stores are stamped, 0.1/0.2 stores are adopted, newer ones refused, migrations run once, in order, atomically."""

import asyncio
import sqlite3
import uuid

import pytest

from aaa.events import SQLiteEventStore, migrate
from aaa.events.event import Event
from aaa.events.migrate import SchemaGap, SchemaTooNew, pending


def ev(i=0):
    return Event(type="x", stream="s", id=f"e{i}", data={"i": i})


# -- the pure planner -------------------------------------------------------------------------
def test_pending_plans_steps_and_refuses_newer_or_gaps():
    assert pending(1, 1, {}) == []
    assert pending(1, 3, {2: "a", 3: "b"}) == [2, 3]
    with pytest.raises(SchemaTooNew, match="upgrade aaa"):
        pending(5, 3, {})
    with pytest.raises(SchemaGap, match=r"\[3\]"):
        pending(1, 3, {2: "a"})


# -- SQLite -----------------------------------------------------------------------------------
async def test_sqlite_new_files_are_stamped_and_reopen_cleanly(tmp_path):
    path = tmp_path / "log.db"
    s = SQLiteEventStore(path)
    assert await s.schema_version() == migrate.SCHEMA_VERSION
    await s.append("s", [ev()])
    s2 = SQLiteEventStore(path)
    assert await s2.schema_version() == migrate.SCHEMA_VERSION and [
        e.id for e in await s2.read("s")
    ] == ["e0"]


async def test_sqlite_adopts_a_file_written_before_versions_existed(tmp_path):
    path = tmp_path / "old.db"
    s = SQLiteEventStore(path)
    await s.append("s", [ev(1)])
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version=0")  # what a 0.1/0.2 file looks like
    raw.commit()
    raw.close()
    again = SQLiteEventStore(path)
    assert await again.schema_version() == 1 and [e.id for e in await again.read("s")] == ["e1"]


async def test_sqlite_refuses_a_newer_file_and_leaves_it_untouched(tmp_path):
    path = tmp_path / "new.db"
    SQLiteEventStore(path)
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version=99")
    raw.commit()
    raw.close()
    with pytest.raises(SchemaTooNew):
        SQLiteEventStore(path)
    raw = sqlite3.connect(path)  # not left locked or half-written
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 99
    raw.close()


async def test_sqlite_runs_migrations_in_order_once_and_rolls_a_failed_step_back(
    tmp_path, monkeypatch
):
    from aaa.events import sqlite as mod

    path = tmp_path / "m.db"
    SQLiteEventStore(path)
    monkeypatch.setattr(migrate, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(
        mod,
        "MIGRATIONS",
        {
            2: "ALTER TABLE events ADD COLUMN note TEXT; CREATE INDEX events_note ON events (note)",
            3: "ALTER TABLE events ADD COLUMN extra TEXT; THIS IS NOT SQL",
        },
    )
    with pytest.raises(sqlite3.Error):
        SQLiteEventStore(path)
    raw = sqlite3.connect(path)  # step 2 committed, step 3 rolled back entirely
    cols = [r[1] for r in raw.execute("PRAGMA table_info(events)")]
    assert (
        "note" in cols
        and "extra" not in cols
        and raw.execute("PRAGMA user_version").fetchone()[0] == 2
    )
    raw.close()
    monkeypatch.setattr(
        mod, "MIGRATIONS", {2: "SELECT 1", 3: "ALTER TABLE events ADD COLUMN extra TEXT"}
    )
    s = SQLiteEventStore(path)  # resumes at step 3 only
    assert await s.schema_version() == 3
    with pytest.raises(SchemaGap):
        monkeypatch.setattr(migrate, "SCHEMA_VERSION", 5)
        SQLiteEventStore(path)


# -- PostgreSQL -------------------------------------------------------------------------------
async def test_postgres_stamps_adopts_and_refuses(pg_dsn):
    from aaa.events.postgres import PostgresEventStore

    s = await PostgresEventStore.create(pg_dsn, partitions=4)
    assert await s.schema_version() == migrate.SCHEMA_VERSION
    await s.append("s", [ev()])
    async with s.pool.acquire() as c:  # a 0.1/0.2 database has no version row
        await c.execute("DELETE FROM aaa_meta WHERE key='schema_version'")
    await s.pool.close()

    s2 = await PostgresEventStore.create(pg_dsn, partitions=4)
    assert await s2.schema_version() == 1 and [e.id for e in await s2.read("s")] == ["e0"]
    async with s2.pool.acquire() as c:
        await c.execute("UPDATE aaa_meta SET value='99' WHERE key='schema_version'")
    await s2.pool.close()
    with pytest.raises(SchemaTooNew):
        await PostgresEventStore.create(pg_dsn, partitions=4)


async def test_postgres_migrations_run_once_even_when_workers_start_together(pg_dsn, monkeypatch):
    from aaa.events import postgres as mod
    from aaa.events.postgres import PostgresEventStore

    first = await PostgresEventStore.create(pg_dsn, partitions=4)
    await first.pool.close()
    monkeypatch.setattr(migrate, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        mod, "MIGRATIONS", {2: ["ALTER TABLE aaa_events ADD COLUMN note text"]}
    )  # a second run would fail: the column exists
    stores = await asyncio.gather(
        *(PostgresEventStore.create(pg_dsn, partitions=4) for _ in range(4))
    )
    assert {await s.schema_version() for s in stores} == {2}
    for s in stores:
        await s.pool.close()


async def test_postgres_a_failing_step_is_atomic(pg_dsn, monkeypatch):
    from aaa.events import postgres as mod
    from aaa.events.postgres import PostgresEventStore

    first = await PostgresEventStore.create(pg_dsn, partitions=4)
    await first.pool.close()
    monkeypatch.setattr(migrate, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        mod, "MIGRATIONS", {2: ["ALTER TABLE aaa_events ADD COLUMN note text", "THIS IS NOT SQL"]}
    )
    with pytest.raises(Exception):  # noqa: B017 - asyncpg's syntax error
        await PostgresEventStore.create(pg_dsn, partitions=4)
    monkeypatch.setattr(migrate, "SCHEMA_VERSION", 1)
    monkeypatch.setattr(mod, "MIGRATIONS", {})
    s = await PostgresEventStore.create(pg_dsn, partitions=4)
    async with s.pool.acquire() as c:
        cols = {
            r["column_name"]
            for r in await c.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name='aaa_events'"
            )
        }
    assert (
        "note" not in cols and await s.schema_version() == 1
    )  # the ALTER was rolled back with the failed step
    await s.pool.close()


# -- Redis ------------------------------------------------------------------------------------
async def test_redis_stamps_adopts_refuses_and_migrates(redis_base_url, monkeypatch):
    from aaa.events import redis as mod
    from aaa.events.redis import RedisEventStore

    prefix = "t" + uuid.uuid4().hex[:8]
    s = await RedisEventStore.create(redis_base_url, partitions=4, prefix=prefix)
    assert await s.schema_version() == migrate.SCHEMA_VERSION
    await s.r.delete(f"{prefix}:meta:schema_version")  # a 0.1/0.2 prefix has no version key
    await s.r.aclose()
    s = await RedisEventStore.create(redis_base_url, partitions=4, prefix=prefix)
    assert await s.schema_version() == 1

    ran = []

    async def step2(r, p):
        ran.append(p)

    monkeypatch.setattr(migrate, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(mod, "MIGRATIONS", {2: step2})
    s2 = await RedisEventStore.create(redis_base_url, partitions=4, prefix=prefix)
    assert await s2.schema_version() == 2 and ran == [prefix]

    monkeypatch.setattr(migrate, "SCHEMA_VERSION", 1)
    monkeypatch.setattr(mod, "MIGRATIONS", {})
    with pytest.raises(SchemaTooNew):
        await RedisEventStore.create(redis_base_url, partitions=4, prefix=prefix)
    await s.r.aclose()
    await s2.r.aclose()
