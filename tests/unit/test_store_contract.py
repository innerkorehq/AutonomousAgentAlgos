import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aaa.core import ConcurrencyError
from aaa.events import Event, InMemoryEventStore, SQLiteEventStore, StoredSnapshot


@pytest.fixture(params=["memory", "sqlite", "postgres", "redis"])
async def store(request, tmp_path, prefix):
    from cluster import Cluster

    kind = request.param
    cluster = None
    if kind == "memory":
        s = InMemoryEventStore()
    elif kind == "sqlite":
        s = SQLiteEventStore(tmp_path / "e.db")
    elif kind == "postgres":
        cluster = Cluster("postgres", admin_dsn=request.getfixturevalue("pg_admin_dsn"))
        await cluster.setup()
        s = (await cluster.node()).store
    else:
        cluster = Cluster("redis", url=request.getfixturevalue("redis_url"), prefix=prefix)
        s = (await cluster.node()).store
    yield s
    if cluster:
        await cluster.close()
    elif hasattr(s, "aclose"):
        await s.aclose()


async def test_append_read_occ(store):
    out = await store.append("r", [Event(type="a"), Event(type="b")], expected_seq=-1)
    assert [e.seq for e in out] == [0, 1] and all(not e.dup for e in out)
    with pytest.raises(ConcurrencyError):
        await store.append("r", [Event(type="c")], expected_seq=-1)
    assert [e.type for e in await store.read("r")] == ["a", "b"]  # failed append left no trace
    await store.append("r", [Event(type="c")], expected_seq=1)
    assert [e.type for e in await store.read("r", 1, limit=1)] == ["b"]
    assert await store.last_seq("r") == 2 and await store.last_seq("empty") == -1


async def test_append_is_idempotent_on_event_id(store):
    e = Event(type="x", data={"n": 1})
    first = await store.append("r", [e])
    again = await store.append("r", [e, Event(type="y")])
    assert (
        again[0].dup
        and again[0].seq == first[0].seq == 0
        and not again[1].dup
        and again[1].seq == 1
    )
    assert len(await store.read("r")) == 2
    assert "dup" not in (await store.read("r"))[0].model_dump()  # transient flag never persisted


async def test_subscribe_replays_then_tails(store):
    await store.append("r", [Event(type="a")])
    it = store.subscribe("r")
    assert (await asyncio.wait_for(anext(it), 2)).type == "a"
    await store.append("r", [Event(type="b")])
    assert (await asyncio.wait_for(anext(it), 2)).type == "b"


async def test_run_index(store):
    await store.append(
        "r1",
        [Event(type="aaa.run.started", run_id="r1"), Event(type="aaa.run.completed", run_id="r1")],
    )
    await store.append("r2", [Event(type="aaa.run.started", run_id="r2"), Event(type="other")])
    await store.append("noise", [Event(type="other")])
    idx = {r.stream: r for r in await store.run_index()}
    assert set(idx) == {"r1", "r2"} and idx["r1"].completed and idx["r2"].completed is None


async def test_caused_ids_are_deterministic():
    p = Event(type="p")
    assert p.caused("x", {"a": 1}).id == p.caused("x", {"a": 1}).id
    assert p.caused("x", {"a": 1}).id != p.caused("x", {"a": 2}).id
    assert p.caused("x", {"a": 1}).id != Event(type="p").caused("x", {"a": 1}).id


async def test_sqlite_survives_reopen_and_rolls_back_failed_append(tmp_path):
    path = tmp_path / "d.db"
    s = SQLiteEventStore(path)
    await s.append("r", [Event(type="a", data={"k": "v"}, meta={"t": "x"})])
    with pytest.raises(ConcurrencyError):
        await s.append("r", [Event(type="lost")], expected_seq=99)
    await s.aclose()
    s2 = SQLiteEventStore(path)
    log = await s2.read("r")
    assert (
        [e.type for e in log] == ["a"] and log[0].data == {"k": "v"} and log[0].meta == {"t": "x"}
    )
    await s2.aclose()
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    con.close()


async def test_sqlite_concurrent_appends_get_unique_contiguous_seqs(tmp_path):
    s = SQLiteEventStore(tmp_path / "c.db")
    await asyncio.gather(*(s.append("r", [Event(type="e", data={"i": i})]) for i in range(50)))
    assert [e.seq for e in await s.read("r")] == list(range(50))
    await s.aclose()


async def test_snapshots_roundtrip_and_are_monotonic(store):
    mk = lambda seq, v=1: StoredSnapshot(
        stream="r", name="n", seq=seq, version=v, state=f'{{"s":{seq}}}'
    )
    assert await store.load_snapshot("r", "n") is None
    await store.save_snapshot(mk(10))
    await store.save_snapshot(mk(5))  # stale writer must not roll back
    got = await store.load_snapshot("r", "n")
    assert (got.seq, got.version, got.state) == (10, 1, '{"s":10}')
    await store.save_snapshot(mk(11, v=2))
    assert (await store.load_snapshot("r", "n")).seq == 11
    assert (
        await store.load_snapshot("r", "other") is None
        and await store.load_snapshot("zz", "n") is None
    )


async def test_unicode_big_and_nested_payloads_roundtrip(store):
    data = {
        "text": "héllo 世界 🚀",
        "big": "x" * 200_000,
        "nested": {"a": [1, 2.5, None, True], "k": {"z": "q"}},
    }
    out = await store.append("u", [Event(type="x", data=data, meta={"tenant": "acme", "w": 0.5})])
    got = (await store.read("u"))[0]
    assert (
        got.data == data and got.meta == {"tenant": "acme", "w": 0.5} and got.seq == out[0].seq == 0
    )


async def test_occ_race_exactly_one_writer_wins(store):
    results = await asyncio.gather(
        *(
            store.append("race", [Event(type="x", data={"i": i})], expected_seq=-1)
            for i in range(8)
        ),
        return_exceptions=True,
    )
    ok = [r for r in results if not isinstance(r, Exception)]
    assert len(ok) == 1 and all(
        isinstance(r, ConcurrencyError) for r in results if isinstance(r, Exception)
    )
    assert len(await store.read("race")) == 1


async def test_many_concurrent_appends_keep_contiguous_order(store):
    await asyncio.gather(*(store.append("c", [Event(type="e", data={"i": i})]) for i in range(40)))
    log = await store.read("c")
    assert [e.seq for e in log] == list(range(40)) and sorted(e.data["i"] for e in log) == list(
        range(40)
    )


async def test_has_is_an_exact_per_stream_id_lookup(store):
    await store.append("h1", [Event(id="a", type="x"), Event(id="b", type="x")])
    await store.append("h2", [Event(id="c", type="x")])
    assert await store.has("h1", "a") and await store.has("h1", "b") and await store.has("h2", "c")
    assert (
        not await store.has("h1", "c")
        and not await store.has("h2", "a")
        and not await store.has("nope", "a")
    )
