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


# -- erasure ------------------------------------------------------------------------------------
def _ev(i, stream, **kw):
    return Event(
        id=f"{stream}-{i}",
        type=kw.pop("type", "x.event"),
        stream=stream,
        run_id=stream,
        tenant="acme",
        subject="someone@example.com",
        data={"secret": f"payload-{i}", "n": i},
        meta={"label": "pii"},
        **kw,
    )


async def _seed(store, stream):
    await store.append(
        stream,
        [
            _ev(0, stream, type="aaa.run.started").model_copy(
                update={"data": {"algo": "a", "input": {"ssn": "123"}, "parent": None}}
            ),
            _ev(1, stream),
            _ev(2, stream, type="aaa.run.completed").model_copy(
                update={"data": {"kind": "success", "value": {"pii": "x"}, "reason": None}}
            ),
        ],
    )


async def test_scrub_erases_payloads_keeps_the_skeleton_and_is_idempotent(store):
    await _seed(store, "gone")
    await _seed(store, "kept")
    await store.save_snapshot(
        StoredSnapshot(stream="gone", name="n", seq=1, version=1, state='{"pii": 1}')
    )
    assert await store.scrub("gone") == 3
    assert await store.scrub("gone") == 3  # idempotent
    after = await store.read("gone")
    assert [e.id for e in after] == ["gone-0", "gone-1", "gone-2"]
    assert [e.seq for e in after] == [0, 1, 2]
    assert [e.type for e in after] == ["aaa.run.started", "x.event", "aaa.run.completed"]
    assert all(e.tenant == "acme" and e.run_id == "gone" for e in after)
    assert all(e.data.get("$erased") is True and e.meta == {} and e.subject is None for e in after)
    assert "ssn" not in str(after) and "payload" not in str(after) and "pii" not in str(after)
    assert after[0].data["algo"] == "a" and after[2].data["kind"] == "success"
    assert await store.load_snapshot("gone", "n") is None
    kept = await store.read("kept")  # other streams are untouched
    assert kept[1].data == {"secret": "payload-1", "n": 1} and kept[1].subject
    # structure queries still work on the scrubbed run, and it is still a finished run
    info = {r.stream: r for r in await store.run_index()}["gone"]
    assert info.started.data["algo"] == "a" and info.completed is not None
    assert await store.has("gone", "gone-1")
    more = await store.append("gone", [Event(id="gone-3", type="aaa.run.erased")])
    assert more[0].seq == 3  # the stream stays appendable (the erasure marker goes here)


async def test_delete_stream_removes_everything_and_the_id_can_be_reused(store):
    await _seed(store, "doomed")
    await _seed(store, "other")
    await store.save_snapshot(
        StoredSnapshot(stream="doomed", name="n", seq=1, version=1, state="{}")
    )
    assert await store.delete_stream("doomed") == 3
    assert await store.read("doomed") == [] and await store.last_seq("doomed") == -1
    assert not await store.has("doomed", "doomed-1")
    assert await store.load_snapshot("doomed", "n") is None
    assert {r.stream for r in await store.run_index()} == {"other"}
    assert len(await store.read("other")) == 3
    again = await store.append("doomed", [Event(id="fresh", type="x.event")], expected_seq=-1)
    assert again[0].seq == 0 and not again[0].dup


async def test_erasure_also_drops_the_bus_copies_of_the_events(store):
    """The transactional outbox holds a second copy of every event until it is trimmed."""
    if not hasattr(store, "bus_backend"):
        pytest.skip("this store has no outbox")
    from aaa.events import partition_of

    await _seed(store, "leaky")
    backend = store.bus_backend()
    p = partition_of("leaky", store.partitions)

    async def copies():
        entries = await backend.read(p, 0, 100, 0)
        return [e for _id, e in entries if e.stream == "leaky" and "payload" in str(e.data)]

    assert await copies()  # the outbox really holds the payloads
    await store.scrub("leaky")
    assert not await copies()
