import pytest
from pydantic import BaseModel

from aaa.core import ConcurrencyError, SchemaError, new_id
from aaa.events import Event, InMemoryEventStore, InProcessBus, Projection, SchemaRegistry


def test_ids_monotonic():
    ids = [new_id() for _ in range(1000)]
    assert ids == sorted(ids) and len(set(ids)) == 1000


async def test_store_occ_and_read():
    s = InMemoryEventStore()
    out = await s.append("r", [Event(type="a"), Event(type="b")], expected_seq=-1)
    assert [e.seq for e in out] == [0, 1]
    with pytest.raises(ConcurrencyError):
        await s.append("r", [Event(type="c")], expected_seq=-1)
    await s.append("r", [Event(type="c")], expected_seq=1)
    assert [e.type for e in await s.read("r", 1)] == ["b", "c"]


async def test_subscribe_replays_then_tails():
    s = InMemoryEventStore()
    await s.append("r", [Event(type="a")])
    it = s.subscribe("r")
    assert (await anext(it)).type == "a"
    await s.append("r", [Event(type="b")])
    assert (await anext(it)).type == "b"


async def test_snapshot_plus_tail_equals_full_replay():
    s = InMemoryEventStore()
    fold = lambda st, e: st + e.data["n"]
    p = Projection(lambda: 0, fold, name="sum", state_type=int, every=2)
    await s.append("r", [Event(type="x", data={"n": i}) for i in range(1, 6)])
    first = await p.load(s, "r")
    assert (first.state, first.seq, first.from_snapshot, first.replayed) == (
        15,
        4,
        -1,
        5,
    )  # snapshot saved
    await s.append("r", [Event(type="x", data={"n": 10})])
    second = await p.load(s, "r")
    assert (second.state, second.from_snapshot, second.replayed) == (25, 4, 1)  # only the tail
    plain = Projection(lambda: 0, fold)
    assert (await plain.load(s, "r")).state == second.state == 25  # == full replay
    stale = Projection(lambda: 0, fold, name="sum", version=2, state_type=int, every=2)
    assert (await stale.load(s, "r")).from_snapshot == -1  # version bump ignores


async def test_bus_pattern_followups_and_isolation():
    seen: list[str] = []

    async def req(e: Event):
        seen.append(e.type)
        return [e.caused("aaa.llm.completed")]

    async def done(e: Event):
        seen.append(e.type)

    async def boom(e: Event):
        raise RuntimeError("bad handler")

    async with InProcessBus() as bus:
        bus.subscribe("aaa.llm.requested", req)
        bus.subscribe("aaa.llm.*", done)
        bus.subscribe("aaa.*", boom)
        await bus.publish(Event(type="aaa.llm.requested"))
        await bus.drain()
    assert seen.count("aaa.llm.completed") == 1 and "aaa.llm.requested" in seen


async def test_bus_group_competes():
    got: list[int] = []

    def mk(i):
        async def h(e):
            got.append(i)

        return h

    async with InProcessBus() as bus:
        bus.subscribe("t", mk(1), group="g")
        bus.subscribe("t", mk(2), group="g")
        for _ in range(4):
            await bus.publish(Event(type="t"))
        await bus.drain()
    assert sorted(got) == [1, 1, 2, 2]


def test_schema_upcast():
    class V2(BaseModel):
        text: str
        lang: str = "en"

    r = SchemaRegistry()
    r.register("user.note/v2", V2)
    r.upcaster("user.note", 1, lambda d: {"text": d["body"]})
    e = r.validate(Event(type="user.note", schema_ref="user.note/v1", data={"body": "hi"}))
    assert e.schema_ref == "user.note/v2" and e.data == {"text": "hi", "lang": "en"}
    with pytest.raises(SchemaError):
        r.validate(Event(type="user.note", schema_ref="user.note/v2", data={}))
