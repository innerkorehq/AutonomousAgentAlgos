import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa.events import Event
from cluster import until


async def put(node, stream, i, **data):
    await node.store.append(stream, [Event(id=f"{stream}-{i}", type="t", data={"i": i, **data})])


async def test_delivery_preserves_per_stream_order(cluster):
    node = await cluster.node()
    got: dict[str, list[int]] = {}

    async def h(e):
        got.setdefault(e.stream, []).append(e.data["i"])

    bus = node.bus()
    bus.subscribe("t", h, group="g")
    async with bus:
        for i in range(12):
            for s in ("a", "b", "c", "d", "e"):
                await put(node, s, i)
        await until(lambda: sum(map(len, got.values())) == 60)
    assert all(v == list(range(12)) for v in got.values()) and len(got) == 5


async def test_competing_consumers_handle_each_event_exactly_once(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    handled: list[tuple[str, str]] = []

    def mk(name):
        async def h(e):
            handled.append((name, e.id))

        return h

    b1, b2 = n1.bus(member="m1"), n2.bus(member="m2")
    b1.subscribe("t", mk("m1"), group="g")
    b2.subscribe("t", mk("m2"), group="g")
    async with b1, b2:
        await until(
            lambda: (
                b1.owned_partitions.get("g")
                and b2.owned_partitions.get("g")
                and len(b1.owned_partitions["g"]) + len(b2.owned_partitions["g"])
                == cluster.partitions
            ),
            15,
        )
        for i in range(10):
            for s in range(20):
                await put(n1, f"s{s}", i)
        await until(lambda: len(handled) >= 200, 15)
        await asyncio.sleep(0.5)  # nothing arrives twice
    ids = [i for _, i in handled]
    assert len(ids) == len(set(ids)) == 200
    assert {n for n, _ in handled} == {"m1", "m2"}  # work really was split


async def test_broadcast_subscribers_each_see_every_event(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    seen = {"a": [], "b": []}

    def mk(k):
        async def h(e):
            seen[k].append(e.id)

        return h

    b1, b2 = n1.bus(), n2.bus()
    b1.subscribe("t", mk("a"))  # group=None => broadcast
    b2.subscribe("t", mk("b"))
    async with b1, b2:
        await asyncio.sleep(0.4)
        for i in range(15):
            await put(n1, f"s{i % 4}", i)
        await until(lambda: len(seen["a"]) == 15 and len(seen["b"]) == 15)


async def test_crash_redelivers_unacknowledged_events_in_order(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    a_seen, b_seen = [], []
    block = asyncio.Event()

    async def ha(e):
        a_seen.append(e.data["i"])
        if e.data["i"] == 5:
            await block.wait()  # A is stuck on event 5 when it "crashes"

    async def hb(e):
        b_seen.append(e.data["i"])

    ba = n1.bus(member="A")
    ba.subscribe("t", ha, group="g")
    await ba.__aenter__()
    for i in range(10):
        await put(n1, "stream-x", i)
    await until(lambda: 5 in a_seen)
    await ba.abort()  # no lease release, no cursor commit for 5..9

    bb = n2.bus(member="B")
    bb.subscribe("t", hb, group="g")
    async with bb:
        await until(lambda: len(b_seen) >= 5, 20)  # waits for A's lease to expire
    assert b_seen[:5] == [5, 6, 7, 8, 9]  # exactly the unacked tail, in order
    assert a_seen == [0, 1, 2, 3, 4, 5]


async def test_concurrency_window_commits_only_the_contiguous_prefix(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    done_a, got_b = [], []
    stuck = asyncio.Event()

    async def ha(e):
        if e.data["i"] == 2:
            await stuck.wait()
        done_a.append(e.data["i"])

    async def hb(e):
        got_b.append(e.data["i"])

    ba = n1.bus(member="A")
    ba.subscribe("t", ha, group="g", concurrency=4)
    await ba.__aenter__()
    for i in range(8):
        await put(n1, "w", i)
    await until(lambda: len(done_a) == 7)  # everything except event 2 finished out of order
    await asyncio.sleep(0.3)
    await ba.abort()

    bb = n2.bus(member="B")
    bb.subscribe("t", hb, group="g", concurrency=4)
    async with bb:
        await until(lambda: {2, 3, 4, 5, 6, 7} <= set(got_b), 20)
    assert 0 not in got_b and 1 not in got_b  # 0,1 were durably acked; 2.. were redelivered


async def test_rebalance_on_join_and_fast_handover_on_graceful_leave(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    P = cluster.partitions

    async def h(e):
        return None

    b1, b2 = n1.bus(member="A"), n2.bus(member="B")
    b1.subscribe("t", h, group="g")
    b2.subscribe("t", h, group="g")
    await b1.__aenter__()
    await until(lambda: len(b1.owned_partitions.get("g", [])) == P)  # alone: owns all
    await b2.__aenter__()
    await until(
        lambda: (
            len(b1.owned_partitions.get("g", [])) == P // 2
            and len(b2.owned_partitions.get("g", [])) == P // 2
        ),
        20,
    )
    assert not set(b1.owned_partitions["g"]) & set(b2.owned_partitions["g"])  # never both
    await b2.__aexit__(None, None, None)  # graceful: leases released
    await until(lambda: len(b1.owned_partitions.get("g", [])) == P, 15)
    await b1.__aexit__(None, None, None)


async def test_a_failing_handler_does_not_stop_the_stream(cluster):
    node = await cluster.node()
    got = []

    async def h(e):
        if e.data["i"] == 3:
            raise RuntimeError("poison")
        got.append(e.data["i"])

    bus = node.bus()
    bus.subscribe("t", h, group="g")
    async with bus:
        for i in range(8):
            await put(node, "p", i)
        await until(lambda: len(got) == 7)
    assert got == [0, 1, 2, 4, 5, 6, 7]


async def test_duplicate_append_is_not_enqueued_twice(cluster):
    node = await cluster.node()
    got = []

    async def h(e):
        got.append(e.id)

    bus = node.bus()
    bus.subscribe("t", h, group="g")
    async with bus:
        for _ in range(3):
            await put(node, "d", 0)  # same event id three times
        await put(node, "d", 1)
        await until(lambda: len(got) >= 2)
        await asyncio.sleep(0.3)
    assert got == ["d-0", "d-1"]


def test_partition_function_is_stable():
    from aaa.events import partition_of

    assert partition_of("run-123", 16) == partition_of("run-123", 16)
    assert {partition_of(f"s{i}", 16) for i in range(200)} == set(range(16))
