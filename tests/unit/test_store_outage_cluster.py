"""Store outages in a cluster: a worker that cannot record a result leaves the message unacknowledged
(another worker finishes the call), the bus redelivers reactions lost to an outage, a consumer never dies
because the backend hiccuped, and REAL dropped connections (Postgres/Redis) are ridden out."""

import asyncio
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from test_store_outage import FlakyStore

from aaa.algos import ToolLoop
from aaa.core import RetryPolicy
from aaa.events import DistributedBus, Event
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import GuardConfig, ToolHandler, ToolRegistry, tool
from cluster import FAST as BUS_FAST
from cluster import until

GUARD = GuardConfig(lease_ttl_s=1.0, heartbeat_s=0.2, wait_poll_s=0.1)


def script(req):
    last = req.messages[-1]
    return tool_call("job") if last.role == "user" else text(f"done: {last.content}")


async def boot(node, member, tools, *, store=None, store_retry=None, result_retry=None):
    bus = DistributedBus(
        node.backend, member=member, redeliver_base_s=0.05, redeliver_max_s=0.2, **BUS_FAST
    )
    eng = Engine(store=store or node.store, bus=bus, store_retry=store_retry)
    await eng.__aenter__()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry(tools)
    ToolHandler(reg, guard=GUARD, result_retry=result_retry).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    return eng, algo


async def test_a_worker_that_cannot_record_a_tool_result_leaves_it_for_another_worker(cluster):
    ran: list[str] = []
    gate_a = asyncio.Event()

    def mk(member):
        @tool
        async def job() -> str:
            """work"""
            ran.append(member)
            if member == "A":
                await gate_a.wait()
            return f"ran-on-{member}"

        return job

    n1, n2 = await cluster.node(), await cluster.node()
    flaky = FlakyStore(n1.store)
    e1, a1 = await boot(
        n1,
        "A",
        [mk("A")],
        store=flaky,
        store_retry=RetryPolicy(max_elapsed_s=0.2, base_delay_s=0.02),
        result_retry=RetryPolicy(
            max_elapsed_s=0.4, base_delay_s=0.05
        ),  # bounded: give up, don't ack
    )
    run = await e1.start(a1, {"task": "go"}, run_id="outage-1")
    await until(lambda: ran == ["A"], 15)
    flaky.down = True  # A's store goes away...
    gate_a.set()  # ...just as its tool finishes
    await asyncio.sleep(2.0)  # A retries, exhausts its budget, logs, does NOT acknowledge
    seen = await n2.store.read(run.run_id)  # (read through the healthy node)
    assert not any(e.type in ("aaa.tool.completed", "aaa.tool.failed") for e in seen)
    await e1.abort()  # A is restarted/replaced; its store is still down

    e2, _ = await boot(n2, "B", [mk("B")])
    out = await asyncio.wait_for(e2.handle(run.run_id).result(), 40)
    log = await n2.store.read(run.run_id)
    await e2.__aexit__(None, None, None)
    c = Counter(e.type for e in log)
    assert (
        out.ok and out.value["text"] == "done: ran-on-B"
    )  # the redelivered request was finished by B
    assert c["aaa.tool.completed"] == 1 and c["aaa.run.completed"] == 1
    assert [e.data["attempt"] for e in log if e.type == "aaa.tool.started"] == [1, 2]


async def test_the_bus_redelivers_a_reaction_lost_to_an_outage_instead_of_failing_the_run(cluster):
    @tool
    async def job() -> str:
        """work"""
        return "ok"

    node = await cluster.node()
    flaky = FlakyStore(node.store)
    eng, algo = await boot(
        node,
        "A",
        [job],
        store=flaky,
        store_retry=RetryPolicy(max_elapsed_s=0.15, base_delay_s=0.02),
    )
    run = await eng.start(algo, {"task": "go"}, run_id="outage-2", budget=Budget(deadline_s=60))
    flaky.down = True  # the very first reaction hits the outage
    await asyncio.sleep(1.5)  # outlasts the local retry budget several times
    flaky.down = False
    out = await asyncio.wait_for(run.result(), 30)
    log = await node.store.read(run.run_id)
    await eng.__aexit__(None, None, None)
    assert out.ok and out.value["text"] == "done: ok"
    assert len({e.id for e in log}) == len(log)  # redelivery produced no duplicate events


class FlakyBackend:
    """Bus backend whose reads/cursor writes raise while `down` (the consumer must survive that)."""

    def __init__(self, real):
        self._real, self.down = real, False

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def read(self, *a, **k):
        if self.down:
            await asyncio.sleep(0.02)
            raise ConnectionError("backend down")
        return await self._real.read(*a, **k)

    async def get_cursor(self, *a):
        if self.down:
            raise ConnectionError("backend down")
        return await self._real.get_cursor(*a)

    async def set_cursor(self, *a):
        if self.down:
            raise ConnectionError("backend down")
        return await self._real.set_cursor(*a)


async def test_a_backend_outage_never_kills_a_consumer(cluster):
    node = await cluster.node()
    backend = FlakyBackend(node.backend)
    got: list[int] = []

    async def h(e):
        got.append(e.data["i"])

    bus = DistributedBus(backend, member="A", **BUS_FAST)
    bus.subscribe("t", h, group="g")
    async with bus:
        await node.store.append("s", [Event(id="a0", type="t", data={"i": 0})])
        await until(lambda: got == [0], 15)
        backend.down = True
        await asyncio.sleep(0.8)  # consumer loops hit errors, must keep going
        await node.store.append("s", [Event(id="a1", type="t", data={"i": 1})])
        backend.down = False
        await node.store.append("s", [Event(id="a2", type="t", data={"i": 2})])
        await until(lambda: got == [0, 1, 2], 20)  # nothing lost, nothing repeated, order kept


async def _kill_connections(cluster):
    if cluster.kind == "postgres":
        import asyncpg

        admin = await asyncpg.connect(cluster.admin_dsn)
        await admin.fetch(
            "select pg_terminate_backend(pid) from pg_stat_activity where datname=$1 and pid<>pg_backend_pid()",
            cluster._db,
        )
        await admin.close()
    else:
        import redis.asyncio as aioredis

        admin = aioredis.from_url(cluster.url)
        await admin.execute_command("CLIENT", "KILL", "TYPE", "normal", "SKIPME", "yes")
        await admin.aclose()


async def test_real_dropped_connections_mid_run_are_ridden_out(cluster):
    """Kill every database/Redis connection repeatedly while a run is in progress."""
    if cluster.kind == "memory":
        pytest.skip("needs a real server")
    ran: list[int] = []

    @tool
    async def job() -> str:
        """slow work"""
        await asyncio.sleep(1.2)
        ran.append(1)
        return "ok"

    node = await cluster.node()
    eng, algo = await boot(node, "A", [job])
    run = await eng.start(algo, {"task": "go"}, run_id="real-kill", budget=Budget(deadline_s=90))
    for _ in range(4):
        await asyncio.sleep(0.35)
        await _kill_connections(cluster)  # LLM, tool, bus, lease and listener connections all die
    out = await asyncio.wait_for(run.result(), 60)
    log = await eng.retry(lambda: node.store.read(run.run_id), "read")
    await eng.__aexit__(None, None, None)
    c = Counter(e.type for e in log)
    assert out.ok and out.value["text"] == "done: ok"
    assert c["aaa.tool.completed"] == 1 and c["aaa.run.completed"] == 1
    assert len({e.id for e in log}) == len(log)
