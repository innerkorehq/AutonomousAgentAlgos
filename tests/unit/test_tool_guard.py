"""The stalled-worker guard: durable ack, exclusive lease, numbered attempts (fencing), delivery
policy, stale-worker behavior, idempotency keys, and configuration."""

import asyncio
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pydantic import ValidationError

from aaa.algos import ToolLoop
from aaa.events import DistributedBus, SQLiteEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import GuardConfig, ToolContext, ToolHandler, ToolRegistry, tool
from cluster import FAST as BUS_FAST
from cluster import until

FAST = GuardConfig(lease_ttl_s=1.0, heartbeat_s=0.2, wait_poll_s=0.1)


def script(req):
    last = req.messages[-1]
    return tool_call("job", order="o-17") if last.role == "user" else text(f"done: {last.content}")


class Freezable:
    """Wraps a bus backend to model a whole-process stall: while frozen, every lease operation
    (partition renewals, execution heartbeats) hangs, so the process silently stops renewing."""

    def __init__(self, real):
        self._real, self.go = real, asyncio.Event()
        self.go.set()

    def __getattr__(self, name):
        return getattr(self._real, name)

    def freeze(self):
        self.go.clear()

    def thaw(self):
        self.go.set()

    async def try_lease(self, *a, **k):
        await self.go.wait()
        return await self._real.try_lease(*a, **k)


async def boot(node, member, tools, *, guard=FAST, freezable=False):
    backend = Freezable(node.backend) if freezable else node.backend
    bus = DistributedBus(backend, member=member, **BUS_FAST)
    eng = Engine(store=node.store, bus=bus)
    await eng.__aenter__()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry(tools)
    handler = ToolHandler(reg, guard=guard)
    handler.attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    return eng, algo, backend


def types(log):
    return Counter(e.type for e in log)


# --------------------------------------------------------------------------- durable ack
async def test_worker_death_mid_tool_call_is_redelivered_and_run_once_more(cluster):
    ran: list[str] = []

    def mk(member, block):
        @tool
        async def job(order: str) -> str:
            """a job"""
            ran.append(member)
            if block:
                await asyncio.Event().wait()
            return f"ran-on-{member}"

        return job

    n1, n2 = await cluster.node(), await cluster.node()
    e1, a1, _ = await boot(n1, "A", [mk("A", True)])
    h = await e1.start(a1, {"task": "go"}, run_id="mid-tool")
    await until(lambda: ran == ["A"], 10)
    await asyncio.sleep(0.4)
    await e1.abort()

    e2, _, _ = await boot(n2, "B", [mk("B", False)])
    out = await asyncio.wait_for(e2.handle(h.run_id).result(), 30)
    log = await n2.store.read(h.run_id)
    await e2.__aexit__(None, None, None)
    assert out.ok and out.value["text"] == "done: ran-on-B" and ran == ["A", "B"]
    started = [e for e in log if e.type == "aaa.tool.started"]
    assert [e.data["attempt"] for e in started] == [1, 2] and types(log)["aaa.tool.completed"] == 1


# --------------------------------------------------------------------------- delivery policy
async def test_at_most_once_never_reruns_a_lost_attempt(cluster):
    ran: list[str] = []

    def mk(member, block):
        @tool(delivery="at_most_once")
        async def job(order: str) -> str:
            """sends an email"""
            ran.append(member)
            if block:
                await asyncio.Event().wait()
            return "sent"

        return job

    n1, n2 = await cluster.node(), await cluster.node()
    e1, a1, _ = await boot(n1, "A", [mk("A", True)])
    h = await e1.start(a1, {"task": "go"}, run_id="amo")
    await until(lambda: ran == ["A"], 10)
    await asyncio.sleep(0.4)
    await e1.abort()

    e2, _, _ = await boot(n2, "B", [mk("B", False)])
    out = await asyncio.wait_for(e2.handle(h.run_id).result(), 30)
    log = await n2.store.read(h.run_id)
    await e2.__aexit__(None, None, None)
    assert ran == ["A"]  # B refused to run it again
    assert out.ok and "OutcomeUnknown" in out.value["text"] and "NOT retried" in out.value["text"]
    failed = next(e for e in log if e.type == "aaa.tool.failed")
    assert failed.data["error"]["type"] == "OutcomeUnknown"


async def test_max_attempts_stops_a_call_that_keeps_killing_workers(cluster):
    ran: list[str] = []

    def mk(member, block):
        @tool(guard={"max_attempts": 2})
        async def job(order: str) -> str:
            """a poison job"""
            ran.append(member)
            if block:
                await asyncio.Event().wait()
            return "never reached"

        return job

    nodes = [await cluster.node() for _ in range(3)]
    e1, a1, _ = await boot(nodes[0], "A", [mk("A", True)])
    h = await e1.start(a1, {"task": "go"}, run_id="poison")
    await until(lambda: ran == ["A"], 10)
    await asyncio.sleep(0.4)
    await e1.abort()
    e2, _, _ = await boot(nodes[1], "B", [mk("B", True)])  # attempt 2 also dies
    await until(lambda: ran == ["A", "B"], 20)
    await asyncio.sleep(0.4)
    await e2.abort()

    e3, _, _ = await boot(nodes[2], "C", [mk("C", False)])  # attempt 3 is refused
    out = await asyncio.wait_for(e3.handle(h.run_id).result(), 30)
    await e3.__aexit__(None, None, None)
    assert ran == ["A", "B"] and out.ok and "ExecutionAbandoned" in out.value["text"]


# --------------------------------------------------------------------------- stalled workers
async def _stalled_scenario(cluster, *, guard, a_finishes_first):
    ran: list[str] = []
    cancelled: list[str] = []
    gate_a, gate_b = asyncio.Event(), asyncio.Event()

    def mk(member):
        gate = gate_a if member == "A" else gate_b

        @tool
        async def job(order: str) -> str:
            """a job"""
            ran.append(member)
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled.append(member)
                raise
            return f"ran-on-{member}"

        return job

    n1, n2 = await cluster.node(), await cluster.node()
    e1, a1, backend_a = await boot(n1, "A", [mk("A")], guard=guard, freezable=True)
    h = await e1.start(a1, {"task": "go"}, run_id="stalled")
    await until(lambda: ran == ["A"], 10)
    await asyncio.sleep(0.4)
    backend_a.freeze()  # A stops renewing anything: a full process stall
    e2, _, _ = await boot(n2, "B", [mk("B")], guard=guard)
    return SimpleNamespace(
        ran=ran,
        cancelled=cancelled,
        gate_a=gate_a,
        gate_b=gate_b,
        backend_a=backend_a,
        e1=e1,
        e2=e2,
        h=h,
        n2=n2,
    )


async def test_stalled_worker_is_cancelled_when_it_wakes_and_only_one_result_exists(cluster):
    if cluster.kind == "postgres":
        pytest.skip("Postgres leases are session-bound: see the next test")
    sc = await _stalled_scenario(cluster, guard=FAST, a_finishes_first=False)
    await until(lambda: sc.ran == ["A", "B"], 30)  # B took over while A was frozen: real overlap
    sc.backend_a.thaw()  # A wakes WHILE the run is still open: its heartbeat now fails
    await until(lambda: sc.cancelled == ["A"], 10)  # the stale tool is cancelled, not left running

    async def audited():
        return any(e.type == "aaa.tool.fenced" for e in await sc.n2.store.read(sc.h.run_id))

    await until(audited, 10)  # ...and the cancellation is recorded for audit
    assert not sc.gate_b.is_set()  # B is still working: the cancel came from the lost lease alone
    sc.gate_b.set()
    out = await asyncio.wait_for(sc.e2.handle(sc.h.run_id).result(), 30)
    log = await sc.n2.store.read(sc.h.run_id)
    await sc.e1.abort()
    await sc.e2.__aexit__(None, None, None)
    c = types(log)
    assert out.ok and out.value["text"] == "done: ran-on-B"
    assert c["aaa.tool.completed"] == 1 and c["aaa.tool.started"] == 2
    assert [e.data["attempt"] for e in log if e.type == "aaa.tool.fenced"] == [1]


@pytest.mark.parametrize("fence", [True, False])
async def test_stale_result_is_rejected_when_fencing_is_on_and_first_finisher_wins_when_off(
    cluster, fence
):
    if cluster.kind == "postgres":
        pytest.skip("Postgres leases are session-bound: see the next test")
    guard = GuardConfig(lease_ttl_s=1.0, heartbeat_s=0.2, wait_poll_s=0.1, fence_results=fence)
    sc = await _stalled_scenario(cluster, guard=guard, a_finishes_first=True)
    await until(lambda: sc.ran == ["A", "B"], 30)
    sc.gate_a.set()  # the stale attempt finishes FIRST, while still frozen
    await asyncio.sleep(0.8)
    sc.gate_b.set()
    out = await asyncio.wait_for(sc.e2.handle(sc.h.run_id).result(), 30)
    log = await sc.n2.store.read(sc.h.run_id)
    await sc.e1.abort()
    await sc.e2.__aexit__(None, None, None)
    c = types(log)
    assert c["aaa.tool.completed"] == 1
    if fence:
        assert out.value["text"] == "done: ran-on-B" and c["aaa.tool.stale_result"] == 1
        assert next(e for e in log if e.type == "aaa.tool.stale_result").data["superseded_by"] == 2
    else:
        assert out.value["text"] == "done: ran-on-A" and c["aaa.tool.stale_result"] == 0


async def test_postgres_session_lease_blocks_takeover_from_a_stalled_but_connected_worker(cluster):
    """Honest limit: Postgres advisory locks live as long as the session. A worker that is frozen
    but still connected keeps its lease, so nobody takes over (no duplicate, but also no progress)
    until it resumes or the connection dies. Use TCP keepalives / idle_session_timeout, or Redis for
    TTL-based takeover of frozen processes."""
    if cluster.kind != "postgres":
        pytest.skip("postgres-specific semantics")
    sc = await _stalled_scenario(cluster, guard=FAST, a_finishes_first=True)
    await asyncio.sleep(3.5)  # three lease lifetimes
    assert sc.ran == ["A"]  # B could not take over
    sc.gate_a.set()
    sc.backend_a.thaw()
    out = await asyncio.wait_for(sc.e2.handle(sc.h.run_id).result(), 30)
    await sc.e1.abort()
    await sc.e2.__aexit__(None, None, None)
    assert out.ok and out.value["text"] == "done: ran-on-A" and sc.ran == ["A"]


# --------------------------------------------------------------------------- single process
async def _local_boot(path, tool_fn, guard):
    eng = await Engine(store=SQLiteEventStore(path)).__aenter__()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry([tool_fn])
    ToolHandler(reg, guard=guard).attach(eng)
    algo = ToolLoop(reg)
    return eng, algo


async def test_idempotency_key_attempt_and_fence_reach_the_tool_and_survive_a_restart(tmp_path):
    seen: list[tuple] = []

    def mk(block):
        @tool(idempotency_key=lambda a: f"pay:{a['order']}")
        async def job(order: str, ctx: ToolContext) -> str:
            """pay an order"""
            seen.append((ctx.attempt, ctx.fence, ctx.idempotency_key))
            if block:
                await asyncio.Event().wait()
            return "paid"

        return job

    db = tmp_path / "idem.db"
    eng, algo = await _local_boot(db, mk(True), FAST)
    h = await eng.start(algo, {"task": "go"}, run_id="r1")
    await until(lambda: len(seen) == 1, 10)
    await asyncio.sleep(0.2)
    await eng.abort()
    await eng.store.aclose()

    eng2, algo2 = await _local_boot(db, mk(False), FAST)
    handles = await eng2.recover([algo2])
    out = await asyncio.wait_for(handles[h.run_id].result(), 20)
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.ok and seen == [
        (1, 1, "pay:o-17"),
        (2, 2, "pay:o-17"),
    ]  # same business key, rising fence


async def test_default_idempotency_key_is_stable_per_logical_call(tmp_path):
    keys: list[str] = []

    @tool
    async def job(order: str, ctx: ToolContext) -> str:
        """x"""
        keys.append(ctx.idempotency_key)
        return "ok"

    db = tmp_path / "k.db"
    eng, algo = await _local_boot(db, job, FAST)
    await eng.run(algo, {"task": "go"}, run_id="run-xyz")
    await eng.__aexit__(None, None, None)
    await eng.store.aclose()
    assert len(keys) == 1 and keys[0].startswith("run-xyz/call_")


@pytest.mark.parametrize("policy", ["at_most_once", "at_least_once"])
async def test_single_process_restart_obeys_the_delivery_policy(tmp_path, policy):
    ran: list[int] = []

    def mk(block):
        @tool(delivery=policy)
        async def job(order: str) -> str:
            """x"""
            ran.append(len(ran) + 1)
            if block:
                await asyncio.Event().wait()
            return "done-once-more"

        return job

    db = tmp_path / f"{policy}.db"
    eng, algo = await _local_boot(db, mk(True), FAST)
    h = await eng.start(algo, {"task": "go"}, run_id="r")
    await until(lambda: ran == [1], 10)
    await asyncio.sleep(0.2)
    await eng.abort()
    await eng.store.aclose()

    eng2, algo2 = await _local_boot(db, mk(False), FAST)
    handles = await eng2.recover([algo2])
    out = await asyncio.wait_for(handles[h.run_id].result(), 20)
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.ok
    if policy == "at_most_once":
        assert ran == [1] and "OutcomeUnknown" in out.value["text"]
    else:
        assert ran == [1, 2] and "done-once-more" in out.value["text"]


async def test_guard_disabled_records_no_attempts_but_still_runs(tmp_path):
    @tool
    async def job(order: str) -> str:
        """x"""
        return "plain"

    db = tmp_path / "off.db"
    eng, algo = await _local_boot(db, job, GuardConfig(enabled=False))
    h = await eng.start(algo, {"task": "go"})
    out = await h.result()
    log = await eng.store.read(h.run_id)
    await eng.__aexit__(None, None, None)
    await eng.store.aclose()
    assert out.ok and out.value["text"] == "done: plain"
    assert types(log)["aaa.tool.started"] == 0


# --------------------------------------------------------------------------- configuration
def test_config_defaults_validation_and_merging():
    c = GuardConfig()
    assert (c.enabled, c.delivery.value, c.max_attempts, c.lease_ttl_s, c.beat) == (
        True,
        "at_least_once",
        3,
        15.0,
        5.0,
    )
    assert (
        GuardConfig(lease_ttl_s=9).beat == 3.0
        and GuardConfig(lease_ttl_s=9, heartbeat_s=1).beat == 1
    )
    for bad in (
        {"max_attempts": 0},
        {"lease_ttl_s": 0},
        {"lease_ttl_s": 5, "heartbeat_s": 5},
        {"delivery": "sometimes"},
        {"on_lost_lease": "panic"},
        {"nonsense": 1},
    ):
        with pytest.raises(ValidationError):
            GuardConfig(**bad)
    with pytest.raises(ValidationError):
        c.enabled = False  # immutable once built

    @tool(delivery="at_most_once", guard={"max_attempts": 1})
    def t1(x: int) -> int:
        """x"""
        return x

    @tool
    def t2(x: int) -> int:
        """x"""
        return x

    h = ToolHandler(ToolRegistry([t1, t2]), guard=GuardConfig(max_attempts=5, lease_ttl_s=30))
    assert h.config_for(t1).delivery.value == "at_most_once" and h.config_for(t1).max_attempts == 1
    assert h.config_for(t1).lease_ttl_s == 30  # inherited from the handler default
    assert h.config_for(t2).delivery.value == "at_least_once" and h.config_for(t2).max_attempts == 5


async def test_inconsistent_tool_override_fails_at_startup_not_at_runtime():
    @tool(guard={"heartbeat_s": 20})  # fine alone (ttl default 15?) ...
    def t(x: int) -> int:
        """x"""
        return x

    async with Engine() as eng:
        with pytest.raises(ValidationError):  # ...but not against this handler default
            ToolHandler(ToolRegistry([t]), guard=GuardConfig(lease_ttl_s=10)).attach(eng)


def test_invalid_tool_guard_is_rejected_when_the_tool_is_defined():
    with pytest.raises(ValidationError):
        tool(lambda: None, delivery="whenever") if False else tool(guard={"max_attempts": -1})(
            lambda: None
        )
