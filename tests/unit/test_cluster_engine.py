import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa.algos import PlanAndExecute, ToolLoop
from aaa.core import OutcomeKind
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool
from cluster import until

CALLS: Counter = Counter()


class CountingLoop(ToolLoop):
    """Records which node handled each event, to prove work really is spread across nodes."""

    def __init__(self, *a, node: str, hits: Counter, **kw):
        super().__init__(*a, **kw)
        self._node, self._hits = node, hits

    async def on_event(self, e, ctx):
        self._hits[self._node] += 1
        return await super().on_event(e, ctx)


def registry():
    @tool
    def work(n: int) -> int:
        """do work"""
        CALLS[n] += 1
        return n * 2

    @tool
    def noop() -> str:
        """noop"""
        return "ok"

    return ToolRegistry([work, noop])


def script(req):
    last = req.messages[-1]
    if last.role == "user":
        n = int("".join(c for c in last.content if c.isdigit()) or 0)
        return tool_call("work", n=n)
    return text(f"result {last.content}")


class Gated:
    """LLM client whose calls never return (a node that dies mid-request)."""

    def __init__(self):
        self.calls = 0

    async def complete(self, req):
        self.calls += 1
        await asyncio.Event().wait()

    async def embed(self, *a, **k): ...
    async def aclose(self): ...


async def node_engine(
    node, member, *, llm=None, algo=None, reg=None, llm_handler=True, tools=True, **eng_kw
):
    eng = Engine(store=node.store, bus=node.bus(member=member), **eng_kw)
    await eng.__aenter__()
    reg = reg or registry()
    if llm_handler:
        LLMHandler(llm or FakeLLMClient(script)).attach(eng)
    if tools:
        ToolHandler(reg).attach(eng)
    return eng, reg


async def all_owned(buses, group, partitions):
    """Wait until every node owns a non-empty, disjoint share and together they cover everything."""

    def ok():
        owned = [set(b.owned_partitions.get(group, [])) for b in buses]
        return (
            all(owned)
            and sum(map(len, owned)) == partitions
            and len(set().union(*owned)) == partitions
        )

    await until(ok, 20)


async def test_two_engines_split_runs_and_each_effect_happens_once(cluster):
    CALLS.clear()
    n1, n2 = await cluster.node(), await cluster.node()
    hits: Counter = Counter()
    e1, r1 = await node_engine(n1, "A")
    e2, r2 = await node_engine(n2, "B")
    a1, a2 = CountingLoop(r1, node="A", hits=hits), CountingLoop(r2, node="B", hits=hits)
    e1.register(a1)
    e2.register(a2)
    await all_owned([e1.bus, e2.bus], "aaa.algo.tool_loop", cluster.partitions)

    handles = [
        await e1.start(a1, {"task": f"compute {i}"}, run_id=f"run-{i}") for i in range(1, 25)
    ]
    outs = await asyncio.wait_for(asyncio.gather(*(h.result() for h in handles)), 30)
    assert all(o.ok for o in outs)
    assert [o.value["text"] for o in outs][:3] == ["result 2", "result 4", "result 6"]
    assert set(hits) == {"A", "B"}  # both nodes hosted algorithm work
    assert sorted(CALLS) == list(range(1, 25)) and set(CALLS.values()) == {1}  # each tool ran once
    # any node can wait on any run, even one it never hosted
    assert (await e2.handle("run-7").result()).value["text"] == "result 14"
    log = await n1.store.read("run-3")
    assert Counter(e.type for e in log)["aaa.run.completed"] == 1
    await e1.__aexit__(None, None, None)
    await e2.__aexit__(None, None, None)


async def test_node_crash_mid_flight_other_node_finishes_every_run(cluster):
    CALLS.clear()
    n1, n2 = await cluster.node(), await cluster.node()
    gated = Gated()
    e1, r1 = await node_engine(n1, "A", llm=gated)  # A's LLM calls hang forever
    a1 = ToolLoop(r1)
    handles = [await e1.start(a1, {"task": f"compute {i}"}, run_id=f"c-{i}") for i in range(1, 11)]
    await until(lambda: gated.calls >= 8, 10)
    await e1.abort()  # A dies with requests in flight

    e2, r2 = await node_engine(n2, "B")  # B takes over after A's leases lapse
    e2.register(ToolLoop(r2))
    outs = await asyncio.wait_for(
        asyncio.gather(*(e2.handle(h.run_id).result() for h in handles)), 40
    )
    assert all(o.ok for o in outs) and outs[0].value["text"] == "result 2"
    assert set(CALLS) == set(range(1, 11)) and all(v == 1 for v in CALLS.values())
    for h in handles:
        log = await n2.store.read(h.run_id)
        c = Counter(e.type for e in log)
        assert c["aaa.run.completed"] == 1 and c["aaa.llm.completed"] == 2  # no duplicated results
        assert len({e.id for e in log}) == len(log)
    await e2.__aexit__(None, None, None)


async def test_cancel_and_budget_work_across_nodes(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    spin = FakeLLMClient(lambda r: tool_call("noop"))
    # node A: algorithm + tools. node B: ONLY the LLM handler — every llm.completed is emitted by B
    e1, r1 = await node_engine(n1, "A", llm_handler=False)
    e2, _ = await node_engine(n2, "B", llm=spin, tools=False)
    algo = ToolLoop(r1, max_turns=999)
    # budget: spend is accrued from events another process wrote, then enforced here
    out = await asyncio.wait_for(e1.run(algo, {"task": "spin"}, budget=Budget(max_steps=12)), 30)
    assert out.kind is OutcomeKind.BUDGET_EXCEEDED

    h = await e1.start(algo, {"task": "spin forever"}, run_id="to-cancel")
    await asyncio.sleep(0.3)
    await e2.handle("to-cancel").cancel("from the other node")  # cancel issued on a different node
    o = await asyncio.wait_for(h.result(), 20)
    assert o.kind is OutcomeKind.CANCELLED and o.reason == "from the other node"
    await e1.__aexit__(None, None, None)
    await e2.__aexit__(None, None, None)


async def test_child_runs_and_parent_announcements_span_nodes(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    plan = {
        "steps": [
            {"id": "a", "description": "A", "depends_on": []},
            {"id": "b", "description": "B", "depends_on": []},
            {"id": "c", "description": "C", "depends_on": ["a", "b"]},
        ]
    }

    def pscript(req):
        sysm = req.messages[0].content if req.messages[0].role == "system" else ""
        if "planner" in sysm:
            return text(json.dumps(plan))
        if "Combine" in sysm:
            return text("ALL DONE")
        return text("step ok")

    e1, r1 = await node_engine(n1, "A", llm=FakeLLMClient(pscript))
    e2, r2 = await node_engine(n2, "B", llm=FakeLLMClient(pscript))
    pe1, pe2 = PlanAndExecute(ToolLoop(r1)), PlanAndExecute(ToolLoop(r2))
    for eng, pe in ((e1, pe1), (e2, pe2)):
        eng.register(pe)
        eng.register(pe.executor)
    await all_owned([e1.bus, e2.bus], "aaa.algo.plan_and_execute", cluster.partitions)
    outs = await asyncio.wait_for(
        asyncio.gather(*(e1.run(pe1, {"task": f"T{i}"}) for i in range(6))), 60
    )
    assert all(
        o.ok and o.value["text"] == "ALL DONE" and set(o.value["results"]) == {"a", "b", "c"}
        for o in outs
    )
    await e1.__aexit__(None, None, None)
    await e2.__aexit__(None, None, None)


class Idle:
    name = "idle"

    def subscriptions(self):
        return ["aaa.run.started"]

    async def on_event(self, e, ctx):
        return []


async def test_sweeper_enforces_deadline_of_a_dead_owner(cluster):
    n1, n2 = await cluster.node(), await cluster.node()
    e1, _ = await node_engine(n1, "A", llm_handler=False, tools=False)
    h = await e1.start(Idle(), {}, budget=Budget(deadline_s=0.5), run_id="doomed")
    await asyncio.sleep(0.1)
    await e1.abort()  # owner (and its timer) dies
    e2, _ = await node_engine(n2, "B", llm_handler=False, tools=False, sweep_interval_s=0.2)
    out = await asyncio.wait_for(e2.handle(h.run_id).result(), 20)
    assert out.kind is OutcomeKind.TIMEOUT
    await e2.__aexit__(None, None, None)
