import json

import pytest

from aaa.algos import FSM, Node, PlanAndExecute, SelfConsistency, ToolLoop, WorkflowGraph
from aaa.algos.control import FSM_STATE, WF
from aaa.algos.ensemble import SAMPLES
from aaa.algos.loops import LOOP
from aaa.algos.plan import PLAN
from aaa.algos.refine import REFINE
from aaa.events import SQLiteEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool

ALL = [LOOP, PLAN, WF, FSM_STATE, REFINE, SAMPLES]


@pytest.fixture
def aggressive(monkeypatch):
    for p in ALL:
        monkeypatch.setattr(p, "every", 3)


@tool
def noop() -> str:
    """noop"""
    return "ok"


def counting(proj):
    n = {"folds": 0}
    orig = proj.fold

    def fold(st, e):
        n["folds"] += 1
        return orig(st, e)

    proj.fold = fold
    return n, lambda: setattr(proj, "fold", orig)


async def run_long(store, turns, snapshots):
    reg = ToolRegistry([noop])
    seen = {"n": 0}

    def script(req):
        seen["n"] += 1
        return tool_call("noop") if seen["n"] <= turns else text("done")

    async with Engine(store=store) as eng:
        LLMHandler(FakeLLMClient(script)).attach(eng)
        ToolHandler(reg).attach(eng)
        out = await eng.run(ToolLoop(reg, max_turns=turns + 5), {"task": "go"})
    assert out.ok
    return out


async def test_snapshots_make_long_runs_linear_not_quadratic(tmp_path, monkeypatch):
    results = {}
    for use_snap in (False, True):
        monkeypatch.setattr(LOOP, "every", 8 if use_snap else 10**9)
        n, restore = counting(LOOP)
        store = SQLiteEventStore(tmp_path / f"{use_snap}.db")
        await run_long(store, 40, use_snap)
        await store.aclose()
        restore()
        results[use_snap] = n["folds"]
    # 40 turns ~ 200 events. Without snapshots every handled event refolds the whole log (O(n^2)).
    assert results[False] > 3000
    assert results[True] < results[False] / 8, results


async def test_snapshot_state_equals_full_replay_for_every_projection(tmp_path, aggressive):
    store = SQLiteEventStore(tmp_path / "eq.db")
    reg = ToolRegistry([noop])
    plan = {
        "steps": [
            {"id": "a", "description": "A", "depends_on": []},
            {"id": "b", "description": "B", "depends_on": ["a"]},
        ]
    }

    def script(req):
        sysm = req.messages[0].content if req.messages[0].role == "system" else ""
        if "planner" in sysm:
            return text(json.dumps(plan))
        if "Combine" in sysm:
            return text("done")
        if "step" in (req.messages[-1].content or "") and req.messages[-1].role == "user":
            return tool_call("noop")
        return text("Answer: 7")

    async with Engine(store=store) as eng:
        LLMHandler(FakeLLMClient(script)).attach(eng)
        ToolHandler(reg).attach(eng)
        await eng.run(PlanAndExecute(ToolLoop(reg)), {"task": "T"})
        await eng.run(SelfConsistency(n=7), {"task": "q"})
        wf = WorkflowGraph([Node("a", "noop"), Node("b", "noop", after=["a"])])
        await eng.run(wf, {})
        fsm = FSM("s", {"s": {"user.go": "t"}}, {"t"})
        h = await eng.start(fsm, {})
        await h.signal("user.go")
        await h.result()
        streams = [r.stream for r in await store.run_index()]
        assert len(streams) >= 6
        checked = 0
        for stream in streams:
            for proj in ALL:
                fast = await proj.load(store, stream, hydrate=eng.hydrate)  # snapshot + tail
                full = await proj.load(store, stream, hydrate=eng.hydrate, use_snapshots=False)
                assert fast.state.model_dump() == full.state.model_dump(), (stream, proj.name)
                assert fast.seq == full.seq
                checked += fast.from_snapshot >= 0
    await store.aclose()
    assert checked >= 5  # snapshots were genuinely exercised, not trivially skipped


async def test_corrupt_stale_or_future_snapshots_are_ignored(tmp_path):
    from aaa.events import Event, StoredSnapshot

    store = SQLiteEventStore(tmp_path / "bad.db")
    await store.append(
        "r", [Event(type="aaa.run.started", run_id="r", data={"input": {"x": 1}}) for _ in range(1)]
    )
    LOOP.every = 1
    try:
        for bad in (
            StoredSnapshot(
                stream="r", name=LOOP.name, seq=0, version=1, state="{not json"
            ),  # corrupt
            StoredSnapshot(
                stream="r", name=LOOP.name, seq=99, version=1, state="{}"
            ),  # ahead of the log
            StoredSnapshot(
                stream="r", name=LOOP.name, seq=0, version=7, state="{}"
            ),  # wrong version
        ):
            await store._run(lambda: store._conn.execute("DELETE FROM snapshots"))
            await store.save_snapshot(bad)
            loaded = await LOOP.load(store, "r")
            assert loaded.from_snapshot == -1 and loaded.state.input == {"x": 1}
    finally:
        LOOP.every = 64
        await store.aclose()


async def test_snapshot_save_is_monotonic(tmp_path):
    from aaa.events import StoredSnapshot

    store = SQLiteEventStore(tmp_path / "m.db")
    mk = lambda seq: StoredSnapshot(stream="r", name="n", seq=seq, version=1, state=str(seq))
    await store.save_snapshot(mk(10))
    await store.save_snapshot(mk(5))  # an older writer must not roll the snapshot back
    assert (await store.load_snapshot("r", "n")).seq == 10
    await store.save_snapshot(mk(11))
    assert (await store.load_snapshot("r", "n")).seq == 11
    await store.aclose()


async def test_recovery_and_budget_work_with_snapshots(tmp_path, aggressive):
    store = SQLiteEventStore(tmp_path / "r.db")
    reg = ToolRegistry([noop])
    async with Engine(store=store) as eng:
        LLMHandler(FakeLLMClient(lambda r: tool_call("noop"))).attach(eng)
        ToolHandler(reg).attach(eng)
        out = await eng.run(
            ToolLoop(reg, max_turns=99), {"task": "spin"}, budget=Budget(max_steps=30)
        )
    await store.aclose()
    assert out.kind.value == "budget_exceeded"
