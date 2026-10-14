import asyncio
import json
import sqlite3

from aaa.algos import PlanAndExecute, ToolLoop
from aaa.core import OutcomeKind
from aaa.events import SQLiteEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


class Gated:
    """LLM client that blocks (forever, until released) when `block(request)` is true."""

    def __init__(self, inner, block=lambda r: False):
        self.inner, self.block, self.gate = inner, block, asyncio.Event()
        self.in_flight = asyncio.Event()

    async def complete(self, req):
        if self.block(req):
            self.in_flight.set()
            await self.gate.wait()
        return await self.inner.complete(req)

    async def embed(self, *a, **k):
        return await self.inner.embed(*a, **k)

    async def aclose(self):
        return None


async def boot(path, llm, registry=None):
    eng = await Engine(store=SQLiteEventStore(path)).__aenter__()
    LLMHandler(llm).attach(eng)
    if registry is not None:
        ToolHandler(registry).attach(eng)
    return eng


async def crash(eng):
    await eng.abort()
    await eng.store.aclose()


async def wait_for(eng, run_id, pred, timeout=3):
    async def poll():
        while not any(pred(e) for e in await eng.store.read(run_id)):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def test_crash_while_llm_call_in_flight_then_resume(tmp_path):
    db = tmp_path / "a.db"
    llm1 = Gated(FakeLLMClient([]), block=lambda r: True)
    eng = await boot(db, llm1)
    h = await eng.start(ToolLoop(ToolRegistry()), {"task": "hi"})
    await asyncio.wait_for(llm1.in_flight.wait(), 3)
    run_id = h.run_id
    await crash(eng)  # request logged, no response

    eng2 = await boot(db, FakeLLMClient([text("hello after crash")]))
    handles = await eng2.recover([ToolLoop(ToolRegistry())])
    out = await asyncio.wait_for(handles[run_id].result(), 3)
    log = await eng2.store.read(run_id)
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.ok and out.value["text"] == "hello after crash"
    types = [e.type for e in log]
    assert types.count("aaa.run.started") == 1 and types.count("aaa.llm.requested") == 1
    assert types.count("aaa.llm.completed") == 1 and types.count("aaa.run.completed") == 1


async def test_crash_mid_parallel_tools_does_not_rerun_finished_tool(tmp_path):
    db = tmp_path / "b.db"
    calls = {"fast": 0, "slow": 0}
    gate = asyncio.Event()

    def make(block):
        @tool
        def fast() -> str:
            """fast"""
            calls["fast"] += 1
            return "F"

        @tool
        async def slow() -> str:
            """slow"""
            calls["slow"] += 1
            if block:
                await gate.wait()
            return "S"

        return ToolRegistry([fast, slow])

    def script(req):
        if req.messages[-1].role == "user":
            from aaa.llm import LLMResponse, Message, ToolCall

            return LLMResponse(
                message=Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="1", name="fast"), ToolCall(id="2", name="slow")],
                )
            )
        return text("|".join(m.content for m in req.messages if m.role == "tool"))

    eng = await boot(db, FakeLLMClient(script), make(block=True))
    h = await eng.start(ToolLoop(make(True)), {"task": "go"})
    await wait_for(eng, h.run_id, lambda e: e.type == "aaa.tool.completed")
    await crash(eng)
    assert calls == {"fast": 1, "slow": 1}

    reg2 = make(block=False)
    eng2 = await boot(db, FakeLLMClient(script), reg2)
    handles = await eng2.recover([ToolLoop(reg2)])
    out = await asyncio.wait_for(handles[h.run_id].result(), 3)
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.value["text"] == "F|S" and calls == {"fast": 1, "slow": 2}  # fast NOT re-run


async def test_unreacted_fact_is_redelivered(tmp_path):
    """Crash after tool.completed was logged but before the algorithm reacted to it."""
    db = tmp_path / "c.db"

    @tool
    def add(a: int, b: int) -> int:
        """add"""
        return a + b

    def script(req):
        return (
            tool_call("add", a=1, b=2)
            if req.messages[-1].role == "user"
            else text(f"got {req.messages[-1].content}")
        )

    class StuckOnToolResult(ToolLoop):
        async def on_event(self, e, ctx):
            if e.type == "aaa.tool.completed":
                await asyncio.Event().wait()  # "crash" before reacting
            return await super().on_event(e, ctx)

    reg = ToolRegistry([add])
    eng = await boot(db, FakeLLMClient(script), reg)
    h = await eng.start(StuckOnToolResult(reg), {"task": "x"})
    await wait_for(eng, h.run_id, lambda e: e.type == "aaa.tool.completed")
    await crash(eng)

    eng2 = await boot(db, FakeLLMClient(script), reg)
    handles = await eng2.recover([ToolLoop(reg)])
    out = await asyncio.wait_for(handles[h.run_id].result(), 3)
    assert out.value["text"] == "got 3" and eng2.last_recovery.redelivered >= 1
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()


async def test_crash_with_child_run_in_flight_resumes_parent_and_child_exactly_once(tmp_path):
    db = tmp_path / "d.db"
    reg = ToolRegistry()
    plan = {
        "steps": [
            {"id": "a", "description": "step A", "depends_on": []},
            {"id": "b", "description": "step B", "depends_on": ["a"]},
        ]
    }

    def script(req):
        sysm = req.messages[0].content if req.messages[0].role == "system" else ""
        if "planner" in sysm:
            return text(json.dumps(plan))
        if "Combine" in sysm:
            return text("ALL DONE")
        return text("result of " + req.messages[-1].content.split("Your step (")[1][0])

    llm1 = Gated(
        FakeLLMClient(script), block=lambda r: "Your step (a)" in (r.messages[-1].content or "")
    )
    eng = await boot(db, llm1, reg)
    h = await eng.start(PlanAndExecute(ToolLoop(reg)), {"task": "T"}, budget=Budget(max_steps=50))
    await asyncio.wait_for(llm1.in_flight.wait(), 3)  # child for step a is mid-LLM-call
    await crash(eng)

    eng2 = await boot(db, FakeLLMClient(script), reg)
    handles = await eng2.recover([PlanAndExecute(ToolLoop(reg)), ToolLoop(reg)])
    assert len(handles) == 2  # parent + child, nothing duplicated
    out = await asyncio.wait_for(handles[h.run_id].result(), 5)
    log = await eng2.store.read(h.run_id)
    index = await eng2.store.run_index()
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.ok and out.value["text"] == "ALL DONE" and set(out.value["results"]) == {"a", "b"}
    assert [e.data["step_id"] for e in log if e.type == "aaa.task.started"] == ["a", "b"]
    assert len(index) == 3 and all(r.completed for r in index)  # parent + exactly 2 children


async def test_lost_child_announcement_is_reissued(tmp_path):
    db = tmp_path / "e.db"
    reg = ToolRegistry()
    plan = {"steps": [{"id": "a", "description": "step A", "depends_on": []}]}

    def script(req):
        sysm = req.messages[0].content if req.messages[0].role == "system" else ""
        if "planner" in sysm:
            return text(json.dumps(plan))
        return text("final" if "Combine" in sysm else "child done")

    class StuckOnChild(PlanAndExecute):
        async def on_event(self, e, ctx):
            if e.type == "aaa.child.completed":
                await asyncio.Event().wait()
            return await super().on_event(e, ctx)

    eng = await boot(db, FakeLLMClient(script), reg)
    h = await eng.start(StuckOnChild(ToolLoop(reg)), {"task": "T"})
    await wait_for(eng, h.run_id, lambda e: e.type == "aaa.child.completed")
    await crash(eng)
    con = sqlite3.connect(db)  # simulate the announcement never landing
    con.execute("DELETE FROM events WHERE type='aaa.child.completed'")
    con.commit()
    con.close()

    eng2 = await boot(db, FakeLLMClient(script), reg)
    handles = await eng2.recover([PlanAndExecute(ToolLoop(reg)), ToolLoop(reg)])
    out = await asyncio.wait_for(handles[h.run_id].result(), 5)
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.ok and out.value["text"] == "final"


class Idle:
    name = "idle"

    def subscriptions(self):
        return ["aaa.run.started"]

    async def on_event(self, e, ctx):
        return []


async def test_deadline_continues_across_crash_and_ledger_is_rebuilt(tmp_path):
    db = tmp_path / "f.db"
    eng = await boot(db, FakeLLMClient([]))
    h = await eng.start(Idle(), {}, budget=Budget(deadline_s=0.4, max_steps=7))
    await asyncio.sleep(0.05)
    await crash(eng)
    await asyncio.sleep(0.45)  # outage longer than the deadline
    eng2 = await boot(db, FakeLLMClient([]))
    handles = await eng2.recover([Idle()])
    out = await asyncio.wait_for(handles[h.run_id].result(), 2)
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.kind is OutcomeKind.TIMEOUT


async def test_budget_spend_survives_restart(tmp_path):
    db = tmp_path / "g.db"
    llm1 = Gated(FakeLLMClient(lambda r: tool_call("noop")), block=lambda r: len(r.messages) >= 6)

    @tool
    def noop() -> str:
        """noop"""
        return "ok"

    reg = ToolRegistry([noop])
    eng = await boot(db, llm1, reg)
    h = await eng.start(ToolLoop(reg, max_turns=99), {"task": "spin"}, budget=Budget(max_steps=9))
    await asyncio.wait_for(llm1.in_flight.wait(), 3)
    spent_before = eng._runs[h.run_id].ledger.steps
    await crash(eng)

    eng2 = await boot(db, FakeLLMClient(lambda r: tool_call("noop")), reg)
    handles = await eng2.recover([ToolLoop(reg, max_turns=99)])
    assert eng2._runs[h.run_id].ledger.steps >= spent_before > 0
    out = await asyncio.wait_for(handles[h.run_id].result(), 5)
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()
    assert out.kind is OutcomeKind.BUDGET_EXCEEDED  # limit applied to TOTAL spend, not per-life


async def test_unknown_algorithm_and_closed_runs_are_left_alone(tmp_path):
    db = tmp_path / "h.db"
    eng = await boot(db, FakeLLMClient([text("done")]))
    done = await eng.run(ToolLoop(ToolRegistry()), {"task": "t"})
    assert done.ok
    open_h = await eng.start(Idle(), {})
    await crash(eng)

    eng2 = await boot(db, FakeLLMClient([]))
    handles = await eng2.recover([ToolLoop(ToolRegistry())])  # nobody supplied Idle
    assert handles == {} and eng2.last_recovery.unknown_algo == [open_h.run_id]
    await eng2.__aexit__(None, None, None)
    await eng2.store.aclose()


async def test_restarting_a_finished_run_id_returns_its_result(tmp_path):
    db = tmp_path / "i.db"
    eng = await boot(db, FakeLLMClient([text("once")]))
    algo = ToolLoop(ToolRegistry())
    first = await eng.run(algo, {"task": "t"}, run_id="fixed-id")
    again = await (await eng.start(algo, {"task": "t"}, run_id="fixed-id")).result()
    n = len([e for e in await eng.store.read("fixed-id") if e.type == "aaa.run.started"])
    await eng.__aexit__(None, None, None)
    await eng.store.aclose()
    assert first.ok and again.value == first.value and n == 1


async def test_sigkill_of_a_real_process_then_recover(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path

    db = tmp_path / "kill.db"
    victim = Path(__file__).parent.parent / "helpers" / "victim.py"
    proc = subprocess.Popen(  # noqa: ASYNC220 - short-lived test helper process
        [sys.executable, "-I", str(victim), str(db)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
    )
    try:
        lines = []
        while "READY" not in (line := await asyncio.to_thread(proc.stdout.readline)):
            lines.append(line)
            assert line, "victim exited early"
        proc.send_signal(signal.SIGKILL)  # no cleanup, no atexit, no flush: a real crash
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()

    @tool
    def add(a: int, b: int) -> int:
        """add"""
        return a + b

    reg = ToolRegistry([add])
    eng = await boot(
        db, FakeLLMClient(lambda r: text(f"the answer is {r.messages[-1].content}")), reg
    )
    handles = await eng.recover([ToolLoop(reg)])
    out = await asyncio.wait_for(handles["the-run"].result(), 5)
    log = await eng.store.read("the-run")
    await eng.__aexit__(None, None, None)
    await eng.store.aclose()
    assert out.ok and out.value["text"] == "the answer is 42"
    types = [e.type for e in log]
    assert types.count("aaa.tool.completed") == 1 and types.count("aaa.run.completed") == 1
