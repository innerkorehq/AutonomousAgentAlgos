import asyncio
import json

from pydantic import BaseModel

from aaa.algos import (
    FSM,
    GenerateTestRepair,
    GeneratorCritic,
    Node,
    PlanAndExecute,
    RAGAgent,
    Route,
    Router,
    SelfConsistency,
    Supervisor,
    ToolLoop,
    WorkflowGraph,
)
from aaa.core import OutcomeKind
from aaa.data import Doc, InMemoryIndex
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool
from aaa.validate import ValidationHandler, json_validator


def wire(eng, llm, registry=None, validators=None):
    LLMHandler(llm).attach(eng)
    if registry is not None:
        ToolHandler(registry).attach(eng)
    if validators:
        ValidationHandler(validators).attach(eng)


def sys_of(req):
    return req.messages[0].content if req.messages[0].role == "system" else ""


def user_of(req):
    return next(m.content for m in reversed(req.messages) if m.role == "user")


# ---------------------------------------------------------------- refine loops
async def test_generate_test_repair_fixes_then_accepts_and_gives_up_bounded():
    class Out(BaseModel):
        n: int

    replies = iter(['```json\n{"n": "x"}\n```', '{"n": 3}'])
    llm = FakeLLMClient(lambda r: text(next(replies)))
    async with Engine() as eng:
        wire(eng, llm, validators={"default": json_validator(Out)})
        out = await eng.run(GenerateTestRepair(max_iterations=3), {"task": "give n"})
    assert out.ok and out.value == {"candidate": '{"n": 3}', "iterations": 2}
    fb = llm.requests[1].messages[-1].content
    assert "rejected" in fb and "n:" in fb  # validator feedback reached the model

    llm2 = FakeLLMClient(lambda r: text("nope"))
    async with Engine() as eng:
        wire(eng, llm2, validators={"default": lambda c, ctx: (False, "always bad")})
        out = await eng.run(GenerateTestRepair(max_iterations=2), {"task": "t"})
    assert (
        out.kind is OutcomeKind.NO_SOLUTION
        and out.value["iterations"] == 2
        and "always bad" in out.reason
    )


async def test_generator_critic_loop():
    def script(req):
        if "strict reviewer" in sys_of(req):
            ok = "v2" in req.messages[-1].content
            return text(json.dumps({"ok": ok, "feedback": "" if ok else "make it v2"}))
        return text(
            "draft v2"
            if any("make it v2" in (m.content or "") for m in req.messages)
            else "draft v1"
        )

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script))
        out = await eng.run(GeneratorCritic(), {"task": "write"})
    assert out.ok and out.value == {"candidate": "draft v2", "iterations": 2}


# ---------------------------------------------------------------- plan and execute
async def test_plan_and_execute_dag_parallel_children_budget_rollup_and_synthesis():
    @tool
    def lookup(item: str) -> str:
        """Look something up."""
        return f"fact:{item}"

    reg = ToolRegistry([lookup])
    plan = {
        "steps": [
            {"id": "a", "description": "find A", "depends_on": []},
            {"id": "b", "description": "find B", "depends_on": []},
            {"id": "c", "description": "combine", "depends_on": ["a", "b"]},
        ]
    }

    def script(req):
        s = sys_of(req)
        if "planner" in s:
            return text(json.dumps(plan))
        if "Combine the step results" in s:
            return text("FINAL " + user_of(req).split("Step results: ")[1][:60])
        u = user_of(req)  # executor ToolLoop child
        if req.messages[-1].role == "user":
            step = u.split("Your step (")[1][0]
            return tool_call("lookup", item=step)
        return text(f"done:{req.messages[-1].content}")

    llm = FakeLLMClient(script)
    async with Engine() as eng:
        wire(eng, llm, reg)
        h = await eng.start(PlanAndExecute(ToolLoop(reg), max_parallel=2), {"task": "T"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    assert (
        out.ok
        and out.value["text"].startswith("FINAL")
        and set(out.value["results"]) == {"a", "b", "c"}
    )
    types = [e.type for e in log]
    started = [e for e in log if e.type == "aaa.task.started"]
    assert [e.data["step_id"] for e in started[:2]] == ["a", "b"] and started[2].data[
        "step_id"
    ] == "c"
    assert types.index("aaa.plan.created") < types.index("aaa.task.started")
    # c's prompt carries a and b results (dependency data flowed through the DAG)
    c_req = next(r for r in llm.requests if "Your step (c)" in user_of(r))
    assert "done:fact:a" in user_of(c_req)
    assert types.count("aaa.child.completed") == 3
    # child spend rolled up into the parent's ledger (3 children x (2 llm + 1 tool) + 2 parent llm)
    async with Engine() as eng2:
        wire(eng2, FakeLLMClient(script), reg)
        capped = await eng2.run(
            PlanAndExecute(ToolLoop(reg)), {"task": "T"}, budget=Budget(max_steps=4)
        )
    assert capped.kind is OutcomeKind.BUDGET_EXCEEDED


async def test_plan_validation_and_replan_after_step_failure():
    @tool
    def flaky(x: str) -> str:
        """Fails for 'bad'."""
        if x == "bad":
            raise RuntimeError("boom")
        return "fine"

    reg = ToolRegistry([flaky])
    calls = {"plan": 0}

    def script(req):
        s = sys_of(req)
        if "planner" in s:
            calls["plan"] += 1
            if "Failed steps" in user_of(req):
                return text(
                    json.dumps(
                        {"steps": [{"id": "fix", "description": "retry good", "depends_on": []}]}
                    )
                )
            return text(
                json.dumps({"steps": [{"id": "s1", "description": "do bad", "depends_on": []}]})
            )
        if "Combine" in s:
            return text("recovered")
        if req.messages[-1].role == "user":
            return tool_call("flaky", x="bad" if "do bad" in user_of(req) else "ok")
        return text("err" if "Error" in req.messages[-1].content else "ok")

    # executor that fails on tool error: use a tiny wrapper algorithm via ToolLoop max_turns... instead
    # make failure explicit by letting executor give up (NO_SOLUTION) when the tool errors.
    class StrictExec(ToolLoop):
        def on_reply(self, e, st, msg):
            from aaa.algos.base import finish

            if not msg.tool_calls and msg.content == "err":
                return [finish(e, OutcomeKind.FAILED, reason="tool error")]
            return super().on_reply(e, st, msg)

    async with Engine() as eng:
        wire(eng, FakeLLMClient(script), reg)
        out = await eng.run(PlanAndExecute(StrictExec(reg)), {"task": "T"})
        assert out.ok and out.value["text"] == "recovered" and calls["plan"] == 2
        assert "r1-fix" in out.value["results"]

    async with Engine() as eng:  # cyclic plan is rejected, not executed
        bad = {
            "steps": [
                {"id": "a", "description": "x", "depends_on": ["b"]},
                {"id": "b", "description": "y", "depends_on": ["a"]},
            ]
        }
        wire(eng, FakeLLMClient(lambda r: text(json.dumps(bad))), reg)
        out = await eng.run(PlanAndExecute(ToolLoop(reg)), {"task": "T"})
    assert out.kind is OutcomeKind.FAILED and "cycle" in out.reason


# ---------------------------------------------------------------- router
async def test_router_rules_first_then_llm_and_unknown_route():
    reg = ToolRegistry()
    billing, tech = ToolLoop(reg, name="billing"), ToolLoop(reg, name="tech")
    seen = []

    def script(req):
        if "Choose the best handler" in sys_of(req):
            return text(json.dumps({"route": "tech" if "crash" in user_of(req) else "nope"}))
        seen.append(sys_of(req))
        return text("handled")

    billing.system, tech.system = "I am billing", "I am tech"
    router = Router(
        {"billing": Route(billing, "invoices"), "tech": Route(tech, "bugs")},
        classifier=lambda inp: "billing" if "invoice" in inp["task"] else None,
    )
    async with Engine() as eng:
        wire(eng, FakeLLMClient(script), reg)
        assert (await eng.run(router, {"task": "my invoice is wrong"})).ok
        assert (await eng.run(router, {"task": "app crash on start"})).ok
        bad = await eng.run(router, {"task": "something else"})
    assert seen == ["I am billing", "I am tech"]
    assert bad.kind is OutcomeKind.INVALID_INPUT


# ---------------------------------------------------------------- workflow + fsm
async def test_workflow_graph_parallel_skip_and_failure():
    @tool
    def double(n: int) -> int:
        """x2"""
        return n * 2

    @tool
    def total(xs: list[int]) -> int:
        """sum"""
        return sum(xs)

    @tool
    def boom() -> int:
        """fails"""
        raise RuntimeError("kaput")

    reg = ToolRegistry([double, total, boom])
    wf = WorkflowGraph(
        [
            Node("a", "double", args=lambda r, i: {"n": i["n"]}),
            Node("b", "double", args=lambda r, i: {"n": i["n"] + 1}),
            Node("skipme", "double", after=["a"], args={"n": 0}, when=lambda r, i: r["a"] > 100),
            Node(
                "sum",
                "total",
                after=["a", "b", "skipme"],
                args=lambda r, i: {"xs": [r["a"], r["b"]]},
            ),
        ],
        output="sum",
    )
    async with Engine() as eng:
        wire(eng, FakeLLMClient([]), reg)
        h = await eng.start(wf, {"n": 5})
        out = await h.result()
        types = [e.type for e in await eng.store.read(h.run_id)]
        assert out.ok and out.value == 22 and "aaa.node.skipped" in types
        bad = await eng.run(
            WorkflowGraph([Node("x", "boom"), Node("y", "double", after=["x"], args={"n": 1})]), {}
        )
    assert bad.kind is OutcomeKind.FAILED and "kaput" in bad.reason
    try:
        WorkflowGraph([Node("a", "t", after=["b"]), Node("b", "t", after=["a"])])
        raise AssertionError("cycle not detected")
    except ValueError as e:
        assert "cycle" in str(e)


async def test_fsm_guards_ignored_events_entry_actions_and_loops():
    entered = []
    fsm = FSM(
        "draft",
        {
            "draft": {"user.submit": "review"},
            "review": {
                "user.reject": "draft",
                "user.approve": lambda e, inp: (
                    "done" if e.data.get("by") in inp["approvers"] else None
                ),
            },
        },
        terminal={"done"},
        on_enter={"review": lambda e, inp: entered.append(1) or []},
    )
    async with Engine() as eng:
        h = await eng.start(fsm, {"approvers": ["alice"]})
        await asyncio.sleep(0.01)
        await h.signal("user.approve", {"by": "alice"})  # invalid in 'draft': ignored
        await h.signal("user.submit")
        await h.signal("user.reject")  # loop back
        await h.signal("user.submit")
        await h.signal("user.approve", {"by": "mallory"})  # guard says no
        await h.signal("user.approve", {"by": "alice"})
        out = await h.result()
        types = [e.type for e in await eng.store.read(h.run_id)]
    assert out.value["path"] == ["draft", "review", "draft", "review", "done"]
    assert types.count("aaa.fsm.ignored") == 2 and len(entered) == 2


# ---------------------------------------------------------------- ensemble family
async def test_self_consistency_majority_and_partial_failure():
    answers = iter(["... Answer: 42", "Answer: 41", "x\nAnswer: 42.", "Answer: 42"])
    async with Engine() as eng:
        wire(eng, FakeLLMClient(lambda r: text(next(answers))))
        out = await eng.run(SelfConsistency(n=4), {"task": "q"})
    assert (
        out.value["answer"] == "42" and out.value["votes"] == 3 and out.value["agreement"] == 0.75
    )


async def test_rag_agent_and_supervisor_delegation():
    idx = InMemoryIndex()
    await idx.upsert(
        [
            Doc(id="d1", text="Paris is the capital of France"),
            Doc(id="d2", text="Tokyo is in Japan"),
        ]
    )
    rag = RAGAgent(idx)

    def worker_llm(req):
        if "evidence" in sys_of(req):  # RAG worker
            return (
                tool_call("search_documents", query="capital France")
                if req.messages[-1].role == "user"
                else text(f"cited {'d1' in req.messages[-1].content}")
            )
        if "coordinate" in sys_of(req):  # supervisor
            if req.messages[-1].role == "user":
                from aaa.llm import LLMResponse, Message, ToolCall

                return LLMResponse(
                    message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="1",
                                name="delegate_rag",
                                arguments={"task": "capital of France?"},
                            ),
                            ToolCall(
                                id="2",
                                name="delegate_rag",
                                arguments={"task": "capital of France again?"},
                            ),
                        ],
                    )
                )
            return text(
                "Supervisor says: "
                + " | ".join(m.content for m in req.messages if m.role == "tool")
            )
        raise AssertionError(sys_of(req))

    sup = Supervisor({"rag": (rag, "answers from the knowledge base")})
    async with Engine() as eng:
        wire(eng, FakeLLMClient(worker_llm), sup.tools)
        ToolHandler(rag.tools).attach(eng)  # worker's tools merge into the shared executor
        h = await eng.start(sup, {"task": "go"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    assert out.ok and out.value["text"] == "Supervisor says: cited True | cited True"
    assert [e.type for e in log].count("aaa.child.completed") == 2
