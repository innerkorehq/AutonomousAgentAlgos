import asyncio

from aaa.algos import ReAct, ToolLoop
from aaa.core import OutcomeKind
from aaa.data import InMemoryBlobStore, InMemoryIndex
from aaa.data.index import Doc
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, search_tool, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(requires_approval=True)
def wire_money(amount: int) -> str:
    """Send money."""
    return f"sent {amount}"


@tool
async def slow() -> str:
    """Never finishes in time."""
    await asyncio.sleep(5)
    return "x"


slow.timeout_s = 0.05


def setup(eng, llm, algo_tools):
    LLMHandler(llm).attach(eng)
    ToolHandler(algo_tools).attach(eng)


def last_user_or_tool(req):
    return req.messages[-1]


async def test_toolloop_end_to_end_with_tool_error_recovery():
    reg = ToolRegistry([add, slow])

    def script(req):
        m = req.messages[-1]
        if m.role == "user":
            return tool_call("add", a=1, b="oops")  # bad args -> model sees the error
        if m.role == "tool" and m.content.startswith("Error (InvalidArguments)"):
            return tool_call("add", a=2, b=3)
        if m.role == "tool":
            return text(f"The answer is {m.content}")
        raise AssertionError(m)

    async with Engine() as eng:
        setup(eng, FakeLLMClient(script), reg)
        out = await eng.run(ToolLoop(reg, system="be brief"), {"task": "2+3?"})
    assert out.ok and out.value == {"text": "The answer is 5", "turns": 3}


async def test_toolloop_parallel_calls_unknown_tool_timeout_and_max_turns():
    reg = ToolRegistry([add, slow])
    state = {"n": 0}

    def script(req):
        state["n"] += 1
        if req.messages[-1].role == "user":
            from aaa.llm import LLMResponse, Message, ToolCall

            return LLMResponse(
                message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="x", name="add", arguments={"a": 1, "b": 1}),
                        ToolCall(id="y", name="nope", arguments={}),
                        ToolCall(id="z", name="slow", arguments={}),
                    ],
                )
            )
        tools = [m for m in req.messages if m.role == "tool"]
        assert tools[0].content == "2"  # order preserved
        assert "UnknownTool" in tools[1].content and "Timeout" in tools[2].content
        return text("done")

    async with Engine() as eng:
        setup(eng, FakeLLMClient(script), reg)
        assert (await eng.run(ToolLoop(reg), {"task": "go"})).ok

    async with Engine() as eng:  # a model that never stops calling tools
        setup(eng, FakeLLMClient(lambda r: tool_call("add", a=1, b=1)), reg)
        out = await eng.run(ToolLoop(reg, max_turns=3), {"task": "loop"})
    assert out.kind is OutcomeKind.NO_SOLUTION


async def test_approval_gate_blocks_until_granted_and_deny_is_reported():
    reg = ToolRegistry([wire_money])

    def script(req):
        m = req.messages[-1]
        return tool_call("wire_money", amount=5) if m.role == "user" else text(m.content)

    for grant, expect in ((True, "sent 5"), (False, "ApprovalDenied")):
        async with Engine() as eng:
            setup(eng, FakeLLMClient(script), reg)
            h = await eng.start(ToolLoop(reg), {"task": "pay"})
            await asyncio.sleep(0.05)
            log = await eng.store.read(h.run_id)
            assert not any(
                e.type in ("aaa.tool.completed", "aaa.tool.failed") for e in log
            )  # blocked
            req = next(e for e in log if e.type == "aaa.approval.requested")
            await h.signal(
                "aaa.approval.granted" if grant else "aaa.approval.denied",
                {"call_id": req.data["call_id"]},
            )
            await h.signal(
                "aaa.approval.granted", {"call_id": req.data["call_id"]}
            )  # duplicate: no re-run
            out = await h.result()
        assert expect in out.value["text"]


async def test_react_text_protocol_and_format_recovery():
    reg = ToolRegistry([add])
    replies = iter(
        [
            "Thought: need add\nAction: add\nAction Input: not json",
            'Thought: retry\nAction: add\nAction Input: {"a": 4, "b": 5}',
            "Thought: done\nFinal Answer: 9",
        ]
    )
    llm = FakeLLMClient(lambda r: text(next(replies)))
    async with Engine() as eng:
        setup(eng, llm, reg)
        out = await eng.run(ReAct(reg), {"task": "4+5"})
    assert out.ok and out.value["text"] == "9"
    assert (
        "Observation: Error: Action Input must be a JSON object"
        in llm.requests[1].messages[-1].content
    )
    assert llm.requests[2].messages[-1].content == "Observation: 9"
    assert "- add: Add two integers." in llm.requests[0].messages[0].content


async def test_claim_check_large_payloads_are_transparent():
    @tool
    def big() -> str:
        """Big output."""
        return "x" * 50_000

    reg = ToolRegistry([big])
    seen = {}

    def script(req):
        m = req.messages[-1]
        if m.role == "user":
            return tool_call("big")
        seen["len"] = len(m.content)
        return text("ok")

    blobs = InMemoryBlobStore()
    async with Engine(blobs=blobs, claim_threshold=1000) as eng:
        setup(eng, FakeLLMClient(script), reg)
        h = await eng.start(ToolLoop(reg), {"task": "go"})
        assert (await h.result()).ok
        raw = await eng.store.read(h.run_id)
    assert seen["len"] == 50_000  # handlers/algos see full data
    claimed = [e for e in raw if e.blobs]
    assert claimed and all(e.data.get("$claim") is True for e in claimed)  # log stays small
    assert all(len(e.model_dump_json()) < 5000 for e in raw)


async def test_budget_stops_toolloop_and_rag_tool():
    idx = InMemoryIndex()
    await idx.upsert(
        [
            Doc(id="d1", text="The capital of France is Paris"),
            Doc(id="d2", text="Bananas are yellow"),
        ]
    )
    reg = ToolRegistry([search_tool(idx)])

    def script(req):
        m = req.messages[-1]
        return (
            tool_call("search_documents", query="capital France")
            if m.role == "user"
            else text(m.content[:40])
        )

    async with Engine() as eng:
        setup(eng, FakeLLMClient(script), reg)
        out = await eng.run(ToolLoop(reg), {"task": "capital?"})
    assert out.ok and '"id": "d1"' in out.value["text"]
    async with Engine() as eng:
        setup(eng, FakeLLMClient(lambda r: tool_call("search_documents", query="x")), reg)
        out = await eng.run(ToolLoop(reg, max_turns=99), {"task": "x"}, budget=Budget(max_steps=6))
    assert out.kind is OutcomeKind.BUDGET_EXCEEDED


async def test_toolloop_stream_option_emits_deltas_and_gives_the_same_answer():
    reg = ToolRegistry([add])

    def script(req):
        m = req.messages[-1]
        return tool_call("add", a=2, b=3) if m.role == "user" else text(f"the sum is {m.content}")

    async with Engine() as eng:
        setup(eng, FakeLLMClient(script), reg)
        h = await eng.start(ToolLoop(reg, stream=True), {"task": "2+3"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    deltas = "".join(e.data["text"] for e in log if e.type == "aaa.llm.delta")
    assert out.ok and out.value["text"] == "the sum is 5" and deltas == "the sum is 5"
