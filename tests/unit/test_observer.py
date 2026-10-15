"""The observer: traces leave the process correctly (whole, ordered, once), and watching never harms a run."""

import asyncio
from datetime import UTC, datetime, timedelta

from aaa.algos import Supervisor, ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.llm.types import LLMResponse, Message, ToolCall, Usage
from aaa.observe import CallableSink, ConsoleSink, InMemorySink, Observer
from aaa.runtime import Engine
from aaa.session import SessionManager
from aaa.tools import ToolHandler, ToolRegistry, tool
from aaa.trace import A


@tool
def lookup(q: str) -> str:
    """look something up"""
    return f"answer for {q}"


def priced(content=None, calls=()):
    return LLMResponse(
        message=Message(role="assistant", content=content, tool_calls=list(calls)),
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        model="gpt-test",
        finish_reason="tool_calls" if calls else "stop",
    )


def two_step(req):
    if req.messages[-1].role == "user":
        return priced(calls=[ToolCall(id="c", name="lookup", arguments={"q": "x"})])
    return priced(content="done")


def make_engine(script=two_step):
    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    return eng, algo


def parents_first(spans):
    seen = set()
    for s in spans:
        assert (
            s.parent_id is None
            or s.parent_id in seen
            or s.parent_id not in {x.span_id for x in spans}
        ), f"{s.name} before its parent"
        seen.add(s.span_id)


async def test_a_finished_run_is_exported_once_whole_and_parents_first():
    eng, algo = make_engine()
    sink = InMemorySink()
    async with eng, Observer(eng, sink, settle_s=0):
        out = await eng.run(algo, {"task": "go"})
    assert out.ok and sink.batches == 1
    kinds = [s.kind for s in sink.spans]
    assert kinds[0] == "agent" and kinds.count("llm") == 2 and kinds.count("tool") == 1
    parents_first(sink.spans)
    assert len({s.span_id for s in sink.spans}) == len(sink.spans)


async def test_a_delegating_run_exports_one_trace_for_the_whole_tree():
    def script(req):
        last = req.messages[-1]
        if last.role == "user" and "sub-task" not in (last.content or ""):
            return priced(
                calls=[
                    ToolCall(id="d", name="delegate_helper", arguments={"task": "sub-task please"})
                ]
            )
        return priced(content="helper result" if last.role == "user" else "all done")

    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    helper = ToolLoop(ToolRegistry([]), name="helper")
    sup = Supervisor({"helper": (helper, "does things")})
    ToolHandler(sup.tools).attach(eng)
    eng.register(sup)
    eng.register(helper)
    sink = InMemorySink()
    async with eng, Observer(eng, sink, settle_s=0):
        assert (await eng.run(sup, {"task": "coordinate"})).ok
    assert sink.batches == 1, "the child run is exported with its root, not separately"
    assert len(sink.by_trace()) == 1
    agents = [s for s in sink.spans if s.kind == "agent"]
    assert len(agents) == 2
    delegate = next(s for s in sink.spans if s.kind == "tool")
    child_root = next(s for s in agents if s.parent_id)
    assert child_root.parent_id == delegate.span_id
    parents_first(sink.spans)


async def test_a_broken_sink_never_fails_the_run_and_does_not_block_other_sinks():
    async def broken(spans):
        raise RuntimeError("backend is down")

    good = InMemorySink()
    eng, algo = make_engine()
    obs = Observer(eng, [CallableSink(broken), good], settle_s=0)
    async with eng, obs:
        out = await eng.run(algo, {"task": "go"})
        assert await obs.drain(5)
    assert out.ok and obs.export_errors == 1 and len(good.spans) > 0


async def test_a_slow_sink_times_out_instead_of_hanging_the_observer():
    async def slow(spans):
        await asyncio.sleep(30)

    eng, algo = make_engine()
    obs = Observer(eng, CallableSink(slow), settle_s=0, sink_timeout_s=0.1)
    async with eng, obs:
        assert (await eng.run(algo, {"task": "go"})).ok
        assert await obs.drain(5)
    assert obs.export_errors == 1


async def test_sampling_is_deterministic_per_trace_and_can_exclude_everything():
    eng, algo = make_engine()
    none, everything = InMemorySink(), InMemorySink()
    async with (
        eng,
        Observer(eng, none, sample=0.0, settle_s=0) as a,
        Observer(eng, everything, sample=1.0, settle_s=0, group="other") as b,
    ):
        await eng.run(algo, {"task": "go"})
        await a.drain(5)
        await b.drain(5)
    assert none.spans == [] and everything.spans


async def test_only_the_requested_tenant_is_exported():
    eng, algo = make_engine()
    acme, any_ = InMemorySink(), InMemorySink()
    async with (
        eng,
        Observer(eng, acme, tenant="acme", settle_s=0) as a,
        Observer(eng, any_, settle_s=0, group="all") as b,
    ):
        await eng.tenant("acme").run(algo, {"task": "go"})
        await eng.tenant("globex").run(algo, {"task": "go"})
        await a.drain(5)
        await b.drain(5)
    assert {s.tenant for s in acme.spans} == {"acme"} and {s.tenant for s in any_.spans} == {
        "acme",
        "globex",
    }


async def test_content_is_captured_only_when_asked():
    eng, algo = make_engine()
    quiet, loud = InMemorySink(), InMemorySink()
    async with (
        eng,
        Observer(eng, quiet, settle_s=0) as a,
        Observer(eng, loud, settle_s=0, capture_content=True, group="loud") as b,
    ):
        await eng.run(algo, {"task": "secret question"})
        await a.drain(5)
        await b.drain(5)
    assert all(s.input is None and s.output is None for s in quiet.spans)
    assert any("secret question" in str(s.input) for s in loud.spans)


async def test_an_open_session_is_exported_incrementally_then_finalized():
    eng, _ = make_engine(lambda req: text("hi"))
    mgr = SessionManager(eng, ToolLoop(ToolRegistry([])))
    sink = InMemorySink()
    async with eng, Observer(eng, sink, settle_s=0) as obs:
        chat = await mgr.create(user="ann")
        await chat.send("one")
        assert await obs.flush_open() > 0
        first_ids = {s.span_id for s in sink.spans}
        root = next(s for s in sink.spans if s.run_id == chat.id and s.kind == "agent")
        assert root.attributes.get(A.PARTIAL) is True and root.status == "incomplete"
        await chat.send("two")
        assert await obs.flush_open() > 0
        newly = [s for s in sink.spans if s.span_id not in first_ids]
        assert newly and all(s.attributes.get(A.TURN) != 0 or s.kind != "session" for s in newly)
        assert sum(1 for s in sink.spans if s.span_id == root.span_id) == 1, (
            "the provisional root is not repeated"
        )
        again = await obs.flush_open()
        assert again == 0, "nothing new, nothing sent"
        await chat.close()
        await obs.drain(5)
    finals = [s for s in sink.spans if s.span_id == root.span_id]
    assert (
        len(finals) == 2 and finals[-1].status == "ok" and not finals[-1].attributes.get(A.PARTIAL)
    )


async def test_history_can_be_exported_after_the_fact():
    eng, algo = make_engine()
    async with eng:
        for _ in range(3):
            await eng.run(algo, {"task": "go"})
        sink = InMemorySink()
        obs = Observer(eng, sink, settle_s=0)
        assert await obs.backfill() > 0 and len(sink.by_trace()) == 3
        assert await obs.backfill(since=datetime.now(UTC) + timedelta(days=1)) == 0
        again = len(sink.spans)
        await obs.backfill()
        assert len(sink.spans) == again, "exporting the same traces again sends nothing new"


async def test_the_console_sink_prints_a_readable_tree():
    import io

    eng, algo = make_engine()
    buf = io.StringIO()
    async with eng, Observer(eng, ConsoleSink(buf), settle_s=0) as obs:
        await eng.run(algo, {"task": "go"})
        await obs.drain(5)
    text_ = buf.getvalue()
    assert text_.splitlines()[0].startswith("invoke_agent tool_loop")
    assert (
        "  chat gpt-test" in text_
        and "  execute_tool lookup" in text_
        and "10+5 tok" in text_
        and "$0.0010" in text_
    )
