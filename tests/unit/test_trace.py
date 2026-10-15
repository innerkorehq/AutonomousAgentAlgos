"""Traces derived from the event log: a pure, deterministic function, valid for any run shape."""

import asyncio
from datetime import UTC, datetime, timedelta

from aaa.algos import Supervisor, ToolLoop
from aaa.events import Event
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.types import LLMResponse, Message, ToolCall, Usage
from aaa.runtime import Engine
from aaa.safe import safe_subscribe
from aaa.tools import ToolHandler, ToolRegistry, tool
from aaa.trace import A, children_of, span_id_of, spans_from_events, trace_id_of


@tool
def lookup(q: str) -> str:
    """look something up"""
    return f"answer for {q}"


@tool
def boom() -> str:
    """always fails"""
    raise ValueError("kaboom")


def priced(content=None, calls=(), **usage):
    return LLMResponse(
        message=Message(role="assistant", content=content, tool_calls=list(calls)),
        usage=Usage(
            prompt_tokens=usage.get("p", 10),
            completion_tokens=usage.get("c", 5),
            total_tokens=usage.get("p", 10) + usage.get("c", 5),
            cached_tokens=usage.get("cached", 0),
        ),
        cost=0.002,
        model="gpt-test",
        finish_reason="tool_calls" if calls else "stop",
    )


async def run(script, tools=(lookup,), algo_factory=None, **start_kw):
    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry(list(tools))
    ToolHandler(reg).attach(eng)
    algo = algo_factory(reg) if algo_factory else ToolLoop(reg)
    eng.register(algo)
    async with eng:
        h = await eng.start(algo, {"task": "go"}, **start_kw)
        out = await h.result()
        events = await eng.store.read(h.run_id)
        return out, events, eng, h


def two_step(req):
    if req.messages[-1].role == "user":
        return priced(calls=[ToolCall(id="c", name="lookup", arguments={"q": "x"})])
    return priced(content="done", cached=4)


def assert_valid_tree(spans):
    ids = [s.span_id for s in spans]
    assert len(set(ids)) == len(ids), "span ids are unique"
    roots = [s for s in spans if s.parent_id is None or s.parent_id not in set(ids)]
    assert len(roots) == 1 and roots[0].kind == "agent"
    for s in spans:
        assert s.end is None or s.end >= s.start, f"negative duration on {s.name}"
        assert len(s.trace_id) == 32 and len(s.span_id) == 16
    return roots[0]


async def test_a_tool_run_becomes_an_agent_span_with_llm_and_tool_children():
    out, events, *_ = await run(two_step)
    assert out.ok
    spans = spans_from_events(events)
    root = assert_valid_tree(spans)
    assert (
        root.name == "invoke_agent tool_loop"
        and root.attributes[A.OUTCOME] == "success"
        and root.status == "ok"
    )
    llm = [s for s in spans if s.kind == "llm"]
    tools = [s for s in spans if s.kind == "tool"]
    assert (
        len(llm) == 2 and len(tools) == 1 and all(s.parent_id == root.span_id for s in llm + tools)
    )
    first, second = sorted(llm, key=lambda s: s.start)
    assert (
        first.attributes[A.MODEL] == "gpt-test"
        and first.attributes[A.INPUT_TOKENS] == 10
        and first.attributes[A.COST] == 0.002
    )
    assert second.attributes[A.CACHED_TOKENS] == 4 and second.attributes[A.FINISH_REASON] == "stop"
    t = tools[0]
    assert (
        t.name == "execute_tool lookup"
        and t.attributes[A.TOOL_NAME] == "lookup"
        and t.attributes[A.TOOL_ATTEMPTS] == 1
        and t.status == "ok"
    )
    assert root.start <= first.start and second.end <= root.end


async def test_the_same_events_always_give_the_same_trace():
    _, events, *_ = await run(two_step)
    a, b = spans_from_events(events), spans_from_events(list(events))
    assert [(s.trace_id, s.span_id, s.parent_id, s.name) for s in a] == [
        (s.trace_id, s.span_id, s.parent_id, s.name) for s in b
    ]
    assert a[0].trace_id == trace_id_of(events[0].run_id) and a[0].span_id == span_id_of(
        "run", events[0].run_id
    )


async def test_content_is_left_out_unless_asked_for_and_then_clipped():
    _, events, *_ = await run(two_step)
    quiet = spans_from_events(events)
    assert all(s.input is None and s.output is None for s in quiet)
    loud = spans_from_events(events, capture_content=True, content_limit=40)
    tool_span = next(s for s in loud if s.kind == "tool")
    assert "answer for x" in tool_span.output and "q" in tool_span.input
    clipped = [s.input for s in loud if s.kind == "llm" and isinstance(s.input, str)]
    assert any("more characters" in c for c in clipped), (
        "long content is clipped, with a note of how much was cut"
    )
    assert all(len(c) <= 40 + 40 for c in clipped)


async def test_failures_and_errors_show_up_as_error_spans():
    def script(req):
        if req.messages[-1].role == "user":
            return priced(calls=[ToolCall(id="c", name="boom", arguments={})])
        return priced(content="recovered")

    out, events, *_ = await run(script, tools=(boom,))
    spans = spans_from_events(events)
    t = next(s for s in spans if s.kind == "tool")
    assert (
        t.status == "error"
        and "kaboom" in (t.status_message or "")
        and t.attributes[A.ERROR_TYPE] == "ValueError"
    )
    assert out.ok and next(s for s in spans if s.kind == "agent").status == "ok"


async def test_a_provider_error_is_an_error_llm_span_and_a_failed_run_is_an_error_root():
    from aaa.llm.errors import FatalError

    def explode(req):
        raise FatalError("provider down")

    out, events, *_ = await run(explode)
    spans = spans_from_events(events)
    assert not out.ok
    assert next(s for s in spans if s.kind == "llm").status == "error"
    root = next(s for s in spans if s.kind == "agent")
    assert root.status == "error" and "provider down" in (root.status_message or "")


async def test_a_run_still_in_progress_has_open_spans():
    _, events, *_ = await run(two_step)
    truncated = [e for e in events if e.type != "aaa.run.completed"]
    spans = spans_from_events(truncated)
    root = next(s for s in spans if s.kind == "agent")
    assert root.end is None and root.status == "incomplete" and not root.closed
    assert spans_from_events([]) == []


async def test_a_delegated_child_run_is_parented_to_the_tool_call_that_started_it():
    def script(req):
        last = req.messages[-1]
        if last.role == "user" and "sub-task" not in (last.content or ""):
            return priced(
                calls=[
                    ToolCall(id="d", name="delegate_helper", arguments={"task": "sub-task please"})
                ]
            )
        if last.role == "user":
            return priced(content="helper result")
        return priced(content="all done")

    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    helper = ToolLoop(ToolRegistry([]), name="helper")
    sup = Supervisor({"helper": (helper, "does things")})
    ToolHandler(sup.tools).attach(eng)
    eng.register(sup)
    eng.register(helper)
    async with eng:
        h = await eng.start(sup, {"task": "coordinate"})
        assert (await h.result()).ok
        parent_events = await eng.store.read(h.run_id)
        (child_id,) = children_of(parent_events)
        child_events = await eng.store.read(child_id)
    parent_spans = spans_from_events(parent_events)
    linked = spans_from_events(child_events, parent_events=parent_events)
    unlinked = spans_from_events(child_events)
    delegate = next(s for s in parent_spans if s.kind == "tool")
    assert linked[0].parent_id == delegate.span_id, "exact parent: the delegating tool call"
    assert unlinked[0].parent_id == next(s for s in parent_spans if s.kind == "agent").span_id, (
        "fallback: the parent run"
    )
    assert linked[0].trace_id == parent_spans[0].trace_id, "one trace for the whole tree"
    assert_valid_tree([*parent_spans, *linked])


async def test_user_and_tenant_are_carried_on_every_span():
    _, events, *_ = await run(two_step, tenant="acme", meta={"user": "ann"})
    assert all(s.tenant == "acme" and s.user == "ann" for s in spans_from_events(events))


def test_guardrail_and_eval_events_map_to_spans():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    base = {"stream": "r", "run_id": "r", "trace_id": "r"}
    events = [
        Event(id="r:started", type="aaa.run.started", time=t0, data={"algo": "x"}, **base),
        Event(
            id="g1",
            type="aaa.guard.verdict",
            time=t0 + timedelta(seconds=1),
            data={"guard": "pii", "stage": "input", "action": "block", "categories": ["email"]},
            **base,
        ),
        Event(
            id="s1",
            type="aaa.eval.score",
            time=t0 + timedelta(seconds=2),
            data={"name": "relevance", "value": 0.9, "usage": {"total_tokens": 7}, "cost": 0.01},
            **base,
        ),
        Event(
            id="r:completed",
            type="aaa.run.completed",
            time=t0 + timedelta(seconds=3),
            data={"kind": "failed", "reason": "blocked"},
            **base,
        ),
    ]
    spans = {s.kind: s for s in spans_from_events(events)}
    assert spans["guardrail"].status == "error" and "email" in spans["guardrail"].status_message
    assert spans["guardrail"].attributes[A.GUARD_ACTION] == "block"
    assert (
        spans["evaluator"].attributes[A.SCORE_VALUE] == 0.9
        and spans["evaluator"].attributes[A.TOTAL_TOKENS] == 7
    )


# ---- safe subscribers --------------------------------------------------------------------------
async def test_a_crashing_watcher_neither_fails_nor_slows_the_run():
    seen: list[str] = []

    async def crashy(e):
        seen.append(e.type)
        raise RuntimeError("watcher bug")

    eng = Engine()
    LLMHandler(FakeLLMClient(two_step)).attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    worker = safe_subscribe(eng, "aaa.*", crashy, name="test-watcher")
    async with eng:
        out = await eng.run(algo, {"task": "go"})
        assert await worker.drain(5)
    assert out.ok, "the watcher's exceptions did not fail the run"
    assert worker.errors == len(seen) > 5


async def test_a_full_queue_drops_events_instead_of_blocking_the_run():
    gate = asyncio.Event()

    async def stuck(e):
        await gate.wait()

    eng = Engine()
    LLMHandler(FakeLLMClient(two_step)).attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    worker = safe_subscribe(eng, "aaa.*", stuck, name="slow", maxsize=3)
    async with eng:
        out = await asyncio.wait_for(eng.run(algo, {"task": "go"}), 10)
        assert out.ok and worker.dropped > 0
        gate.set()


def test_score_helpers():
    from aaa.score import Score

    assert Score(name="a", value=True).numeric == 1.0 and Score(name="a", value=0.2).passed is False
    assert (
        Score(name="a", value="good").numeric is None
        and Score(name="a", value="good").passed is None
    )
