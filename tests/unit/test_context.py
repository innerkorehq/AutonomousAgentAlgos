"""Context management: strategies keep a conversation valid, the policy fits it to a window, and the
LLMHandler applies it as a send-time view without rewriting the logical conversation or the log."""

import random

import pytest

from aaa.algos import ToolLoop
from aaa.context import (
    CalibratedCounter,
    ClearToolResults,
    ContextPolicy,
    DropOldest,
    HeuristicCounter,
    Summarize,
    SummaryBook,
    TruncateToolResults,
    groups,
)
from aaa.context.strategies import Work, head_end, is_summary
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.errors import ContextOverflow
from aaa.llm.fake import text, tool_call
from aaa.llm.types import LLMRequest, LLMResponse, Message, ToolCall, ToolSpec
from aaa.runtime import Engine, Ledger
from aaa.tools import ToolHandler, ToolRegistry, tool

# ---- helpers ----------------------------------------------------------------------------------


def turn(i: int, size: int = 400, calls: int = 1) -> list[Message]:
    cs = [
        ToolCall(id=f"c{i}_{k}", name="lookup", arguments={"q": f"item {i}"}) for k in range(calls)
    ]
    return [
        Message(role="assistant", content=f"thinking about {i}", tool_calls=cs),
        *[
            Message(
                role="tool", tool_call_id=c.id, name="lookup", content=f"result {i} " + "x" * size
            )
            for c in cs
        ],
    ]


def convo(n: int, size: int = 400) -> list[Message]:
    msgs = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="the task"),
    ]
    for i in range(n):
        msgs += turn(i, size)
    return msgs


def well_formed(msgs: list[Message]) -> bool:
    """Every tool message answers a call made by the assistant message that starts its group."""
    for s, e in groups(msgs):
        first = msgs[s]
        if first.role == "tool":
            return False
        if first.role == "assistant" and first.tool_calls:
            ids = {c.id for c in first.tool_calls}
            if any(m.role != "tool" or m.tool_call_id not in ids for m in msgs[s + 1 : e]):
                return False
    return True


# ---- tokens ------------------------------------------------------------------------------------


def test_the_counter_grows_with_content_and_counts_tools_and_calls():
    c = HeuristicCounter()
    small, big = [Message(role="user", content="hi")], [Message(role="user", content="hi " * 500)]
    assert c.count(big) > c.count(small) > 0
    tools = [
        ToolSpec(
            name="lookup", description="d" * 200, parameters={"type": "object", "properties": {}}
        )
    ]
    assert c.count(small, tools) > c.count(small)
    assert c.count(turn(1)) > c.count([Message(role="assistant", content="thinking about 1")])


def test_the_calibrated_counter_learns_the_providers_real_count():
    msgs = convo(5)
    c = CalibratedCounter(HeuristicCounter())
    raw = c.base.count(msgs)
    assert c.count(msgs) == raw
    for _ in range(12):  # the provider consistently reports 1.5x what we estimate
        c.observe("m", raw, int(raw * 1.5))
    assert 1.4 < c.ratio("m") < 1.6
    c.model = "m"
    assert abs(c.count(msgs) - raw * 1.5) < raw * 0.1
    c.observe("m", raw, raw * 100)  # an absurd reading cannot blow the estimate up
    assert c.ratio("m") < 3.0


# ---- structure ---------------------------------------------------------------------------------


def test_groups_keep_tool_calls_with_their_results():
    msgs = [
        Message(role="user", content="u"),
        *turn(0, calls=3),
        Message(role="assistant", content="done"),
    ]
    assert groups(msgs) == [(0, 1), (1, 5), (5, 6)]
    assert head_end(convo(2)) == 2
    assert (
        head_end(
            [
                Message(role="system", content="a"),
                Message(role="system", content="b"),
                Message(role="user", content="u"),
            ]
        )
        == 3
    )


def make_work(msgs, target, *, keep_recent=3):
    counter = HeuristicCounter()
    return Work(
        messages=list(msgs),
        tools=[],
        counter=counter,
        target=target,
        keep_recent=keep_recent,
        summary_upto=0,
        original=list(msgs),
    )


@pytest.mark.parametrize("seed", range(25))
async def test_no_strategy_ever_breaks_the_conversation(seed):
    rng = random.Random(seed)
    msgs = [Message(role="system", content="sys"), Message(role="user", content="task")]
    for i in range(rng.randint(3, 25)):
        msgs += turn(i, size=rng.choice([50, 800, 9000]), calls=rng.randint(1, 3))
        if rng.random() < 0.3:
            msgs.append(Message(role="user", content=f"follow-up {i}"))
    base = HeuristicCounter()
    for strategy in (
        TruncateToolResults(2000),
        ClearToolResults(2),
        DropOldest(),
        Summarize(summarize=_stub_summary()),
    ):
        w = make_work(msgs, target=max(300, base.count(msgs) // rng.randint(2, 6)))
        before = w.tokens()
        await strategy.apply(w)
        out = w.messages
        assert well_formed(out), strategy.name
        assert out[:2] == msgs[:2], "the head (instructions and task) is never touched"
        assert w.tokens() <= before, "a strategy never makes it bigger"
        tail = groups(msgs, 2)[-3:]
        tail_msgs = [m for s, e in tail for m in msgs[s:e]]
        assert out[len(out) - len(tail_msgs) :] == tail_msgs, (
            "the most recent turns survive verbatim"
        )


# ---- individual strategies ---------------------------------------------------------------------


async def test_truncate_shortens_only_old_oversized_results():
    msgs = convo(6, size=5000)
    w = make_work(msgs, target=10)
    await TruncateToolResults(max_chars=1000).apply(w)
    lengths = [len(m.content or "") for m in w.messages if m.role == "tool"]
    assert all(n < 1300 for n in lengths[:-3]) and all(n > 4000 for n in lengths[-3:])
    assert "characters removed" in w.messages[3].content and w.applied == ["truncate_tool_results"]
    assert w.messages[3].content.startswith("result 0")  # the start survives


async def test_clear_blanks_old_results_but_keeps_the_calls():
    msgs = convo(8)
    w = make_work(msgs, target=10)
    await ClearToolResults(keep_last=2).apply(w)
    results = [m.content for m in w.messages if m.role == "tool"]
    assert results[:-3].count("[tool result removed to save context]") == len(results) - 3
    assert results[-1].startswith("result 7") and sum(1 for m in w.messages if m.tool_calls) == 8
    await ClearToolResults(keep_last=2).apply(w)  # idempotent
    assert w.applied == ["clear_tool_results"]


async def test_drop_oldest_fits_the_budget_and_says_it_dropped_something():
    msgs = convo(20, size=600)
    w = make_work(msgs, target=HeuristicCounter().count(msgs) // 3)
    await DropOldest().apply(w)
    assert w.fits() and well_formed(w.messages)
    marker = w.messages[head_end(w.messages)]
    assert marker.role == "system" and "removed to fit the context window" in marker.content


def _stub_summary(calls=None):
    async def fn(text_: str, previous: str | None) -> str:
        if calls is not None:
            calls.append((text_, previous))
        return f"SUMMARY<{len(text_)}>" + (f" after [{previous}]" if previous else "")

    return fn


async def test_summarize_replaces_the_old_middle_and_records_what_it_covers():
    msgs = convo(14, size=600)
    calls: list = []
    w = make_work(msgs, target=HeuristicCounter().count(msgs) // 3)
    await Summarize(summarize=_stub_summary(calls)).apply(w)
    h = head_end(w.messages)
    assert is_summary(w.messages[h]) and well_formed(w.messages) and w.fits()
    (rec,) = w.new_summaries
    assert 2 < rec.upto < len(msgs) and rec.text.startswith("SUMMARY")
    assert w.messages[h + 1 :] == msgs[rec.upto :]  # everything after the summary is verbatim
    assert len(calls) == 1 and "result 0" in calls[0][0]


# ---- policy ------------------------------------------------------------------------------------


async def test_below_the_trigger_the_conversation_is_left_alone():
    p = ContextPolicy(max_input_tokens=100_000)
    msgs = convo(5)
    out = await p.prepare(msgs)
    assert out.messages == msgs and out.report is None


async def test_compaction_stops_at_the_target_not_the_trigger():
    msgs = convo(40, size=600)
    p = ContextPolicy(max_input_tokens=6000, reserve_output=0, trigger=0.85, target=0.5)
    out = await p.prepare(msgs)
    budget = 6000
    assert out.report and out.report["tokens_before"] > budget
    assert out.report["tokens_after"] <= budget * 0.5 and well_formed(out.messages)
    assert out.report["applied"], "something was applied"


async def test_a_stored_summary_is_reused_by_content_not_regenerated():
    msgs = convo(40, size=600)
    calls: list = []
    p = ContextPolicy(
        max_input_tokens=8000,
        reserve_output=0,
        strategies=[Summarize(summarize=_stub_summary(calls)), DropOldest()],
    )
    book = SummaryBook()
    first = await p.prepare(msgs, book=book)
    for rec in first.new_summaries:
        book.add(rec)
    assert len(calls) == 1 and len(book.records) == 1
    grown = [*msgs, *turn(30, 600), *turn(31, 600)]
    second = await p.prepare(grown, book=book)  # same prefix + 2 new turns: reuse, no new call
    assert len(calls) == 1 and second.new_summaries == []
    assert (
        second.messages[: head_end(second.messages) + 1]
        == first.messages[: head_end(first.messages) + 1]
    )
    other = [msgs[0], Message(role="user", content="a DIFFERENT task"), *msgs[2:]]
    await p.prepare(other, book=book)  # the hash differs: the old summary must not be applied
    assert len(calls) == 2


async def test_the_next_summary_folds_in_the_previous_one():
    calls: list = []
    p = ContextPolicy(
        max_input_tokens=5000,
        reserve_output=0,
        strategies=[Summarize(summarize=_stub_summary(calls))],
    )
    book = SummaryBook()
    msgs = convo(25, size=600)
    r1 = await p.prepare(msgs, book=book)
    [book.add(r) for r in r1.new_summaries]
    longer = [*msgs, *[m for i in range(30, 60) for m in turn(i, 600)]]
    r2 = await p.prepare(longer, book=book)
    assert r2.new_summaries and calls[-1][1] is not None and calls[-1][1].startswith("SUMMARY")
    assert r2.new_summaries[0].upto > r1.new_summaries[0].upto


def test_invalid_budgets_are_rejected():
    with pytest.raises(ValueError):
        ContextPolicy(max_input_tokens=1000, trigger=0.5, target=0.6)
    with pytest.raises(ValueError):
        Summarize()


# ---- through the engine -----------------------------------------------------------------------


@tool
def lookup(q: str) -> str:
    """look something up"""
    return f"answer for {q} " + "z" * 1500


def long_run_script(turns: int, seen: list[LLMRequest]):
    def script(req):
        seen.append(req)
        return (
            tool_call("lookup", q=f"step {len(seen)}") if len(seen) <= turns else text("all done")
        )

    return script


async def run_long(policy, turns=25, handler_hook=None, **kw):
    seen: list[LLMRequest] = []
    eng = Engine()
    handler = LLMHandler(FakeLLMClient(long_run_script(turns, seen)), context=policy, **kw)
    handler.attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg, max_turns=turns + 5)
    eng.register(algo)
    async with eng:
        h = await eng.start(algo, {"task": "keep going"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    return out, log, seen, handler


async def test_a_long_run_stays_inside_the_window_and_still_finishes():
    policy = ContextPolicy(max_input_tokens=3500, reserve_output=0)
    out, _, seen, _ = await run_long(policy, turns=25)
    assert out.ok and out.value["text"] == "all done"
    counter = HeuristicCounter()
    biggest = max(counter.count(r.messages, r.tools) for r in seen)
    assert biggest <= 3500, f"a request of ~{biggest} tokens was sent"
    assert all(well_formed(r.messages) for r in seen)
    unmanaged = max(counter.count(r.messages) for r in (await run_long(None, turns=25))[2])
    assert unmanaged > 3500 * 3, "without a policy the same run would be far over the window"


async def test_the_log_and_the_delta_requests_are_not_rewritten_by_compaction():
    policy = ContextPolicy(max_input_tokens=3500, reserve_output=0)
    _, log, _, handler = await run_long(policy, turns=25)
    requests = [e for e in log if e.type == "aaa.llm.requested"]
    assert all("conv" in e.data for e in requests[1:]), (
        "later requests are still deltas against the previous one"
    )
    assert all(len(e.data["conv"]["delta"]) <= 2 for e in requests[1:])
    completed = [e for e in log if e.type == "aaa.llm.completed"]
    reports = [e.data["context"] for e in completed if "context" in e.data]
    assert reports and all(r["tokens_after"] <= r["tokens_before"] for r in reports)
    assert handler.rebuilds == 0  # the cached LOGICAL transcript kept growing normally


async def test_summaries_are_events_reused_after_a_restart_and_billed_to_the_run():
    calls: list = []
    policy = ContextPolicy(
        max_input_tokens=3500,
        reserve_output=0,
        strategies=[Summarize(summarize=_stub_summary(calls)), DropOldest()],
    )
    seen: list[LLMRequest] = []
    eng = Engine()
    holder: dict = {}

    def script(req):
        seen.append(req)
        if len(seen) == 16:  # mid-run, "the process restarts": forget every in-memory cache
            holder["h"]._books.clear()
            holder["h"]._convs.clear()
            holder["h"]._remember = lambda *a, **k: None  # and do not refill the transcript cache
        return tool_call("lookup", q=f"step {len(seen)}") if len(seen) <= 22 else text("done")

    handler = LLMHandler(FakeLLMClient(script), context=policy)
    holder["h"] = handler
    handler.attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg, max_turns=40)
    eng.register(algo)
    async with eng:
        h = await eng.start(algo, {"task": "go"})
        assert (await h.result()).ok
        log = await eng.store.read(h.run_id)
    summaries = [e for e in log if e.type == "aaa.context.summary"]
    assert summaries and len(summaries) == len(calls)
    assert len({e.data["upto"] for e in summaries}) == len(summaries)  # one per distinct prefix
    assert handler.rebuilds >= 1, "the restart really forced a rebuild from the log"
    assert all(is_summary(next(m for m in r.messages if is_summary(m))) for r in seen[-3:])
    # a summary event that reports spend is counted by the ledger without counting as a step
    spent = Ledger().fold(
        summaries[0].model_copy(
            update={"data": {**summaries[0].data, "usage": {"total_tokens": 321}, "cost": 0.5}}
        )
    )
    assert spent.tokens == 321 and spent.cost == 0.5 and spent.steps == 0


async def test_a_provider_overflow_triggers_one_deeper_retry():
    class Strict(FakeLLMClient):
        """A provider whose tokenizer counts 1.5x what our estimate says, with a 3000-token window."""

        def __init__(self) -> None:
            super().__init__(lambda req: text("ok"))
            self.sizes: list[float] = []

        async def complete(self, request, /):
            n = HeuristicCounter().count(request.messages, request.tools) * 1.5
            self.sizes.append(n)
            if n > 3000:
                raise ContextOverflow("too long")
            return await super().complete(request)

    client = Strict()
    # we believe the window is 3000 and our estimate is exact; the provider disagrees
    policy = ContextPolicy(max_input_tokens=3000, reserve_output=0, trigger=0.9, target=0.8)
    eng = Engine()
    LLMHandler(client, context=policy).attach(eng)
    algo = ToolLoop(ToolRegistry([]))
    eng.register(algo)
    msgs = [m.model_dump() for m in convo(40, 500)]
    async with eng:
        out = await eng.run(algo, {"messages": msgs})
    assert out.ok and len(client.sizes) == 2, client.sizes  # refused once, then fitted
    assert client.sizes[0] > 3000 >= client.sizes[1]


def test_response_type_is_unchanged_by_the_context_feature():
    assert "context" not in LLMResponse.model_fields
