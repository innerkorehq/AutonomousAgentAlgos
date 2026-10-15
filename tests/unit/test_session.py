"""Chat sessions: durable multi-turn conversations built from the engine's own parts."""

import asyncio

import pytest

from aaa.algos import ToolLoop
from aaa.context import ContextPolicy, DropOldest, Summarize
from aaa.core import OutcomeKind
from aaa.events import InMemoryEventStore
from aaa.llm import FakeLLMClient, FakeStatefulClient, LLMHandler
from aaa.llm.errors import FatalError
from aaa.llm.fake import text
from aaa.llm.types import LLMResponse, Message, Usage
from aaa.runtime import Budget, Engine
from aaa.session import SessionBusy, SessionClosed, SessionManager
from aaa.tools import ToolRegistry


def echo(req):
    users = [m.content for m in req.messages if m.role == "user"]
    assistants = [m for m in req.messages if m.role == "assistant"]
    return text(f"seen {len(users)} user and {len(assistants)} assistant; last: {users[-1]}")


def build(script=echo, *, client=None, system="Be brief.", store=None, algo=None, **mgr_kw):
    eng = Engine(store=store)
    client = client or FakeLLMClient(script)
    LLMHandler(client).attach(eng)
    mgr = SessionManager(eng, algo or ToolLoop(ToolRegistry([])), system=system, **mgr_kw)
    return eng, mgr, client


async def test_the_conversation_is_remembered_from_turn_to_turn():
    eng, mgr, client = build()
    async with eng:
        chat = await mgr.create(user="ann")
        r1 = await chat.send("my name is Ann")
        r2 = await chat.send("what is my name?")
        assert r1.ok and r1.text == "seen 1 user and 0 assistant; last: my name is Ann"
        assert r2.text == "seen 2 user and 1 assistant; last: what is my name?"
        history = await chat.history()
        assert [(t.n, t.user, t.status) for t in history] == [
            (0, "my name is Ann", "done"),
            (1, "what is my name?", "done"),
        ]
        msgs = await chat.messages()
        assert [m.role for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
        assert client.requests[1].messages[0] == Message(role="system", content="Be brief.")


async def test_only_one_turn_runs_at_a_time():
    gate = asyncio.Event()

    class Slow(FakeLLMClient):
        async def complete(self, request, /):
            await gate.wait()
            return await super().complete(request)

    eng, mgr, _ = build(client=Slow(echo))
    async with eng:
        chat = await mgr.create()
        first = asyncio.create_task(chat.send("first"))
        for _ in range(100):
            if any(t.status == "pending" for t in await chat.history()):
                break
            await asyncio.sleep(0.01)
        with pytest.raises(SessionBusy):
            await chat.send("second, too soon")
        gate.set()
        assert (await first).ok
        assert (await chat.send("now it is fine")).ok


async def test_two_senders_racing_for_the_same_turn_cannot_both_win():
    eng, mgr, _ = build()
    async with eng:
        a = await mgr.create()
        b = await mgr.get(a.id)
        results = await asyncio.gather(a.send("from a"), b.send("from b"), return_exceptions=True)
        wins = [r for r in results if not isinstance(r, Exception)]
        busy = [r for r in results if isinstance(r, SessionBusy)]
        assert len(wins) == 1 and len(busy) == 1, results
        assert len(await a.history()) == 1


async def test_a_failed_turn_leaves_no_trace_in_what_the_model_sees_next():
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FatalError("provider exploded")
        return echo(req)

    eng, mgr, _ = build(flaky)
    async with eng:
        chat = await mgr.create()
        bad = await chat.send("this one fails")
        assert not bad.ok and bad.kind == "failed" and "exploded" in (bad.reason or "")
        good = await chat.send("this one works")
        assert good.ok and good.text == "seen 1 user and 0 assistant; last: this one works"
        assert [t.status for t in await chat.history()] == ["failed", "done"]


async def test_closing_ends_the_session_and_it_is_an_ordinary_finished_run():
    eng, mgr, _ = build()
    async with eng:
        chat = await mgr.create(user="ann")
        await chat.send("hello")
        assert [s.id for s in await mgr.list(user="ann")] == [chat.id]
        await chat.close()
        with pytest.raises(SessionClosed):
            await chat.send("anyone there?")
        out = await eng.handle(chat.id).result()
        assert out.kind is OutcomeKind.SUCCESS and out.value == {"turns": 1}
        assert await mgr.list(user="ann") == []
        assert [s.id for s in await mgr.list(user="ann", open_only=False)] == [chat.id]
        assert len(await (await mgr.get(chat.id)).history()) == 1  # still readable


async def test_a_turn_survives_the_process_that_sent_it():
    eng, mgr, _ = build()
    async with eng:
        chat = await mgr.create()
        n, _ = await chat._submit("hello")  # "the sender died right after sending"
        other = await mgr.get(chat.id)  # any other process
        result = await other.wait(n)
        assert result.ok and result.text.endswith("last: hello")


async def test_a_session_resumes_after_the_engine_restarts():
    store = InMemoryEventStore()
    eng1, mgr1, _ = build(store=store)
    async with eng1:
        chat = await mgr1.create(tenant="acme", user="ann")
        await chat.send("remember the number 7")
        sid = chat.id
    eng2, mgr2, _ = build(store=store)  # a new process on the same store
    async with eng2:
        chat2 = await mgr2.get(sid)
        r = await chat2.send("what was the number?")
        assert r.text == "seen 2 user and 1 assistant; last: what was the number?"


async def test_tenants_are_isolated_and_a_tenants_sessions_can_be_erased():
    eng, mgr, _ = build()
    async with eng:
        acme, globex = mgr.tenant("acme"), mgr.tenant("globex")
        a = await acme.create(user="ann", session_id="support")
        g = await globex.create(user="ann", session_id="support")  # the same id, no clash
        assert a.id == "acme/support" and g.id == "globex/support"
        await a.send("acme secret plan")
        with pytest.raises(KeyError):
            await acme.get("globex/support")
        assert [s.id for s in await acme.list()] == ["acme/support"]
        await a.close()
        report = await eng.tenant("acme").erase_all()
        assert "acme/support" in report.streams
        assert "secret plan" not in str([e.data for e in await eng.store.read("acme/support")])
        assert any("acme/support:t0" == s for s in report.streams), (
            "the turn runs are erased with it"
        )
        assert (await globex.get("support")) is not None  # the other tenant is untouched


async def test_a_whole_conversation_budget_ends_the_session():
    def costly(req):
        return LLMResponse(
            message=Message(role="assistant", content="ok"),
            usage=Usage(total_tokens=10),
            cost=0.4,
            model="fake",
        )

    eng, mgr, _ = build(costly, session_budget=Budget(max_cost=0.7))
    async with eng:
        chat = await mgr.create()
        assert (await chat.send("one")).ok
        second = await chat.send("two")  # this turn pushes the conversation over budget
        assert second.text == "ok", "the answer that crossed the line is still delivered"
        with pytest.raises(SessionClosed):
            await chat.send("three")
        out = await eng.handle(chat.id).result()
        assert out.kind is OutcomeKind.BUDGET_EXCEEDED


async def test_replies_can_be_streamed_as_they_are_written():
    eng, mgr, _ = build(algo=ToolLoop(ToolRegistry([]), stream=True))
    async with eng:
        chat = await mgr.create()
        s = await chat.stream("hello there")
        pieces = [p async for p in s]
        final = await s.result()
        assert len(pieces) > 1 and "".join(pieces).strip() == final.text.strip()


async def test_with_a_stateful_provider_later_turns_continue_instead_of_resending_history():
    client = FakeStatefulClient(echo)
    eng, mgr, _ = build(client=client)
    async with eng:
        chat = await mgr.create()
        for i in range(4):
            r = await chat.send(f"message {i}")
            assert r.text == f"seen {i + 1} user and {i} assistant; last: message {i}", (
                "the provider sees it all"
            )
    (_, first_cont), *later = client.sent
    assert first_cont is None
    assert all(cont is not None for _, cont in later)
    assert all(n <= 2 for n, _ in later), (
        f"a later turn sends the system prompt and the new message, not history: {client.sent}"
    )


async def test_if_the_provider_forgets_a_session_the_whole_history_is_resent():
    client = FakeStatefulClient(echo, lose_state_after=2)
    eng, mgr, _ = build(client=client)
    async with eng:
        chat = await mgr.create()
        texts = [(await chat.send(f"m{i}")).text for i in range(4)]
    assert texts[3] == "seen 4 user and 3 assistant; last: m3", "nothing was lost"
    assert [c for _, c in client.sent].count(None) >= 2


async def test_a_long_history_is_summarized_once_and_the_summary_is_reused():
    summaries: list[str] = []

    async def summarize(chunk: str, previous: str | None) -> str:
        summaries.append(chunk)
        return "EARLIER: " + chunk[:40].replace("\n", " ")

    policy = ContextPolicy(
        max_input_tokens=900,
        reserve_output=0,
        strategies=[Summarize(summarize=summarize), DropOldest()],
    )
    eng, mgr, client = build(lambda req: text("a long and rambling answer " * 20), context=policy)
    async with eng:
        chat = await mgr.create()
        for i in range(10):
            assert (await chat.send(f"question number {i} " + "details " * 30)).ok
        log = await eng.store.read(chat.id)
    made = [e for e in log if e.type == "aaa.session.summary"]
    assert made and len(summaries) == len(made), "each distinct prefix was summarized exactly once"
    last = client.requests[-1].messages
    assert any((m.content or "").startswith("[Summary of the earlier conversation]") for m in last)
    from aaa.context import HeuristicCounter

    assert max(HeuristicCounter().count(r.messages) for r in client.requests) < 2200


async def test_a_session_works_across_processes_a_client_submits_a_worker_answers(cluster):
    """The client process only holds a ClientBus (it hosts nothing); the turn is thought through by the worker,
    and the reply is recorded there. Runs on memory, Postgres and Redis."""
    from aaa.events import ClientBus, DistributedBus
    from cluster import FAST

    worker_node, client_node = await cluster.node(), await cluster.node()
    worker = Engine(
        store=worker_node.store, bus=DistributedBus(worker_node.backend, member="W", **FAST)
    )
    LLMHandler(FakeLLMClient(echo)).attach(worker)
    SessionManager(
        worker, ToolLoop(ToolRegistry([])), system="Be brief."
    )  # registers the algorithms
    client = Engine(store=client_node.store, bus=ClientBus())
    client_mgr = SessionManager(client, ToolLoop(ToolRegistry([])), system="Be brief.")
    async with worker, client:
        chat = await client_mgr.tenant("acme").create(user="ann")
        first = await asyncio.wait_for(chat.send("hello"), 30)
        second = await asyncio.wait_for(chat.send("again"), 30)
        assert second.text == "seen 2 user and 1 assistant; last: again" and first.ok
        # a second client process reattaches by id and sees the same conversation
        again = await client_mgr.tenant("acme").get(chat.id)
        assert [t.user for t in await again.history()] == ["hello", "again"]
