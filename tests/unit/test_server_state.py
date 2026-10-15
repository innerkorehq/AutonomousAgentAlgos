"""Provider-side conversation state: the handler continues from the provider's own handle instead of resending
the conversation, and degrades safely (restart, expiry, context compaction, erasure)."""

from aaa.algos import ToolLoop
from aaa.context import ContextPolicy
from aaa.llm import FakeLLMClient, FakeStatefulClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool

TURNS = 6


@tool
def lookup(q: str) -> str:
    """look something up"""
    return f"answer for {q} " + "z" * 300


def script(turns=TURNS, seen=None):
    calls = {"n": 0}  # counted here, not from the messages: compaction legitimately hides old ones

    def fn(req):
        if seen is not None:
            seen.append(req)
        calls["n"] += 1
        return (
            tool_call("lookup", q=f"step {calls['n']}") if calls["n"] <= turns else text("finished")
        )

    return fn


async def run(client, *, context=None, hook=None, input=None):
    eng = Engine()
    handler = LLMHandler(client, context=context)
    handler.attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg, max_turns=30)
    eng.register(algo)
    if hook:
        hook(handler)
    async with eng:
        h = await eng.start(algo, input or {"task": "work through it"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
        return out, log, handler, eng


async def test_later_turns_send_only_what_is_new_and_the_provider_sees_everything():
    seen: list = []
    client = FakeStatefulClient(script(seen=seen))
    out, _, _, _ = await run(client)
    assert out.ok and out.value["text"] == "finished"
    sizes = [n for n, _ in client.sent]
    assert client.sent[0][1] is None, (
        "the first request carries the whole (one-message) conversation"
    )
    assert all(c is not None for _, c in client.sent[1:]), (
        "every later request continues provider state"
    )
    assert max(sizes[1:]) <= 1, (
        f"a continuation carries just the new tool result, not the history: {sizes}"
    )
    assert len(seen[-1].messages) > 2 * TURNS, "...yet the provider-side conversation is complete"


async def test_the_log_records_the_handle_and_how_each_request_was_sent():
    client = FakeStatefulClient(script())
    _, log, _, _ = await run(client)
    done = [e for e in log if e.type == "aaa.llm.completed"]
    assert all(e.data["state_id"].startswith("state_") for e in done)
    assert [e.data["state"] for e in done] == ["fresh"] + ["continued"] * (len(done) - 1)
    first_calls = done[0].data["response"]["message"]["tool_calls"]
    assert first_calls[0]["provider_id"].startswith("prov_") and first_calls[0]["id"].startswith(
        "call_"
    )


async def test_ordinary_clients_are_unchanged_and_leave_no_new_keys_in_events():
    out, log, _, _ = await run(FakeLLMClient(script()))
    assert out.ok
    for e in log:
        if e.type == "aaa.llm.completed":
            assert (
                "state_id" not in e.data
                and "state" not in e.data
                and "state_id" not in e.data["response"]
            )
            assert all("provider_id" not in c for c in e.data["response"]["message"]["tool_calls"])
    requests = [e for e in log if e.type == "aaa.llm.requested"]
    assert (
        "resume" in requests[0].data["request"]
        and requests[0].data["request"]["continue_from"] is None
    )


async def test_a_restart_in_the_middle_still_continues_because_the_handle_is_in_the_log():
    client = FakeStatefulClient(script())

    def hook(handler):
        calls = {"n": 0}
        orig = client.complete

        async def spying(request, /):
            calls["n"] += 1
            if (
                calls["n"] == 4
            ):  # "the process restarts": forget the in-memory conversation and handles
                handler._convs.clear()
                handler._states.clear()
                handler._remember = lambda *a, **k: None
            return await orig(request)

        client.complete = spying  # type: ignore[method-assign]

    out, _, handler, _ = await run(client, hook=hook)
    assert out.ok and handler.rebuilds >= 1
    assert all(c is not None for _, c in client.sent[1:]), (
        "the handle was recovered from the log, not lost"
    )


async def test_when_the_provider_loses_the_state_the_full_conversation_is_sent_instead():
    client = FakeStatefulClient(script(), lose_state_after=3)
    out, log, _, _ = await run(client)
    assert out.ok and out.value["text"] == "finished"
    modes = [e.data["state"] for e in log if e.type == "aaa.llm.completed"]
    assert "fallback" in modes
    assert [n for n, c in client.sent if c is None][1:], "a full request followed the expiry"


async def test_a_conversation_the_context_policy_rewrites_is_sent_fresh_not_continued():
    client = FakeStatefulClient(script(turns=14))
    policy = ContextPolicy(max_input_tokens=1200, reserve_output=0)
    out, log, _, _ = await run(client, context=policy)
    assert out.ok
    reports = [e for e in log if e.type == "aaa.llm.completed" and e.data.get("context")]
    assert reports, "the policy did compact"
    for e in reports:
        assert e.data["state"] in ("fresh", "fallback"), (
            "never continue a chain the policy had to rewrite"
        )
    assert any(c is not None for _, c in client.sent), (
        "and before it was needed, continuation was used"
    )


async def test_a_new_run_can_resume_provider_state_from_an_earlier_one():
    client = FakeStatefulClient(
        lambda req: text(f"reply {sum(1 for m in req.messages if m.role == 'user')}")
    )
    out1, log1, _, _ = await run(
        client,
        input={
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "first"},
            ]
        },
    )
    state = next(e.data["state_id"] for e in reversed(log1) if e.type == "aaa.llm.completed")
    sent_before = len(client.sent)
    history = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": out1.value["text"]},
        {"role": "user", "content": "second"},
    ]
    out2, _, _, _ = await run(
        client, input={"messages": history, "provider_state": {"state_id": state, "covered": 2}}
    )
    assert out2.value["text"] == "reply 2", "the provider still saw both user turns"
    n, cont = client.sent[sent_before]
    assert cont == state and n == 2, (
        f"only the system prompt and the new user message were sent ({n} messages)"
    )


async def test_erasing_a_run_deletes_the_conversation_the_provider_holds():
    client = FakeStatefulClient(script(turns=2))
    _, log, _, eng = await run(client)
    ids = [e.data["state_id"] for e in log if e.type == "aaa.llm.completed"]
    assert ids and all(i in client.states for i in ids)
    async with eng:
        report = await eng.erase(log[0].run_id)
    assert report.provider_state == len(ids)
    assert not any(i in client.states for i in ids) and set(client.deleted) == set(ids)
