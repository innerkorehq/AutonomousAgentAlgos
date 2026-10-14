"""Long conversations must cost O(1) per turn in log size, state size and handler work."""

import itertools
import json

from aaa.algos import GenerateTestRepair, ToolLoop
from aaa.core import OutcomeKind
from aaa.data import InMemoryBlobStore
from aaa.events import Event, SQLiteEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool

BLOB = "y" * 300


@tool
def lookup(i: int) -> str:
    """small record"""
    return f"{i}:{BLOB}"


def norm(ids: dict[str, int], i):
    return None if i is None else ids.setdefault(i, len(ids))


def normalized(transcripts):
    """Tool-call ids embed the (random) run id; compare transcripts modulo id naming."""
    out = []
    for t in transcripts:
        ids: dict[str, int] = {}
        out.append([(r, c, [norm(ids, x) for x in cs], norm(ids, tid)) for r, c, cs, tid in t])
    return out


def long_script(turns, seen=None):
    n = {"calls": 0}

    def script(req):
        n["calls"] += 1
        if seen is not None:
            seen.append(
                [
                    (m.role, m.content, [c.id for c in m.tool_calls], m.tool_call_id)
                    for m in req.messages
                ]
            )
        return tool_call("lookup", i=n["calls"]) if n["calls"] <= turns else text("done")

    return script


async def run(turns, *, store=None, pre=None, engine_kw=None, system=""):
    """`pre(n, handler)` runs just before the handler resolves the n-th LLM request's transcript
    (fault injection point: wipe the cache, shrink it, ...)."""
    reg = ToolRegistry([lookup])
    seen: list = []
    handler = LLMHandler(FakeLLMClient(long_script(turns, seen)))
    if pre:
        orig, count = handler._transcript, {"n": 0}

        async def wrapped(e, req):
            count["n"] += 1
            pre(count["n"], handler)
            return await orig(e, req)

        handler._transcript = wrapped
    async with Engine(store=store, **(engine_kw or {})) as eng:
        handler.attach(eng)
        ToolHandler(reg).attach(eng)
        h = await eng.start(ToolLoop(reg, max_turns=turns + 5, system=system), {"task": "go"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    return out, log, seen, handler


async def test_request_events_stay_constant_size_as_the_conversation_grows():
    out, log, _, _ = await run(120)
    assert out.ok
    sizes = [len(e.model_dump_json()) for e in log if e.type == "aaa.llm.requested"]
    assert len(sizes) == 121
    early, late = sum(sizes[10:20]) / 10, sum(sizes[-10:]) / 10
    assert late < early * 1.2, (early, late)  # was ~12x bigger at turn 120 before
    assert all(e.data.get("conv") for e in log if e.type == "aaa.llm.requested" and e.seq > 1)


async def test_algorithm_state_snapshot_does_not_grow_with_turns(tmp_path):
    sizes = {}
    for turns in (40, 160):
        store = SQLiteEventStore(tmp_path / f"{turns}.db")
        out, log, _, _ = await run(turns, store=store)
        assert out.ok
        snap = await store.load_snapshot(log[0].stream, "aaa.loop")
        sizes[turns] = len(snap.state)
        await store.aclose()
    assert sizes[160] <= sizes[40] + 50 and sizes[160] < 1500, sizes


async def test_model_still_sees_the_exact_full_transcript_each_turn():
    out, _log, seen, _ = await run(4, system="be terse")
    assert out.ok and len(seen) == 5
    for k, transcript in enumerate(seen, start=1):
        roles = [r for r, *_ in transcript]
        assert roles == ["system", "user"] + ["assistant", "tool"] * (k - 1)
        calls = [
            (c, tid)
            for r, _, cs, tid in transcript
            if r in ("assistant", "tool")
            for c in (cs or [None])
        ]
        # every tool message answers the id of the assistant call right before it
        for (r1, _, cs, _), (r2, _, _, tid) in itertools.pairwise(transcript):
            if r1 == "assistant":
                assert r2 == "tool" and tid == cs[0]
    assert seen[-1][-1][1].startswith("4:") and calls


async def test_conversation_survives_cache_loss_identically():
    """A restart / failover / different worker = empty cache: the handler must rebuild from the log
    and the model must see byte-identical requests to an uninterrupted run."""
    _, _, baseline, _ = await run(9)

    def wipe(call, handler):
        if call in (3, 4, 8):
            handler._convs.clear()

    out, _, interrupted, handler = await run(9, pre=wipe)
    assert out.ok and normalized(interrupted) == normalized(baseline)
    assert handler.rebuilds == 3


async def test_cache_eviction_and_claim_checked_history_still_rebuild_correctly():
    _, _, baseline, _ = await run(7)
    out, log, seen, handler = await run(
        7,
        engine_kw={"blobs": InMemoryBlobStore(), "claim_threshold": 200},  # events live in blobs
        pre=lambda c, h: (
            setattr(h, "max_cached_conversations", 0),
            h._convs.clear(),
        ),  # never cached
    )
    assert out.ok and normalized(seen) == normalized(baseline)
    assert handler.rebuilds >= 6
    assert any(e.data.get("$claim") for e in log)  # the rebuild really crossed blobs


async def test_missing_history_fails_the_run_cleanly_not_silently():
    class Bogus:
        name = "bogus"

        def subscriptions(self):
            return ["aaa.run.started", "aaa.llm.failed"]

        async def on_event(self, e, ctx):
            if e.type == "aaa.llm.failed":
                return [
                    e.caused(
                        "aaa.run.completed",
                        {"kind": "failed", "reason": e.data["error"]["message"]},
                    )
                ]
            from aaa.llm import LLMRequest

            body = LLMRequest(messages=[]).model_dump(mode="json")
            return [
                e.caused(
                    "aaa.llm.requested",
                    {"request": body, "tag": "x", "conv": {"prev": "does-not-exist", "delta": []}},
                )
            ]

    async with Engine() as eng:
        LLMHandler(FakeLLMClient([])).attach(eng)
        out = await eng.run(Bogus(), {})
    assert out.kind is OutcomeKind.FAILED and "cannot rebuild conversation" in out.reason


async def test_repair_loop_continues_the_same_conversation_through_deltas():
    replies = iter(["nope", "still nope", '["a","b","c"]'])
    llm = FakeLLMClient(lambda r: text(next(replies)))

    def check(candidate, ctx):
        try:
            return len(json.loads(candidate)) == 3, "need a JSON array of 3 items"
        except ValueError:
            return False, "not JSON"

    from aaa.validate import ValidationHandler

    async with Engine() as eng:
        LLMHandler(llm).attach(eng)
        ValidationHandler({"default": check}).attach(eng)
        h = await eng.start(GenerateTestRepair(max_iterations=4), {"task": "three items"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    assert out.ok and out.value["iterations"] == 3
    last = [(m.role, m.content) for m in llm.requests[-1].messages]
    assert [r for r, _ in last] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]  # full history kept
    gens = [e for e in log if e.type == "aaa.llm.requested" and e.data["tag"] == "gen"]
    assert "conv" not in gens[0].data and gens[1].data["conv"]["prev"] == gens[0].id


async def test_concurrent_conversations_do_not_cross_contaminate():
    import asyncio

    reg = ToolRegistry([lookup])
    seen: dict[str, list] = {}

    def script(req):
        task = next(m.content for m in req.messages if m.role == "user")
        seen.setdefault(task, []).append(len(req.messages))
        n = len(seen[task])
        return tool_call("lookup", i=n) if n <= 5 else text(f"done {task}")

    async with Engine() as eng:
        handler = LLMHandler(FakeLLMClient(script))
        handler.max_cached_conversations = 3  # fewer slots than live conversations
        handler.attach(eng)
        ToolHandler(reg).attach(eng)
        algo = ToolLoop(reg, max_turns=20)
        outs = await asyncio.gather(*(eng.run(algo, {"task": f"job-{i}"}) for i in range(8)))
    assert all(o.ok for o in outs) and [o.value["text"] for o in outs] == [
        f"done job-{i}" for i in range(8)
    ]
    assert all(
        v == [1, 3, 5, 7, 9, 11] for v in seen.values()
    )  # each saw its own growing transcript
    assert handler.rebuilds > 0  # eviction was actually exercised


async def test_tool_handler_idempotency_check_is_constant_time():
    """A redelivered request must not re-run the tool, and the check must not scan the stream."""
    reads = {"n": 0}
    calls = {"n": 0}

    @tool
    def once() -> str:
        """count"""
        calls["n"] += 1
        return "ok"

    reg = ToolRegistry([once])
    async with Engine() as eng:
        orig = eng.store.read

        async def counting_read(*a, **k):
            reads["n"] += 1
            return await orig(*a, **k)

        eng.store.read = counting_read
        handler = ToolHandler(reg)
        handler.attach(eng)
        req = Event(
            type="aaa.tool.requested",
            stream="s",
            run_id="r",
            data={"call_id": "c1", "name": "once", "arguments": {}},
        )
        await eng.store.append("s", [req])
        await handler._on_requested(req)
        await eng.bus.drain()
        import asyncio

        await asyncio.sleep(0.05)
        before = reads["n"]
        handler._started.clear()  # simulate a redelivery after restart
        await handler._on_requested(req)
        await asyncio.sleep(0.05)
    assert calls["n"] == 1 and reads["n"] == before  # skipped, and without reading the log
