"""A temporary event-store outage must never fail a run or lose a reaction.

Outages are scripted with a proxy store that raises ConnectionError while `down` is set (plus real
connection kills against Postgres/Redis in the cluster tests below)."""

import asyncio
import logging

import pytest

from aaa.algos import ToolLoop
from aaa.core import (
    ConcurrencyError,
    OutcomeKind,
    Redeliver,
    RetryPolicy,
    StoreUnavailable,
    is_transient,
    retrying,
)
from aaa.events import Event, InMemoryEventStore, InProcessBus
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


class FlakyStore:
    """Delegates to a real store; while `down`, every operation raises ConnectionError."""

    def __init__(self, inner, fail_types=None):
        self._inner, self.down, self.fail_types = inner, False, fail_types

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _check(self, events=()):
        if self.down and (
            self.fail_types is None or any(e.type in self.fail_types for e in events)
        ):
            raise ConnectionError("db down")

    async def append(self, stream, events, expected_seq=-2):
        self._check(events)
        return await self._inner.append(stream, events, expected_seq)

    async def read(self, *a, **k):
        self._check()
        return await self._inner.read(*a, **k)

    async def last_seq(self, *a):
        self._check()
        return await self._inner.last_seq(*a)

    async def has(self, *a):
        self._check()
        return await self._inner.has(*a)

    async def run_index(self):
        self._check()
        return await self._inner.run_index()

    async def load_snapshot(self, *a):
        self._check()
        return await self._inner.load_snapshot(*a)

    async def save_snapshot(self, *a):
        self._check()
        return await self._inner.save_snapshot(*a)


async def outage(store, start_s, length_s):
    await asyncio.sleep(start_s)
    store.down = True
    await asyncio.sleep(length_s)
    store.down = False


# ----------------------------------------------------------------------------------- the primitives
def test_retry_policy_backoff_is_bounded_and_jitter_stays_within_it():
    p = RetryPolicy(jitter=False, base_delay_s=0.1, max_delay_s=1.0)
    assert [round(p.delay(i), 2) for i in range(1, 6)] == [0.1, 0.2, 0.4, 0.8, 1.0]
    j = RetryPolicy(base_delay_s=0.1, max_delay_s=1.0)
    assert all(0 <= j.delay(7) <= 1.0 for _ in range(50))
    with pytest.raises(Exception):  # noqa: B017 - frozen model
        p.max_attempts = 3


def test_transient_classification_matches_real_driver_errors_and_spares_logic_errors():
    import asyncpg
    import redis.exceptions as r

    assert all(
        map(
            is_transient,
            [
                ConnectionResetError(),
                TimeoutError(),
                StoreUnavailable("x"),
                asyncpg.exceptions.InternalClientError("killed"),  # what Postgres really raised
                asyncpg.exceptions.ConnectionDoesNotExistError("x"),
                r.ConnectionError("x"),
            ],
        )
    )
    assert not any(
        map(
            is_transient,
            [
                ValueError("bug"),
                ConcurrencyError("x"),
                FileNotFoundError("blob"),
                asyncpg.exceptions.UniqueViolationError("x"),
                r.ResponseError("x"),
                KeyError("k"),
            ],
        )
    )


async def test_retrying_succeeds_after_flaky_calls_and_gives_up_with_store_unavailable():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise ConnectionError("blip")
        return "ok"

    fast = RetryPolicy(base_delay_s=0.001, max_delay_s=0.002)
    assert await retrying(flaky, fast) == "ok" and calls["n"] == 4

    async def dead():
        raise ConnectionError("down")

    with pytest.raises(StoreUnavailable) as e:
        await retrying(dead, RetryPolicy(max_attempts=3, base_delay_s=0.001))
    assert isinstance(e.value.__cause__, ConnectionError)

    async def bug():
        raise ValueError("not an outage")

    with pytest.raises(ValueError):  # logic errors are never retried
        await retrying(bug, fast)


async def test_in_process_bus_redelivers_in_order_and_does_not_drop_the_event():
    seen: list[tuple[str, int]] = []
    fails = {"n": 0}

    async def h(e):
        if e.data["i"] == 1 and fails["n"] < 2:
            fails["n"] += 1
            raise Redeliver("store down")
        seen.append((e.type, e.data["i"]))

    async with InProcessBus(redeliver_base_s=0.01, redeliver_max_s=0.02) as bus:
        bus.subscribe("t", h)
        for i in range(3):
            await bus.publish(Event(type="t", data={"i": i}))
        await bus.drain()
    assert (
        seen == [("t", 0), ("t", 1), ("t", 2)] and fails["n"] == 2
    )  # retried twice, order preserved


# ----------------------------------------------------------------------------------- the original repro
@tool
async def slow() -> str:
    """takes a moment"""
    await asyncio.sleep(0.3)
    RAN.append(1)
    return "ok"


RAN: list[int] = []


def model(req):
    return tool_call("slow") if req.messages[-1].role == "user" else text("done")


async def boot(store, **engine_kw):
    eng = Engine(
        store=store, bus=InProcessBus(redeliver_base_s=0.05, redeliver_max_s=0.1), **engine_kw
    )
    await eng.__aenter__()
    llm = FakeLLMClient(model)
    LLMHandler(llm).attach(eng)
    reg = ToolRegistry([slow])
    ToolHandler(reg).attach(eng)
    return eng, reg, llm


async def test_a_brief_outage_while_a_tool_runs_does_not_strand_the_run():
    """The reported bug: result append fails during an outage -> run stuck at tool.started until its deadline."""
    RAN.clear()
    store = FlakyStore(InMemoryEventStore())
    eng, reg, _ = await boot(store)
    asyncio.create_task(outage(store, 0.15, 0.5))  # down from t=0.15s to t=0.65s; tool ends ~t=0.3s
    run = await eng.start(ToolLoop(reg), {"task": "go"}, budget=Budget(deadline_s=8))
    out = await asyncio.wait_for(run.result(), 10)
    log = await store.read(run.run_id)
    await eng.__aexit__(None, None, None)
    types = [e.type for e in log]
    assert out.ok and out.value["text"] == "done"  # NOT a timeout
    assert types.count("aaa.tool.completed") == 1 and RAN == [
        1
    ]  # the tool was not re-run to get its result recorded


async def test_outage_longer_than_the_retry_budget_redelivers_instead_of_failing_the_run():
    RAN.clear()
    store = FlakyStore(InMemoryEventStore())
    eng, reg, _ = await boot(
        store, store_retry=RetryPolicy(max_elapsed_s=0.15, base_delay_s=0.02, max_delay_s=0.05)
    )
    run = await eng.start(ToolLoop(reg), {"task": "go"}, budget=Budget(deadline_s=10))
    store.down = True  # right as the algorithm starts reacting
    await asyncio.sleep(1.2)  # >> the 0.15s local retry budget
    store.down = False
    out = await asyncio.wait_for(run.result(), 10)
    await eng.__aexit__(None, None, None)
    assert out.ok and out.value["text"] == "done"  # no failed run, nothing lost


async def test_model_is_not_called_again_when_only_the_result_write_hit_the_outage():
    RAN.clear()
    store = FlakyStore(InMemoryEventStore())
    eng, reg, llm = await boot(store)  # default budget: rides out a 0.5s outage locally
    asyncio.create_task(outage(store, 0.01, 0.5))
    run = await eng.start(ToolLoop(reg), {"task": "go"}, budget=Budget(deadline_s=10))
    out = await asyncio.wait_for(run.result(), 15)
    await eng.__aexit__(None, None, None)
    assert out.ok and len(llm.requests) == 2  # one call per turn: no paid repeats


# ----------------------------------------------------------------------------------- bugs still fail the run
class Buggy:
    name = "buggy"

    def subscriptions(self):
        return ["aaa.run.started"]

    async def on_event(self, e, ctx):
        raise ValueError("a genuine bug in the algorithm")


async def test_a_genuinely_buggy_algorithm_still_fails_the_run(caplog):
    caplog.set_level(logging.CRITICAL)
    async with Engine(store=FlakyStore(InMemoryEventStore())) as eng:
        out = await asyncio.wait_for(eng.run(Buggy(), {}), 5)
    assert out.kind is OutcomeKind.FAILED and "genuine bug" in out.reason


# ----------------------------------------------------------------------------------- cleanup on failure
async def test_a_failed_start_leaves_no_phantom_run_and_can_be_retried():
    store = FlakyStore(InMemoryEventStore())
    eng, reg, _ = await boot(store, store_retry=RetryPolicy(max_attempts=2, base_delay_s=0.01))
    store.down = True
    with pytest.raises(StoreUnavailable):
        await eng.start(ToolLoop(reg), {"task": "go"}, run_id="job-1")
    assert "job-1" not in eng._runs  # forgotten: nothing durable exists
    store.down = False
    out = await asyncio.wait_for(eng.run(ToolLoop(reg), {"task": "go"}, run_id="job-1"), 10)
    await eng.__aexit__(None, None, None)
    assert out.ok


async def test_a_completion_that_could_not_be_written_can_be_retried():
    store = FlakyStore(InMemoryEventStore())
    eng, reg, _ = await boot(store, store_retry=RetryPolicy(max_attempts=2, base_delay_s=0.01))
    run = await eng.start(ToolLoop(reg), {"task": "x"}, run_id="r", budget=Budget(deadline_s=30))
    await asyncio.wait_for(run.result(), 10)
    run2 = await eng.start(ToolLoop(reg), {"task": "y"}, run_id="r2", budget=Budget(deadline_s=30))
    await asyncio.wait_for(run2.result(), 10)
    r3 = eng._runs  # sanity: both completed normally
    assert r3["r"].done and r3["r2"].done

    # now: completing an OPEN run while the store is down must not mark it done locally
    class Idle:
        name = "idle"

        def subscriptions(self):
            return ["aaa.run.started"]

        async def on_event(self, e, ctx):
            return []

    h = await eng.start(Idle(), {}, run_id="open")
    store.down = True
    with pytest.raises(StoreUnavailable):
        await eng.complete("open", OutcomeKind.CANCELLED, reason="x")
    assert eng._runs["open"].done is False
    store.down = False
    await eng.complete("open", OutcomeKind.CANCELLED, reason="x")
    assert (await h.result()).kind is OutcomeKind.CANCELLED
    await eng.__aexit__(None, None, None)


async def test_losing_a_stream_delta_never_fails_the_model_call():
    store = FlakyStore(InMemoryEventStore(), fail_types={"aaa.llm.delta"})
    store.down = True  # deltas can never be written...
    eng = Engine(store=store)
    await eng.__aenter__()
    LLMHandler(FakeLLMClient(lambda r: text("one two three"))).attach(eng)
    run = await eng.start(ToolLoop(ToolRegistry(), stream=True), {"task": "hi"})
    out = await asyncio.wait_for(run.result(), 10)  # ...but the call and the run are fine
    await eng.__aexit__(None, None, None)
    assert out.ok and out.value["text"] == "one two three"


# ----------------------------------------------------------------------------------- acknowledgement hygiene
class _Spy:
    """Stands in for aaa.events.Deferral and records whether the bus message was acknowledged."""

    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


async def _launch_one(store, monkeypatch, **handler_kw):
    import aaa.tools.handler as th

    spy = _Spy()
    monkeypatch.setattr(th, "defer_ack", lambda: spy)
    eng = Engine(store=store, store_retry=RetryPolicy(max_attempts=2, base_delay_s=0.01))
    await eng.__aenter__()
    reg = ToolRegistry([slow])
    handler = ToolHandler(reg, **handler_kw)
    handler.attach(eng)
    run = await eng.start(ToolLoop(reg), {"task": "go"}, budget=Budget(deadline_s=30))
    req = Event(
        type="aaa.tool.requested",
        stream=run.run_id,
        run_id=run.run_id,
        data={"call_id": "c1", "name": "slow", "arguments": {}},
    )
    await eng.store.append(run.run_id, [req])
    return eng, handler, spy, req


async def test_a_call_cancelled_before_its_first_step_still_acknowledges_its_message(monkeypatch):
    """The race that stalled shutdown for 30s: the run ends right after the call was launched."""
    eng, handler, spy, req = await _launch_one(InMemoryEventStore(), monkeypatch)
    handler._launch(req, handler.registry.get("slow"))
    for task in list(eng._tasks):
        task.cancel()  # before the coroutine ever got to run its `finally`
    await asyncio.sleep(0.05)
    await eng.__aexit__(None, None, None)
    assert spy.released == 1 and ("c1" not in {k[1] for k in handler._started})


async def test_a_result_that_could_not_be_recorded_is_not_acknowledged(monkeypatch):
    RAN.clear()
    store = FlakyStore(InMemoryEventStore())
    eng, handler, spy, req = await _launch_one(
        store,
        monkeypatch,
        result_retry=RetryPolicy(max_elapsed_s=0.2, base_delay_s=0.02),  # bounded: gives up
    )
    store.fail_types = {
        "aaa.tool.completed",
        "aaa.tool.failed",
    }  # the tool runs, but its result can't be written
    store.down = True
    handler._launch(req, handler.registry.get("slow"))
    await asyncio.sleep(1.2)  # tool finishes (0.3s), retries exhaust
    assert RAN == [1] and spy.released == 0  # it ran once; the message stays unacknowledged
    store.down = False
    await eng.__aexit__(None, None, None)
