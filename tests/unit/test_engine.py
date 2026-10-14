import asyncio

from aaa.core import OutcomeKind
from aaa.events import Event
from aaa.events.cloudevents import from_cloudevent, to_cloudevent
from aaa.runtime import Budget, Ctx, Engine


class Counter:
    """Counts tool calls via events until `target`, then completes. Holds no run state."""

    name = "counter"

    def __init__(self, target: int, boom: bool = False):
        self.target, self.boom = target, boom

    def subscriptions(self):
        return ["aaa.run.started", "aaa.tool.completed", "user.approval"]

    async def on_event(self, e: Event, ctx: Ctx):
        if self.boom:
            raise ValueError("kaput")
        if e.type == "user.approval":
            return [e.caused("aaa.run.completed", {"kind": "success", "value": "approved"})]
        n = sum(1 for x in await ctx.history() if x.type == "aaa.tool.completed")
        if n >= self.target:
            return [e.caused("aaa.run.completed", {"kind": "success", "value": n})]
        return [e.caused("aaa.tool.completed", {"n": n})]


async def test_run_success_and_log_is_source_of_truth():
    async with Engine() as eng:
        h = await eng.start(Counter(3), {})
        out = await h.result()
        assert out.ok and out.value == 3
        types = [e.type for e in await eng.store.read(h.run_id)]
        assert types[0] == "aaa.run.started" and types[-1] == "aaa.run.completed"
        assert types.count("aaa.tool.completed") == 3
        assert [e.type async for e in h.events()] == types


async def test_budget_exceeded():
    async with Engine() as eng:
        out = await eng.run(Counter(100), {}, budget=Budget(max_steps=5))
        assert out.kind is OutcomeKind.BUDGET_EXCEEDED


async def test_timeout_and_cancel_and_failure():
    class Idle(Counter):
        async def on_event(self, e, ctx):
            return []

    async with Engine() as eng:
        assert (
            await eng.run(Idle(1), {}, budget=Budget(deadline_s=0.05))
        ).kind is OutcomeKind.TIMEOUT
        h = await eng.start(Idle(1), {})
        await h.cancel("user")
        assert (await h.result()).kind is OutcomeKind.CANCELLED
        out = await eng.run(Counter(1, boom=True), {})
        assert out.kind is OutcomeKind.FAILED and "kaput" in out.reason


async def test_external_signal_and_no_events_after_done():
    class Wait(Counter):
        async def on_event(self, e, ctx):
            return [] if e.type == "aaa.run.started" else await super().on_event(e, ctx)

    async with Engine() as eng:
        h = await eng.start(Wait(1), {})
        await asyncio.sleep(0.01)
        await h.signal("user.approval")
        assert (await h.result()).value == "approved"
        await h.signal("user.approval")  # late: run already done, handler not invoked
        await eng.bus.drain()
        completed = [e for e in await eng.store.read(h.run_id) if e.type == "aaa.run.completed"]
        assert len(completed) == 1


def test_cloudevents_roundtrip():
    e = Event(
        type="user.x",
        subject="s",
        run_id="r",
        trace_id="t",
        causation_id="c",
        schema_ref="user.x/v1",
        data={"a": 1},
        meta={"tenant": "acme", "w": 0.5},
        seq=3,
    )
    ce = to_cloudevent(e)
    assert ce["specversion"] == "1.0" and ce["runid"] == "r" and ce["seq"] == 3
    assert from_cloudevent(ce) == e
    assert from_cloudevent(__import__("json").dumps(ce)) == e
