"""Memory through the Engine: recall in the prompt, scoping, degradation, write-back, tools, crash
safety, erasure, redaction and spend accounting."""

from aaa.algos import ToolLoop
from aaa.core import OutcomeKind
from aaa.events import Event
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.memory import (
    AddResult,
    CallableMemory,
    FakeMemory,
    MemoryEntry,
    MemoryHandler,
    Scope,
    WriteBack,
    memory_tools,
)
from aaa.memory.handler import STARTED
from aaa.privacy import standard_redactor
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry

ANN = Scope(tenant="acme", user="ann")


def build(memory, *, recall=True, write_back=None, tools=None, script=None, **engine_kw):
    seen: list = []  # every LLM request the model was given

    def default(req):
        seen.append(req)
        return text("done")

    def scripted(req):
        seen.append(req)
        return script(req)

    eng = Engine(**engine_kw)
    LLMHandler(FakeLLMClient(script and scripted or default)).attach(eng)
    mh = MemoryHandler(memory, write_back=write_back)
    mh.attach(eng)
    reg = ToolRegistry(tools or [])
    if tools:
        ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg, recall=recall or None)
    eng.register(algo)
    return eng, algo, seen, mh


def system_text(req) -> str:
    return "\n".join(m.content or "" for m in req.messages if m.role == "system")


async def test_recalled_memories_reach_the_prompt_and_are_recorded_in_the_log():
    mem = FakeMemory()
    await mem.add([MemoryEntry(text="Ann is vegetarian")], ANN)
    eng, algo, seen, _ = build(mem)
    async with eng:
        h = await eng.tenant("acme").start(algo, {"task": "what food does ann like"}, user="ann")
        assert (await h.result()).ok
        log = await eng.store.read(h.run_id)
    assert "Ann is vegetarian" in system_text(seen[0])
    types = [e.type for e in log]
    assert types.index("aaa.memory.requested") < types.index("aaa.memory.completed")
    assert types.index("aaa.memory.completed") < types.index("aaa.llm.requested")
    done = next(e for e in log if e.type == "aaa.memory.completed")
    assert [i["text"] for i in done.data["items"]] == [
        "Ann is vegetarian"
    ]  # replay sees exactly this
    assert done.tenant == "acme" and done.id.endswith(":result")


async def test_one_customers_memories_never_reach_another_customers_prompt():
    mem = FakeMemory()
    await mem.add([MemoryEntry(text="acme secret roadmap food")], ANN)
    eng, algo, seen, _ = build(mem)
    async with eng:
        for tenant, user in (("globex", "ann"), ("acme", "bob")):
            h = await eng.tenant(tenant).start(algo, {"task": "roadmap food"}, user=user)
            await h.result()
    assert all("secret roadmap" not in system_text(r) for r in seen)


async def test_a_memory_backend_that_fails_never_fails_the_run():
    class Broken(FakeMemory):
        async def search(self, *a, **kw):
            raise ConnectionError("memory service is down")

    eng, algo, seen, _ = build(Broken())
    async with eng:
        h = await eng.tenant("acme").start(algo, {"task": "hello"}, user="ann")
        out = await h.result()
        log = await eng.store.read(h.run_id)
    assert out.ok and len(seen) == 1
    failed = next(e for e in log if e.type == "aaa.memory.failed")
    assert failed.data["tag"] == "recall"
    assert "ConnectionError: memory service is down" in failed.data["error"]["message"]


async def test_recall_without_an_attached_handler_is_skipped_not_hung():
    eng = Engine()
    LLMHandler(FakeLLMClient(lambda r: text("ok"))).attach(eng)
    algo = ToolLoop(ToolRegistry([]), recall=True)
    eng.register(algo)
    async with eng:
        out = await eng.run(algo, {"task": "x"})
    assert out.ok


async def test_write_back_remembers_the_task_and_answer_once():
    mem = FakeMemory()
    eng, algo, _, _ = build(mem, recall=False, write_back=WriteBack())
    async with eng:
        h = await eng.tenant("acme").start(
            algo, {"task": "my favourite colour is teal"}, user="ann"
        )
        await h.result()
        await _settle(eng, h.run_id, "aaa.memory.completed")
        log = await eng.store.read(h.run_id)
    texts = [m.text for m in await mem.list(ANN)]
    assert texts == ["my favourite colour is teal", "done"]
    wb = [e for e in log if e.type == "aaa.memory.completed" and e.data["tag"] == "writeback"]
    assert len(wb) == 1 and wb[0].id == f"{h.run_id}:writeback:result"


async def test_write_back_skips_failed_runs():
    mem = FakeMemory()
    eng, algo, _, _ = build(mem, recall=False, write_back=WriteBack())
    async with eng:
        h = await eng.tenant("acme").start(algo, {"task": "x"}, user="ann")
        await h.cancel()
        assert (await h.result()).kind is OutcomeKind.CANCELLED
    assert await mem.list(ANN) == []


async def test_the_model_manages_memory_with_tools_but_cannot_choose_whose():
    mem = FakeMemory()

    def script(req):
        last = req.messages[-1]
        if last.role == "user":
            return tool_call("remember", text="Ann prefers window seats")
        return text("noted")

    eng, algo, _, _ = build(mem, recall=False, tools=memory_tools(mem), script=script)
    async with eng:
        h = await eng.tenant("acme").start(algo, {"task": "remember my seat"}, user="ann")
        assert (await h.result()).ok
    assert [m.text for m in await mem.list(ANN)] == ["Ann prefers window seats"]
    assert await mem.list(Scope(tenant="globex", user="ann")) == []
    spec = {t.name: t for t in memory_tools(mem)}["remember"].parameters
    assert set(spec["properties"]) == {"text"}  # no scope, tenant or user argument to forge
    assert spec.get("additionalProperties") is False


async def test_recall_tool_reads_only_the_runs_own_scope():
    mem = FakeMemory()
    await mem.add([MemoryEntry(text="acme budget is large")], ANN)
    await mem.add([MemoryEntry(text="globex budget is tiny")], Scope(tenant="globex", user="ann"))

    def script(req):
        last = req.messages[-1]
        return tool_call("recall", query="budget") if last.role == "user" else text(last.content)

    eng, algo, _, _ = build(mem, recall=False, tools=memory_tools(mem), script=script)
    async with eng:
        out = await eng.tenant("acme").run(algo, {"task": "budget?"}, user="ann")
    assert "acme budget" in out.value["text"] and "globex" not in out.value["text"]


async def test_a_write_that_died_midway_is_reported_not_repeated_on_a_backend_without_dedupe():
    stored: list[str] = []
    mem = CallableMemory(
        add=lambda entries, scope: stored.extend(e.text for e in entries),
        search=lambda q, s, k: [],
    )  # idempotent=False: the safe default
    eng, _, _, mh = build(mem, recall=False)
    async with eng:
        run = await eng.tenant("acme").start(ToolLoop(ToolRegistry([])), {"task": "x"}, user="ann")
        await run.result()
        req = Event(
            id="req-1",
            type="aaa.memory.requested",
            stream=run.run_id,
            run_id=run.run_id,
            tenant="acme",
            meta={"user": "ann"},
            data={"op": "remember", "tag": "main", "entries": [{"text": "fact"}]},
        )
        # a previous worker recorded the attempt marker and then died
        await eng.store.append(run.run_id, [req.caused(STARTED, {}, id="req-1:attempt")])
        [reply] = await mh.handle(req)
    assert reply.type == "aaa.memory.failed" and reply.data["error"]["type"] == "OutcomeUnknown"
    assert stored == []  # not written a second time


async def test_a_backend_with_idempotent_writes_is_retried_safely():
    mem = FakeMemory()  # idempotent=True
    eng, _, _, mh = build(mem, recall=False)
    assert mh.delivery == "at_least_once"
    async with eng:
        run = await eng.tenant("acme").start(ToolLoop(ToolRegistry([])), {"task": "x"}, user="ann")
        await run.result()
        req = Event(
            id="req-2",
            type="aaa.memory.requested",
            stream=run.run_id,
            run_id=run.run_id,
            tenant="acme",
            meta={"user": "ann"},
            data={"op": "remember", "tag": "main", "entries": [{"text": "same fact"}]},
        )
        await mh.handle(req)
        await mh.handle(req)  # redelivery
    assert len(await mem.list(ANN)) == 1


async def test_the_redactor_scrubs_what_reaches_the_memory_backend():
    mem = FakeMemory()
    eng, algo, _, _ = build(mem, redactor=standard_redactor())
    async with eng:
        await eng.tenant("acme").run(
            algo, {"task": "email me at ann@example.com about teal"}, user="ann"
        )
    searched = [q for op, q in mem.calls if op == "search"]
    assert searched and "example.com" not in searched[0] and "[REDACTED:email]" in searched[0]


async def test_tenant_erasure_reaches_the_memory_backends():
    mem, other = FakeMemory(), FakeMemory()
    await mem.add([MemoryEntry(text="ann fact")], ANN)
    await mem.add([MemoryEntry(text="bob fact")], Scope(tenant="acme", user="bob"))
    await mem.add([MemoryEntry(text="globex fact")], Scope(tenant="globex"))
    no_forget = CallableMemory(add=lambda e, s: None, search=lambda q, s, k: [])
    eng = Engine(memory=[mem, no_forget])
    eng.memories.append(other)
    async with eng:
        scope = eng.tenant("acme")
        one = await scope.forget_memory(user="ann")
        assert one.forgotten["FakeMemory"] == 1 and one.unsupported == ["CallableMemory"]
        assert [m.text for m in await mem.list(Scope(tenant="acme"))] == ["bob fact"]
        report = await scope.erase_all()
    assert report.memory is not None and report.memory.forgotten["FakeMemory"] == 1
    assert [m.text for m in await mem.list(Scope(tenant="globex"))] == ["globex fact"]  # untouched


async def test_memory_spend_is_counted_in_the_run_ledger():
    class Spendy(FakeMemory):
        async def add(self, entries, scope, *, key=None):
            r = await super().add(entries, scope, key=key)
            return AddResult(ids=r.ids, usage={"total_tokens": 120}, cost=0.002)

    eng, algo, _, _ = build(Spendy(), recall=False, write_back=WriteBack())
    async with eng:
        h = await eng.tenant("acme").start(algo, {"task": "remember teal"}, user="ann")
        await h.result()
        await _settle(eng, h.run_id, "aaa.memory.completed")
        log = await eng.store.read(h.run_id)
    from aaa.runtime import Ledger

    ledger = Ledger()
    for e in log:
        ledger = ledger.fold(e)
    assert ledger.tokens >= 120 and ledger.cost >= 0.002


async def _settle(eng, run_id, type_):
    """Write-back happens just after the run completes: wait until its event is in the log."""
    import asyncio

    for _ in range(200):
        if any(
            e.type == type_ and e.data.get("tag") == "writeback"
            for e in await eng.store.read(run_id)
        ):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("write-back never completed")
