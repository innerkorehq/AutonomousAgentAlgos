"""Redaction, tenancy, erasure and retention, end to end through the Engine."""

import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aaa.algos import ToolLoop
from aaa.core import RunNotFinished
from aaa.data.blob import InMemoryBlobStore
from aaa.events import DistributedBus, Event, InMemoryEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.privacy import (
    ChainRedactor,
    KeyRedactor,
    PatternRedactor,
    standard_redactor,
)
from aaa.runtime import Engine, RetentionPolicy
from aaa.tools import ToolHandler, ToolRegistry, tool
from cluster import FAST

# ---- redactors ---------------------------------------------------------------------------------


def ev(data, type_="x.test", **kw):
    return Event(type=type_, data=data, **kw)


def test_patterns_redact_emails_cards_ssn_and_secrets_but_not_lookalikes():
    r = PatternRedactor()
    out = r.redact(
        ev(
            {
                "text": "mail ann@example.com, card 4111 1111 1111 1111, ssn 123-45-6789, "
                "key sk-abcdefghijklmnop1234 and Bearer abcdefghijklmnopqrstuv",
                "order": "1234 5678 9012 3456",  # 16 digits but fails the Luhn check: left alone
                "version": "2024-05-06",
            }
        )
    )
    t = out.data["text"]
    for needle in ("ann@example.com", "4111", "123-45-6789", "sk-abcdef", "abcdefghijklmnopqrstuv"):
        assert needle not in t
    assert t.count("[REDACTED:") == 5
    assert out.data["order"] == "1234 5678 9012 3456" and out.data["version"] == "2024-05-06"


def test_key_redactor_matches_whole_keys_so_accounting_fields_survive():
    r = KeyRedactor()
    out = r.redact(
        ev(
            {
                "usage": {"prompt_tokens": 7, "total_tokens": 9},
                "args": {
                    "Password": "hunter2",
                    "api-key": "abc",
                    "nested": {"authorization": "Bearer x", "token": "t", "tokens": 3},
                },
                "headers": [{"Set-Cookie": "sid=1"}],
            }
        )
    )
    assert out.data["usage"] == {"prompt_tokens": 7, "total_tokens": 9}
    a = out.data["args"]
    assert a["Password"] == a["api-key"] == "[REDACTED]"
    assert a["nested"] == {"authorization": "[REDACTED]", "token": "[REDACTED]", "tokens": 3}
    assert out.data["headers"] == [{"Set-Cookie": "[REDACTED]"}]


def test_structural_fields_are_never_rewritten():
    r = standard_redactor()
    e = ev(
        {
            "call_id": "call_4111111111111111",
            "name": "a@b.co",  # top-level `name` is the tool name: structural
            "arguments": {"name": "ann@example.com"},  # a user field called name: scrubbed
            "usage": {"cost": 1.5},
        }
    )
    out = r.redact(e)
    assert out.data["call_id"] == "call_4111111111111111" and out.data["name"] == "a@b.co"
    assert out.data["arguments"]["name"] == "[REDACTED:email]"
    assert out.data["usage"] == {"cost": 1.5}


def test_redaction_is_idempotent_scoped_by_type_and_covers_meta():
    r = ChainRedactor(
        KeyRedactor(only=["user.*"]), PatternRedactor(only=["user.*"], skip=["user.audit"])
    )
    e = ev({"password": "p", "note": "a@b.co"}, "user.msg", meta={"who": "a@b.co"})
    once = r.redact(e)
    assert once.data == {"password": "[REDACTED]", "note": "[REDACTED:email]"}
    assert once.meta == {"who": "[REDACTED:email]"}
    assert r.redact(once) == once
    assert r.redact(ev({"note": "a@b.co"}, "aaa.llm.requested")).data == {"note": "a@b.co"}
    assert r.redact(ev({"note": "a@b.co"}, "user.audit")).data == {"note": "a@b.co"}


# ---- engine fixtures ---------------------------------------------------------------------------


def script(req):
    last = req.messages[-1]
    return tool_call("lookup", q="x") if last.role == "user" else text(f"done: {last.content}")


def make_engine(*, blobs=None, store=None, **kw):
    @tool
    async def lookup(q: str) -> str:
        """look something up"""
        return "customer ann@example.com has card 4111 1111 1111 1111 " + "z" * 400

    eng = Engine(store=store, blobs=blobs, **kw)
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry([lookup])
    ToolHandler(reg).attach(eng)
    algo = ToolLoop(reg)
    eng.register(algo)
    return eng, algo


# ---- redaction through the engine --------------------------------------------------------------


async def test_the_log_the_bus_and_the_blobs_never_see_the_original():
    blobs = InMemoryBlobStore()
    eng, algo = make_engine(blobs=blobs, claim_threshold=200, redactor=standard_redactor())
    seen: list[Event] = []

    async def spy(e):
        seen.append(e)

    eng.subscribe("aaa.*", spy)
    async with eng:
        out = await eng.run(algo, {"task": "mail bob@example.com about it"}, run_id="r1")
        log = await eng.store.read("r1")
    assert out.ok
    everything = " ".join(str(e.data) for e in [*log, *seen])
    assert "example.com" not in everything and "4111" not in everything
    assert "[REDACTED:email]" in everything
    assert log[-1].data["kind"] == "success"  # the outcome is intact
    assert any(e.blobs for e in log), "the big tool result was claim-checked"
    for ref in {b for e in log for b in e.blobs}:
        raw = (await blobs.get(ref)).decode()
        assert "example.com" not in raw and "4111" not in raw


# ---- tenancy -----------------------------------------------------------------------------------


async def test_tenant_is_stamped_on_every_event_of_a_run_and_its_children():
    eng, algo = make_engine()
    async with eng:
        acme = eng.tenant("acme")
        h = await acme.start(algo, {"task": "go"}, run_id="job-1")
        assert h.run_id == "acme/job-1"
        await h.result()
        child = await eng.start(algo, {"task": "kid"}, parent="acme/job-1", run_id="acme/job-1.kid")
        await child.result()
        for rid in ("acme/job-1", "acme/job-1.kid"):
            log = await eng.store.read(rid)
            assert {e.tenant for e in log} == {"acme"}, rid
        sig = await h.signal("x.after", {"k": 1})  # an external event gets the run's tenant too
        assert sig is not None and sig.tenant == "acme"


async def test_tenants_cannot_see_touch_or_probe_each_others_runs():
    eng, algo = make_engine()
    async with eng:
        acme, globex = eng.tenant("acme"), eng.tenant("globex")
        a = await acme.start(algo, {"task": "a"}, run_id="order-1")
        g = await globex.start(algo, {"task": "g"}, run_id="order-1")  # same id, no collision
        assert a.run_id != g.run_id
        await a.result(), await g.result()
        assert [r.stream for r in await acme.runs()] == ["acme/order-1"]
        assert [r.stream for r in await globex.runs()] == ["globex/order-1"]
        with pytest.raises(KeyError):
            await acme.handle("globex/order-1")  # someone else's run looks like a missing one
        with pytest.raises(KeyError):
            await acme.erase("globex/order-1")
        with pytest.raises(KeyError):
            await acme.start(algo, {"task": "x"}, parent="globex/order-1")
        assert len(await eng.store.read("globex/order-1")) == len(
            await eng.store.read("acme/order-1")
        )
        assert (await (await acme.handle("order-1")).result()).ok  # bare id resolves in-tenant


async def test_require_tenant_refuses_untenanted_runs_and_bad_names_are_rejected():
    eng, algo = make_engine(require_tenant=True)
    async with eng:
        with pytest.raises(ValueError, match="requires a tenant"):
            await eng.start(algo, {"task": "x"})
        assert (await eng.tenant("acme").run(algo, {"task": "x"})).ok
    for bad in ("", "a/b", " pad"):
        with pytest.raises(ValueError):
            eng.tenant(bad)


async def test_identical_large_payloads_of_two_tenants_are_separate_blobs():
    blobs = InMemoryBlobStore()
    eng, algo = make_engine(blobs=blobs, claim_threshold=200)
    async with eng:
        for t in ("acme", "globex"):
            await eng.tenant(t).run(algo, {"task": "same"}, run_id="r")
        la = {b.sha256 for e in await eng.store.read("acme/r") for b in e.blobs}
        lg = {b.sha256 for e in await eng.store.read("globex/r") for b in e.blobs}
        assert la and lg and not (la & lg)
        await eng.tenant("acme").erase("r")
        for e in await eng.store.read("globex/r"):  # globex's payloads still hydrate
            await eng.hydrate(e)


# ---- erasure and retention ---------------------------------------------------------------------


async def test_erasing_a_finished_run_removes_payloads_blobs_and_snapshots_but_keeps_the_audit_trail():
    blobs = InMemoryBlobStore()
    eng, algo = make_engine(blobs=blobs, claim_threshold=200)
    async with eng:
        acme = eng.tenant("acme")
        h = await acme.start(algo, {"task": "secret plan for ann@example.com"}, run_id="r")
        await h.result()
        before = await eng.store.read("acme/r")
        refs = [b for e in before for b in e.blobs]
        assert refs and all([await blobs.exists(b) for b in refs])
        await eng.project(__import__("aaa.runtime.engine", fromlist=["RUNMETA"]).RUNMETA, "acme/r")
        rep = await acme.erase("r", reason="gdpr-request-42")
        assert rep.mode == "scrub" and rep.events == len(before) and rep.blobs == len(refs)
        assert not any([await blobs.exists(b) for b in refs])
        after = await eng.store.read("acme/r")
        assert [e.id for e in after[:-1]] == [e.id for e in before]
        assert after[-1].type == "aaa.run.erased" and after[-1].data["reason"] == "gdpr-request-42"
        assert "ann@example.com" not in str([e.data for e in after])
        assert all(e.tenant == "acme" for e in after)
        assert (await h.result()).ok  # the verdict survives; the payload does not
        again = await acme.erase("r")  # idempotent
        assert again.events == len(after)
        assert len(await eng.store.read("acme/r")) == len(after)  # marker not duplicated


async def test_delete_mode_removes_the_run_entirely_and_open_runs_are_protected():
    eng, algo = make_engine()
    async with eng:
        done = await eng.start(algo, {"task": "x"}, run_id="d")
        await done.result()
        rep = await eng.erase("d", delete=True)
        assert rep.mode == "delete" and await eng.store.read("d") == []

        eng.register(algo)
        open_run = await eng.start(algo, {"task": "x"}, run_id="o")  # may or may not have finished
        await open_run.cancel()
        await open_run.result()
        await eng.erase("o")  # finished (cancelled) runs can be erased
        with pytest.raises(KeyError):
            await eng.erase("never-existed")


async def test_erasing_a_run_that_is_still_open_is_refused_unless_forced():
    store = InMemoryEventStore()
    await store.append(
        "open",
        [Event(type="aaa.run.started", stream="open", run_id="open", data={"algo": "a"})],
    )
    eng = Engine(store=store)
    async with eng:
        with pytest.raises(RunNotFinished):
            await eng.erase("open")
        assert (await eng.erase("open", force=True)).events == 1


async def test_purge_erases_only_old_finished_runs_of_the_requested_tenant():
    eng, algo = make_engine()
    async with eng:
        for t in ("acme", "globex"):
            await eng.tenant(t).run(algo, {"task": "x"}, run_id="r")
        assert (await eng.purge(older_than=timedelta(days=1))).erased == []  # nothing is that old
        report = await eng.tenant("acme").purge(older_than=timedelta(0))
        assert report.streams == ["acme/r"]
        assert "[REDACTED" not in str(await eng.store.read("globex/r"))
        assert (await eng.store.read("globex/r"))[0].data.get("$erased") is None
        assert (await eng.store.read("acme/r"))[0].data.get("$erased") is True
        assert (await eng.purge(older_than=timedelta(0))).streams == [
            "globex/r"
        ]  # acme: done already
        assert (await eng.tenant("acme").erase_all()).erased == []


async def test_retention_policies_run_from_sweep_first_matching_policy_wins():
    eng, algo = make_engine(
        retention=[
            RetentionPolicy(older_than=timedelta(days=365), tenant="vip"),  # vip: keep a year
            RetentionPolicy(older_than=timedelta(0), delete=True),  # everyone else: delete now
        ]
    )
    async with eng:
        await eng.tenant("vip").run(algo, {"task": "x"}, run_id="r")
        await eng.tenant("other").run(algo, {"task": "x"}, run_id="r")
        report = await eng.sweep()
        assert report.erased == ["other/r"]
        assert await eng.store.read("other/r") == []
        assert (await eng.store.read("vip/r"))[0].data.get("$erased") is None
        assert (await eng.sweep()).erased == []  # rate-limited and nothing left to do


# ---- in a cluster (real stores and bus) ---------------------------------------------------------


async def test_erasure_through_an_engine_on_every_backend(cluster):
    node = await cluster.node()
    bus = DistributedBus(node.backend, member="A", **FAST)
    eng, algo = make_engine(store=node.store, bus=bus)
    async with eng:
        acme = eng.tenant("acme")
        h = await acme.start(algo, {"task": "for ann@example.com"}, run_id="c1")
        out = await h.result()
        assert out.ok
        await acme.erase("c1")
        log = await node.store.read("acme/c1")
        assert log[-1].type == "aaa.run.erased"
        assert "ann@example.com" not in str([e.data for e in log])
        assert [r.stream for r in await acme.runs()] == ["acme/c1"]
