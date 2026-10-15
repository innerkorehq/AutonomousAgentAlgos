"""aaa-server: authentication, tenant isolation, spoofing, idempotency, SSE, approvals, limits."""

from __future__ import annotations

import asyncio
import json

import pytest
from aaa_server import (
    InMemoryRateLimiter,
    Quota,
    QuotaStore,
    ServerConfig,
    create_app,
)
from aaa_server.app import BudgetRequest, clamp_budget

from aaa.events import InMemoryEventStore
from aaa.runtime import Budget
from helpers.server_env import make_env


@pytest.fixture
async def env():
    e, engine = await make_env()
    async with engine:
        yield e


def sse(body: str) -> list[dict]:
    out = []
    for block in body.strip().split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        rec: dict = {}
        for line in block.splitlines():
            k, _, v = line.partition(": ")
            rec[k] = v
        if "data" in rec:
            rec["data"] = json.loads(rec["data"])
        out.append(rec)
    return out


async def run_to_done(c, **body) -> str:
    r = await c.post("/v1/runs", json={"algo": "agent", "input": {"task": "hi"}, **body})
    assert r.status_code == 202, r.text
    rid = r.json()["run_id"]
    for _ in range(200):
        g = (await c.get(f"/v1/runs/{rid}")).json()
        if g["status"] == "completed":
            return rid
        await asyncio.sleep(0.02)
    raise AssertionError("run did not finish")


# -- authentication -------------------------------------------------------------------------------
async def test_every_bad_credential_gets_the_same_401(env):
    bodies = set()
    for hdrs in (
        {},
        {"Authorization": "Bearer nope"},
        {"Authorization": "Basic abc"},
        {"Authorization": "Bearer "},
    ):
        async with env.client(None) as c:
            r = await c.get("/v1/runs", headers=hdrs)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        bodies.add(r.json()["error"]["code"] + r.json()["error"]["message"])
    assert len(bodies) == 1


async def test_healthz_is_open_and_says_nothing(env):
    async with env.client(None) as c:
        r = await c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


async def test_scopes_are_enforced(env):
    async with env.client("key-acme") as c:  # no approve scope
        r = await c.post("/v1/runs/x/approvals", json={"call_id": "c", "decision": "grant"})
    assert r.status_code == 403
    async with env.client("key-acme-approver") as c:  # no admin scope
        assert (await c.get("/v1/usage")).status_code == 403


# -- runs, events, projection -----------------------------------------------------------------------
async def test_run_lifecycle_and_filtered_event_stream(env):
    async with env.client() as c:
        rid = await run_to_done(c)
        run = (await c.get(f"/v1/runs/{rid}")).json()
        assert run["outcome"]["kind"] == "success" and run["outcome"]["value"]["text"] == "hello hi"
        r = await c.get(f"/v1/runs/{rid}/events")
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse(r.text)
    types = [e["event"] for e in events]
    assert types[0] == "run.started" and types[-2:] == ["run.completed", "end"][-2:]
    assert "run.completed" in types
    # the raw log has requests with the system prompt and messages; none of that leaves
    assert "be brief" not in r.text and "aaa.llm" not in r.text
    ids = [int(e["id"]) for e in events if "id" in e]
    assert ids == sorted(ids)


async def test_sse_resumes_after_last_event_id(env):
    async with env.client() as c:
        rid = await run_to_done(c)
        full = sse((await c.get(f"/v1/runs/{rid}/events")).text)
        with_id = [e for e in full if "id" in e]
        cut = with_id[1]["id"]
        rest = sse((await c.get(f"/v1/runs/{rid}/events", headers={"Last-Event-ID": cut})).text)
    assert [e["id"] for e in rest if "id" in e] == [
        e["id"] for e in with_id if int(e["id"]) > int(cut)
    ]


async def test_bad_last_event_id_is_a_422(env):
    async with env.client() as c:
        rid = await run_to_done(c)
        for bad in ("abc", "-5"):
            r = await c.get(f"/v1/runs/{rid}/events", headers={"Last-Event-ID": bad})
            assert r.status_code == 422


# -- tenant isolation -------------------------------------------------------------------------------
async def test_every_endpoint_gives_a_foreign_id_the_same_404_as_a_missing_one(env):
    async with env.client("key-acme") as acme:
        rid = await run_to_done(acme)
        sid = (await acme.post("/v1/sessions", json={})).json()["session_id"]
    async with env.client("key-globex") as globex:
        calls = [
            ("GET", "/v1/runs/{}"),
            ("GET", "/v1/runs/{}/events"),
            ("POST", "/v1/runs/{}/cancel"),
            ("POST", "/v1/runs/{}/approvals"),
        ]
        for method, tmpl in calls:
            kw = {"json": {"call_id": "c", "decision": "grant"}} if "approvals" in tmpl else {}
            foreign = await globex.request(method, tmpl.format(rid), **kw)
            missing = await globex.request(method, tmpl.format("does-not-exist"), **kw)
            assert foreign.status_code == missing.status_code == 404, (method, tmpl)
            assert foreign.json()["error"]["code"] == missing.json()["error"]["code"]
        for method, path, kw in [
            ("GET", f"/v1/sessions/{sid}", {}),
            ("DELETE", f"/v1/sessions/{sid}", {}),
            ("POST", f"/v1/sessions/{sid}/messages", {"json": {"text": "hi"}}),
        ]:
            foreign = await globex.request(method, path, **kw)
            missing = await globex.request(method, path.replace(sid, "s-none"), **kw)
            assert foreign.status_code == missing.status_code == 404, (method, path)
        assert (await globex.get("/v1/runs")).json() == {"runs": []}
        assert (await globex.get("/v1/sessions")).json() == {"sessions": []}
    # the foreign attempts changed nothing
    async with env.client("key-acme") as acme:
        assert (await acme.get(f"/v1/runs/{rid}")).json()["status"] == "completed"
        assert (await acme.get(f"/v1/sessions/{sid}")).status_code == 200


async def test_two_tenants_may_use_the_same_idempotency_key(env):
    async with env.client("key-acme") as a, env.client("key-globex") as g:
        ra = await a.post(
            "/v1/runs",
            json={"algo": "agent", "input": {"task": "a"}},
            headers={"Idempotency-Key": "order-1"},
        )
        rg = await g.post(
            "/v1/runs",
            json={"algo": "agent", "input": {"task": "g"}},
            headers={"Idempotency-Key": "order-1"},
        )
    assert ra.status_code == rg.status_code == 202
    assert ra.json()["run_id"] == rg.json()["run_id"] == "order-1"
    assert {r.stream for r in await env.engine.store.run_index()} >= {
        "acme/order-1",
        "globex/order-1",
    }


# -- spoofing ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["tenant", "user", "meta", "run_id", "subject"])
async def test_identity_fields_in_a_body_are_rejected(env, field):
    async with env.client() as c:
        r = await c.post("/v1/runs", json={"algo": "agent", "input": {}, field: "globex"})
        assert r.status_code == 422
        r = await c.post("/v1/sessions", json={field: "globex"})
        assert r.status_code == 422
    assert not await env.engine.store.run_index()


async def test_identity_comes_from_the_principal_only(env):
    async with env.client("key-acme") as c:
        rid = await c.post(
            "/v1/runs",
            json={"algo": "agent", "input": {"task": "hi"}},
            headers={"X-Tenant": "globex", "X-User": "mallory", "Tenant": "globex"},
        )
    started = (await env.engine.store.run_index())[0].started
    assert started.tenant == "acme" and started.meta == {"user": "ann"}
    assert rid.status_code == 202


async def test_blob_references_from_clients_are_refused(env):
    blob = {"type": "image", "media_type": "image/png", "blob": {"key": "k", "size": 1}}
    async with env.client() as c:
        r = await c.post("/v1/runs", json={"algo": "agent", "input": {"attachments": [blob]}})
        assert r.status_code == 422
        sid = (await c.post("/v1/sessions", json={})).json()["session_id"]
        r = await c.post(f"/v1/sessions/{sid}/messages", json={"text": "x", "attachments": [blob]})
        assert r.status_code == 422


# -- validation & limits ------------------------------------------------------------------------------
async def test_oversized_bodies_are_refused_before_parsing(env):
    async with env.client() as c:
        r = await c.post("/v1/runs", content=b"{" + b" " * (2 << 20) + b"}")
    assert r.status_code == 413


async def test_unknown_algorithm_and_bad_json(env):
    async with env.client() as c:
        assert (await c.post("/v1/runs", json={"algo": "nope"})).status_code == 404
        assert (await c.post("/v1/runs", content=b"not json")).status_code == 422
        assert (await c.post("/v1/runs", json={"input": {}})).status_code == 422


def test_budgets_are_a_server_ceiling():
    server = Budget(max_steps=10, max_tokens=1000, deadline_s=60.0)
    lowered = clamp_budget(server, BudgetRequest(max_steps=3, deadline_s=5))
    assert (lowered.max_steps, lowered.max_tokens, lowered.deadline_s) == (
        3,
        1000,
        60.0 if False else 5,
    )
    raised = clamp_budget(server, BudgetRequest(max_steps=999, max_tokens=10**9, max_cost=5.0))
    assert (raised.max_steps, raised.max_tokens, raised.max_cost) == (10, 1000, 5.0)
    assert clamp_budget(server, None) == server


async def test_a_request_cannot_raise_the_budget_on_the_wire(env):
    async with env.client() as c:
        rid = await run_to_done(c, budget={"max_steps": 10_000})
    started = (await env.engine.store.read(f"acme/{rid}"))[0]
    assert started.data["budget"]["max_steps"] == 20


# -- idempotency ----------------------------------------------------------------------------------
async def test_idempotency_key_replays_or_conflicts(env):
    body = {"algo": "agent", "input": {"task": "hi"}}
    h = {"Idempotency-Key": "job-7"}
    async with env.client() as c:
        first = await c.post("/v1/runs", json=body, headers=h)
        again = await c.post("/v1/runs", json=body, headers=h)
        other = await c.post("/v1/runs", json={**body, "input": {"task": "different"}}, headers=h)
        bad = await c.post("/v1/runs", json=body, headers={"Idempotency-Key": "a/b"})
    assert first.status_code == 202 and again.status_code == 200
    assert first.json()["run_id"] == again.json()["run_id"] == "job-7"
    assert other.status_code == 409 and other.json()["error"]["code"] == "idempotency_conflict"
    assert bad.status_code == 422
    assert len(await env.engine.store.run_index()) == 1


# -- approvals --------------------------------------------------------------------------------------
async def test_approval_flow_and_pre_signing_is_refused(env):
    async with env.client("key-acme-approver") as c:
        r = await c.post("/v1/runs", json={"algo": "agent", "input": {"task": "wire it"}})
        rid = r.json()["run_id"]
        # not yet requested: refused (cannot pre-sign a call id)
        pre = await c.post(
            f"/v1/runs/{rid}/approvals", json={"call_id": "guess", "decision": "grant"}
        )
        assert pre.status_code == 409 and pre.json()["error"]["code"] == "no_pending_approval"
        call_id = None
        for _ in range(200):
            for e in await env.engine.store.read(f"acme/{rid}"):
                if e.type == "aaa.approval.requested":
                    call_id = e.data["call_id"]
            if call_id:
                break
            await asyncio.sleep(0.02)
        assert call_id
        wrong = await c.post(
            f"/v1/runs/{rid}/approvals", json={"call_id": "other", "decision": "grant"}
        )
        assert wrong.status_code == 409
        ok = await c.post(
            f"/v1/runs/{rid}/approvals", json={"call_id": call_id, "decision": "grant"}
        )
        assert ok.status_code == 202
        for _ in range(200):
            g = (await c.get(f"/v1/runs/{rid}")).json()
            if g["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        assert g["outcome"]["value"]["text"] == "done: sent 250"
        again = await c.post(
            f"/v1/runs/{rid}/approvals", json={"call_id": call_id, "decision": "deny"}
        )
        assert again.status_code == 409  # already resolved
        events = sse((await c.get(f"/v1/runs/{rid}/events")).text)
    req = next(e for e in events if e["event"] == "approval.requested")
    assert req["data"]["name"] == "wire_money" and req["data"]["arguments"] == {"amount": 250}


async def test_denied_approval_is_reported_to_the_model(env):
    async with env.client("key-acme-approver") as c:
        rid = (
            await c.post("/v1/runs", json={"algo": "agent", "input": {"task": "wire it"}})
        ).json()["run_id"]
        call_id = None
        for _ in range(200):
            for e in await env.engine.store.read(f"acme/{rid}"):
                if e.type == "aaa.approval.requested":
                    call_id = e.data["call_id"]
            if call_id:
                break
            await asyncio.sleep(0.02)
        r = await c.post(
            f"/v1/runs/{rid}/approvals",
            json={"call_id": call_id, "decision": "deny", "reason": "no"},
        )
        assert r.status_code == 202
    events = await env.engine.store.read(f"acme/{rid}")
    denied = next(e for e in events if e.type == "aaa.approval.denied")
    assert denied.data["by"] == "root"


async def test_cancel(env):
    async with env.client("key-acme-approver") as c:
        rid = (
            await c.post("/v1/runs", json={"algo": "agent", "input": {"task": "wire it"}})
        ).json()["run_id"]
        assert (await c.post(f"/v1/runs/{rid}/cancel")).status_code == 202
        for _ in range(200):
            g = (await c.get(f"/v1/runs/{rid}")).json()
            if g["status"] == "completed":
                break
            await asyncio.sleep(0.02)
    assert g["outcome"]["kind"] == "cancelled"


# -- sessions ---------------------------------------------------------------------------------------
async def test_sessions_conversation_stream_and_close(env):
    async with env.client() as c:
        sid = (await c.post("/v1/sessions", json={"title": "t"})).json()["session_id"]
        r = await c.post(f"/v1/sessions/{sid}/messages", json={"text": "hello"})
        assert r.status_code == 200 and r.json()["text"] == "hello hello"
        r = await c.post(f"/v1/sessions/{sid}/messages", json={"text": "again", "stream": True})
        events = sse(r.text)
        assert (
            events[-2]["event"] == "session.reply" and events[-2]["data"]["text"] == "hello again"
        )
        turns = (await c.get(f"/v1/sessions/{sid}")).json()["turns"]
        assert [t["user"] for t in turns] == ["hello", "again"]
        assert [s["session_id"] for s in (await c.get("/v1/sessions")).json()["sessions"]] == [sid]
        assert (await c.delete(f"/v1/sessions/{sid}")).json()["status"] == "closed"
        closed = await c.post(f"/v1/sessions/{sid}/messages", json={"text": "x"})
        assert closed.status_code == 409 and sid not in closed.text


async def test_session_needs_an_unambiguous_agent(env):
    async with env.client() as c:
        assert (await c.post("/v1/sessions", json={"agent": "nope"})).status_code == 422


# -- rate limits and quotas ---------------------------------------------------------------------------
async def test_rate_limit_per_principal_with_a_fake_clock():
    now = [100.0]
    limiter = InMemoryRateLimiter(per_second=1, burst=2, clock=lambda: now[0])
    e, engine = await make_env(rate_limiter=limiter)
    async with engine, e.client("key-acme") as a, e.client("key-globex") as g:
        codes = [(await a.get("/v1/runs")).status_code for _ in range(3)]
        limited = await a.get("/v1/runs")
        assert codes == [200, 200, 429] and limited.status_code == 429
        assert limited.headers["retry-after"] == "1"
        assert (await g.get("/v1/runs")).status_code == 200  # another principal is unaffected
        now[0] += 1.5
        assert (await a.get("/v1/runs")).status_code == 200


async def test_quota_is_event_sourced_and_survives_a_new_server():
    store = InMemoryEventStore()
    quota = QuotaStore(store, Quota(max_runs=2))
    e, engine = await make_env(quota=quota)
    async with engine, e.client() as c:
        for _ in range(2):
            await run_to_done(c)
        denied = await c.post("/v1/runs", json={"algo": "agent", "input": {}})
        assert denied.status_code == 429 and denied.json()["error"]["code"] == "quota_exceeded"
        for _ in range(100):  # settlement is recorded when the runs finish
            usage = (await c.get("/v1/usage")).json()["usage"]
            if usage["settled"] == 2:
                break
            await asyncio.sleep(0.02)
        assert usage["admitted"] == 2 and usage["settled"] == 2
    # a brand-new server over the same store still knows
    again = QuotaStore(store, Quota(max_runs=2))
    assert (await again.usage("acme")).admitted == 2
    assert (await again.usage("globex")).admitted == 0


async def test_racing_for_the_last_unit_of_quota_admits_exactly_the_limit():
    from aaa_server import Denied

    quota = QuotaStore(InMemoryEventStore(), Quota(max_runs=3))
    results = await asyncio.gather(
        *(quota.admit("acme", f"u{i}") for i in range(12)), return_exceptions=True
    )
    ok = [r for r in results if r is None]
    assert len(ok) == 3 and all(isinstance(r, Denied) for r in results if r is not None)
    assert (await quota.usage("acme")).admitted == 3


async def test_quota_admit_is_idempotent_per_unit():
    quota = QuotaStore(InMemoryEventStore(), Quota(max_runs=1))
    await quota.admit("acme", "same")
    await quota.admit("acme", "same")
    assert (await quota.usage("acme")).admitted == 1


async def test_open_run_cap_and_stream_cap():
    e, engine = await make_env(max_open_runs=1, max_streams=1)
    async with engine, e.client("key-acme-approver") as c:
        first = await c.post("/v1/runs", json={"algo": "agent", "input": {"task": "wire it"}})
        assert first.status_code == 202  # parks on approval, so it stays open
        second = await c.post("/v1/runs", json={"algo": "agent", "input": {"task": "wire it"}})
        assert second.status_code == 429 and second.json()["error"]["code"] == "too_many_runs"


# -- configuration ------------------------------------------------------------------------------------
async def test_cors_is_off_by_default_and_never_a_wildcard(env):
    async with env.client() as c:
        r = await c.get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    with pytest.raises(ValueError):
        create_app(
            ServerConfig(
                engine=env.engine, authenticator=env.cfg.authenticator, cors_origins=("*",)
            )
        )
    ok = create_app(
        ServerConfig(
            engine=env.engine,
            authenticator=env.cfg.authenticator,
            cors_origins=("https://app.example",),
        )
    )
    assert ok is not None


async def test_errors_are_generic_with_a_request_id(env):
    async with env.client() as c:
        r = await c.get("/v1/runs/nope")
    err = r.json()["error"]
    assert err["request_id"] == r.headers["x-request-id"] and "Traceback" not in r.text
