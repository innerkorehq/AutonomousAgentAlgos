"""Behaviour specific to each adapter (the shared contract lives in test_memory_contract.py)."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa.memory import MemoryEntry, MemoryHandler, Scope
from aaa.memory.graphiti import GraphitiMemory
from aaa.memory.langgraph import LangGraphStoreMemory
from aaa.memory.mem0 import Mem0Memory
from aaa.memory.supermemory import SupermemoryMemory
from aaa.runtime import Engine
from helpers.memory_stubs import (
    StubGraphiti,
    StubMem0,
    StubMem0Hosted,
    StubStore,
    SupermemoryServer,
    graphiti_ops,
)

ANN = Scope(tenant="acme", user="ann")


def fact(text, **kw):
    return MemoryEntry(text=text, **kw)


# ---- Scope.slug ----------------------------------------------------------------------------------
def test_slugs_are_safe_for_strict_backends_and_never_collide():
    import re

    scopes = [
        Scope(tenant="acme"),
        Scope(tenant="acme", user="ann"),
        Scope(tenant="acme", user="ann", agent="bot"),
        Scope(tenant="acme", user="a nn"),
        Scope(tenant="acme", user="a_nn"),
        Scope(tenant="acme", user="a-nn"),
        Scope(tenant="acme.co", user="ann@x.io"),
        Scope(tenant="acme", user="ann", namespace=("proj", "x")),
        Scope(),
    ]
    slugs = [s.slug() for s in scopes]
    assert len(set(slugs)) == len(slugs)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in slugs)  # Graphiti / Supermemory / Mem0
    assert Scope(tenant="acme", user="ann").slug().startswith(Scope(tenant="acme").slug() + "--")
    assert (
        Scope(tenant="acme", user="ann", session="s1").slug()
        == Scope(tenant="acme", user="ann").slug()
    )


def test_long_slugs_are_shortened_deterministically_and_stay_distinct():
    a, b = Scope(tenant="t" * 150, user="u1"), Scope(tenant="t" * 150, user="u2")
    assert len(a.slug(max_len=100)) == 100 and a.slug(max_len=100) == a.slug(max_len=100)
    assert a.slug(max_len=100) != b.slug(max_len=100)


# ---- Mem0 ----------------------------------------------------------------------------------------
async def test_mem0_scopes_to_a_tenant_qualified_entity_and_tags_metadata():
    stub = StubMem0()
    mem = Mem0Memory(stub, threshold=0.3)
    await mem.add([fact("x is true", role="user")], ANN)
    await mem.search("x", ANN)
    add, search = stub.calls[0], stub.calls[1]
    assert add[2]["user_id"] == ANN.slug() and add[2]["metadata"]["aaa_tenant"] == "acme"
    assert add[1][0] == [{"role": "user", "content": "x is true"}]
    assert search[2]["filters"] == {"user_id": ANN.slug()} and search[2]["threshold"] == 0.3


async def test_mem0_tenant_wide_scope_uses_the_wildcard_entity_plus_tenant_metadata():
    stub = StubMem0()
    mem = Mem0Memory(stub)
    await mem.search("x", Scope(tenant="acme"))
    assert stub.calls[0][2]["filters"] == {"user_id": "*", "aaa_tenant": "acme"}


async def test_the_hosted_mem0_client_gets_identity_inside_filters_and_page_size():
    stub = StubMem0Hosted()
    mem = Mem0Memory(stub)
    assert mem.hosted
    await mem.add([fact("hosted fact")], ANN)
    await mem.list(ANN, limit=7)
    add = next(c for c in stub.calls if c[0] == "add")
    assert add[2]["filters"] == {"user_id": ANN.slug()} and "user_id" not in add[2]
    assert next(c for c in stub.calls if c[0] == "get_all")[2]["page_size"] == 7


async def test_a_pending_mem0_platform_write_is_reported_as_pending():
    class Pending(StubMem0Hosted):
        async def add(self, messages, **kw):
            return {"status": "PENDING", "event_id": "ev-1"}

    res = await Mem0Memory(Pending()).add([fact("later")], ANN)
    assert res.pending and res.ids == [] and res.detail["event_id"] == "ev-1"


async def test_mem0_writes_are_never_retried_automatically():
    mem = Mem0Memory(StubMem0())
    assert mem.idempotent is False
    assert MemoryHandler(mem).delivery == "at_most_once"  # a dead worker is reported, not repeated
    assert MemoryHandler(GraphitiMemory(StubGraphiti())).delivery == "at_least_once"


# ---- Supermemory -----------------------------------------------------------------------------------
async def test_supermemory_documents_are_idempotent_through_custom_id_and_wait_for_processing():
    server = SupermemoryServer()
    mem = SupermemoryMemory("sk-test", client=server.client, wait=True)
    assert mem.idempotent
    first = await mem.add([fact("Ann likes jazz")], ANN, key="req-1")
    again = await mem.add([fact("Ann likes jazz")], ANN, key="req-1")
    assert first.ids == again.ids and len(server.docs) == 1 and not first.pending
    assert any(
        m == "GET" and p.startswith("/v3/documents/") for m, p, _ in server.requests
    )  # polled
    body = next(b for m, p, b in server.requests if (m, p) == ("POST", "/v3/documents"))
    assert (
        body["containerTag"] == ANN.slug()
        and body["customId"]
        and body["content"] == "Ann likes jazz"
    )


async def test_supermemory_without_wait_returns_pending():
    mem = SupermemoryMemory("k", client=SupermemoryServer().client)
    assert (await mem.add([fact("queued")], ANN)).pending


async def test_supermemory_sends_the_bearer_key_and_exposes_the_profile():
    seen: list[httpx.Request] = []
    server = SupermemoryServer()

    def spy(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return server.handle(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(spy), base_url="https://sm.test")
    mem = SupermemoryMemory("sk-secret", client=client, ingest="memories")
    await mem.add([fact("Ann is a pilot")], ANN)
    assert (await mem.profile(ANN))["static"] == ["Ann is a pilot"]
    # a caller-supplied client carries its own auth; the key we were given is only used for our own
    own = SupermemoryMemory("sk-secret")
    assert own.http.headers["authorization"] == "Bearer sk-secret"
    await own.aclose()


async def test_supermemory_container_tags_respect_the_100_character_limit():
    assert len(SupermemoryMemory.tag(Scope(tenant="t" * 300, user="u"))) <= 100


async def test_supermemory_delete_tolerates_already_forgotten_memories():
    server = SupermemoryServer()
    mem = SupermemoryMemory("k", client=server.client, ingest="memories")
    mid = (await mem.add([fact("temp")], ANN)).ids[0]
    assert await mem.delete([mid, "never-existed"], ANN) == 1


# ---- Graphiti --------------------------------------------------------------------------------------
async def test_graphiti_returns_only_current_facts_unless_asked_about_a_time(monkeypatch):
    g = StubGraphiti()
    monkeypatch.setattr("aaa.memory.graphiti._ops", lambda: graphiti_ops(g))
    mem = GraphitiMemory(g)
    t0 = datetime(2023, 1, 1, tzinfo=UTC)
    await mem.add([fact("Ann works at Initech", time=t0)], ANN)
    await mem.add([fact("Ann works at Globex", time=t0 + timedelta(days=400))], ANN)
    g.edges[0].invalid_at = t0 + timedelta(days=400)  # the newer fact superseded the older one
    now = [m.text for m in await mem.search("Ann works", ANN)]
    assert now == ["Ann works at Globex"]
    then = [m.text for m in await mem.search("Ann works", ANN, as_of=t0 + timedelta(days=30))]
    assert then == ["Ann works at Initech"]
    assert not GraphitiMemory(g, current_only=False).current_only


async def test_graphiti_episodes_get_deterministic_uuids_and_the_scope_group(monkeypatch):
    g = StubGraphiti()
    monkeypatch.setattr("aaa.memory.graphiti._ops", lambda: graphiti_ops(g))
    mem = GraphitiMemory(g)
    await mem.add([fact("one", role="user")], ANN, key="k")
    await mem.add([fact("one", role="user")], ANN, key="k")
    calls = [c for c in g.calls if c[0] == "add_episode"]
    assert calls[0][2]["uuid"] == calls[1][2]["uuid"] and calls[0][2]["group_id"] == ANN.slug()
    assert calls[0][2]["episode_body"] == "user: one" and len(g.episodes) == 1


async def test_graphiti_tenant_wide_scope_expands_to_every_group_of_the_tenant(monkeypatch):
    g = StubGraphiti()
    monkeypatch.setattr("aaa.memory.graphiti._ops", lambda: graphiti_ops(g))
    mem = GraphitiMemory(g)
    await mem.add([fact("alpha fact")], ANN)
    await mem.add([fact("beta fact")], Scope(tenant="acme", user="bob"))
    await mem.add([fact("gamma fact")], Scope(tenant="acme2", user="ann"))  # a different tenant
    hits = await mem.search("fact", Scope(tenant="acme"), k=10)
    assert {h.text for h in hits} == {"alpha fact", "beta fact"}  # "acme2" is not a prefix match


# ---- LangGraph / LangMem ---------------------------------------------------------------------------
def test_langgraph_namespaces_are_hierarchical_so_tenant_wide_is_a_prefix():
    mem = LangGraphStoreMemory(StubStore(), root="mem")
    assert mem.namespace_for(Scope(tenant="acme")) == ("mem", "acme")
    assert mem.namespace_for(ANN) == ("mem", "acme", "ann")
    assert mem.namespace_for(Scope(tenant="acme", agent="bot")) == ("mem", "acme", "_", "bot")
    assert mem.namespace_for(Scope()) == ("mem", "_")
    assert mem.namespace_for(Scope(tenant="a.b"))[1] == "a_2e_b"  # labels may not contain "."


async def test_langgraph_adapter_reads_langmems_own_records():
    store = StubStore()
    mem = LangGraphStoreMemory(store)
    ns = mem.namespace_for(ANN)
    await store.aput(
        ns, "m1", {"kind": "Memory", "content": {"content": "Ann prefers aisle seats"}}
    )
    await store.aput(ns, "m2", {"content": "Ann is based in Oslo"})  # a plain-string content record
    hits = await mem.search("aisle seats", ANN)
    assert [h.text for h in hits] == ["Ann prefers aisle seats"] and hits[0].id == "m1"
    assert {m.text for m in await mem.list(ANN)} == {
        "Ann prefers aisle seats",
        "Ann is based in Oslo",
    }


async def test_langgraph_ranks_by_the_stores_score_when_it_has_an_embedding_index():
    class Scored(StubStore):
        async def asearch(self, prefix, /, *, query=None, **kw):
            rows = await super().asearch(prefix, query=query, **kw)
            for r, s in zip(rows, (0.2, 0.9), strict=False):
                r.score = s
            return rows

    mem = LangGraphStoreMemory(Scored())
    await mem.add([fact("weak match"), fact("strong match")], ANN)
    hits = await mem.search("anything", ANN)
    assert [h.score for h in hits] == [0.2, 0.9]  # the store's scores are passed through untouched


# ---- an adapter plugged into the engine ------------------------------------------------------------
async def test_an_external_adapter_serves_the_memory_handler_end_to_end():
    from aaa.memory import Scope as S

    server = SupermemoryServer()
    mem = SupermemoryMemory("k", client=server.client, ingest="memories")
    await mem.add([fact("Ann is a vegetarian")], ANN)
    eng = Engine()
    MemoryHandler(mem).attach(eng)
    async with eng:
        assert eng.memories == [mem]
        report = await eng.tenant("acme").forget_memory(user="ann")
    assert report.forgotten == {"SupermemoryMemory": 1} and not report.unsupported
    assert await mem.search("vegetarian", S(tenant="acme", user="ann")) == []
