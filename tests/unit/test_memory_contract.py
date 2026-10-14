"""One behavioural contract every Memory backend must satisfy (the adapters for external libraries
are held to the same suite with stubbed clients in test_memory_adapters.py)."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa.memory import (
    CallableMemory,
    Deletable,
    FakeMemory,
    Forgettable,
    Listable,
    Memory,
    MemoryEntry,
    MissingMemoryCapability,
    NativeMemory,
    Profiled,
    Scope,
    Updatable,
    require,
    supports,
)
from aaa.memory.graphiti import GraphitiMemory
from aaa.memory.langgraph import LangGraphStoreMemory
from aaa.memory.mem0 import Mem0Memory
from aaa.memory.supermemory import SupermemoryMemory
from helpers.memory_stubs import (
    StubGraphiti,
    StubMem0,
    StubMem0Hosted,
    StubStore,
    SupermemoryServer,
    graphiti_ops,
)


def _callable_over_fake() -> CallableMemory:
    inner = FakeMemory()
    return CallableMemory(
        add=lambda entries, scope, key=None: inner.add(entries, scope, key=key),
        search=lambda q, scope, k, filters=None, as_of=None: inner.search(
            q, scope, k=k, filters=filters, as_of=as_of
        ),
        update=inner.update,
        delete=inner.delete,
        forget=inner.forget,
        profile=inner.profile,
        list=lambda scope, limit: inner.list(scope, limit=limit),
        idempotent=True,
    )


ALL = {"update", "delete", "forget", "list", "temporal", "filters", "key"}
# What each backend can honestly do. A missing feature skips that test; it is not a failure,
# but it is written down here so the gaps are visible in one place.
FEATURES = {
    "fake": ALL,
    "native": ALL | {"exact_dedupe"},
    "callable": ALL,
    "mem0": {
        "update",
        "delete",
        "forget",
        "list",
        "filters",
    },  # no point-in-time recall, no idempotency key
    "mem0-hosted": {"update", "delete", "forget", "list", "filters"},
    "graphiti": {
        "delete",
        "forget",
        "list",
        "temporal",
        "key",
    },  # facts are derived: no update, no free-form metadata
    "supermemory": {"update", "delete", "forget", "list", "filters"},  # ingest="memories": no key
    "langgraph": ALL,
}


@pytest.fixture(params=list(FEATURES))
def memory(request, monkeypatch):
    name = request.param
    if name == "mem0":
        m: object = Mem0Memory(StubMem0())
    elif name == "mem0-hosted":
        m = Mem0Memory(StubMem0Hosted())
    elif name == "graphiti":
        g = StubGraphiti()
        monkeypatch.setattr("aaa.memory.graphiti._ops", lambda: graphiti_ops(g))
        m = GraphitiMemory(g)
    elif name == "supermemory":
        m = SupermemoryMemory("k", client=SupermemoryServer().client, ingest="memories")
    elif name == "langgraph":
        m = LangGraphStoreMemory(StubStore())
    else:
        m = {"fake": FakeMemory, "native": NativeMemory, "callable": _callable_over_fake}[name]()
    m.features = FEATURES[name]  # type: ignore[attr-defined]
    return m


def need(memory, feature: str) -> None:
    if feature not in memory.features:
        pytest.skip(f"{type(memory).__name__} does not provide {feature!r}")


ACME = Scope(tenant="acme", user="ann")
ACME_BOB = Scope(tenant="acme", user="bob")
GLOBEX = Scope(tenant="globex", user="ann")


def fact(text, **kw):
    return MemoryEntry(text=text, **kw)


async def test_add_then_search_returns_the_relevant_memory_first(memory):
    assert isinstance(memory, Memory)
    await memory.add([fact("Ann is vegetarian"), fact("Ann lives in Lisbon")], ACME)
    hits = await memory.search("what food does ann eat vegetarian", ACME, k=5)
    assert hits and hits[0].text == "Ann is vegetarian"
    assert hits[0].id and (hits[0].score is None or hits[0].score > 0)
    assert len(await memory.search("lisbon", ACME, k=1)) == 1


async def test_a_search_with_nothing_relevant_returns_nothing(memory):
    await memory.add([fact("Ann lives in Lisbon")], ACME)
    assert await memory.search("quantum chromodynamics", ACME) == []


async def test_tenants_and_users_never_see_each_others_memories(memory):
    await memory.add([fact("secret plan alpha")], ACME)
    await memory.add([fact("secret plan beta")], GLOBEX)
    assert [h.text for h in await memory.search("secret plan", ACME)] == ["secret plan alpha"]
    assert [h.text for h in await memory.search("secret plan", GLOBEX)] == ["secret plan beta"]
    assert await memory.search("secret plan", ACME_BOB) == []  # same tenant, other user
    assert (
        await memory.search("secret plan", Scope()) == []
    )  # an untenanted scope sees no tenant data
    wider = Scope(tenant="acme")  # a tenant-wide scope sees all of the tenant's users
    await memory.add([fact("bob prefers tea")], ACME_BOB)
    assert {h.text for h in await memory.search("secret plan bob tea", wider, k=10)} == {
        "secret plan alpha",
        "bob prefers tea",
    }


async def test_a_repeated_write_with_the_same_key_stores_once(memory):
    need(memory, "key")
    first = await memory.add([fact("Ann likes jazz")], ACME, key="req-1")
    again = await memory.add([fact("Ann likes jazz")], ACME, key="req-1")
    assert first.ids == again.ids
    assert len(await memory.search("jazz", ACME, k=10)) == 1


async def test_the_same_fact_written_twice_is_not_duplicated(memory):
    await memory.add([fact("Ann likes jazz")], ACME)
    await memory.add([fact("ann  likes   JAZZ")], ACME)
    need(memory, "exact_dedupe")  # content-addressed ids: part of NativeMemory's own contract
    assert len(await memory.search("jazz", ACME, k=10)) == 1


async def test_temporal_facts_can_be_asked_about_at_a_point_in_time(memory):
    need(memory, "temporal")
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    await memory.add([fact("Ann works at Initech", time=t0)], ACME)
    items = await memory.search("Ann works", ACME)
    assert items and items[0].valid_at == t0
    assert await memory.search("Ann works", ACME, as_of=t0 - timedelta(days=1)) == []
    assert await memory.search("Ann works", ACME, as_of=t0 + timedelta(days=1))


async def test_filters_match_on_metadata(memory):
    need(memory, "filters")
    await memory.add([fact("deploy runbook", meta={"team": "infra"})], ACME)
    await memory.add([fact("deploy checklist", meta={"team": "web"})], ACME)
    hits = await memory.search("deploy", ACME, k=10, filters={"team": "infra"})
    assert [h.text for h in hits] == ["deploy runbook"]


async def test_update_changes_the_text_and_only_within_the_owners_scope(memory):
    need(memory, "update")
    assert supports(memory, Updatable)
    a = (await memory.add([fact("Ann drinks coffee")], ACME)).ids[0]
    changed = await memory.update(a, "Ann drinks tea", ACME)
    assert changed.text == "Ann drinks tea"
    assert [h.text for h in await memory.search("tea", ACME)] == ["Ann drinks tea"]
    with pytest.raises(KeyError):
        await memory.update(a, "hijack", GLOBEX)  # another tenant cannot touch it


async def test_delete_removes_by_id_and_only_within_the_owners_scope(memory):
    need(memory, "delete")
    assert supports(memory, Deletable)
    a = (await memory.add([fact("Ann drinks coffee")], ACME)).ids[0]
    assert await memory.delete([a], GLOBEX) == 0  # another tenant cannot delete it
    assert await memory.search("coffee", ACME)
    assert await memory.delete([a], ACME) == 1
    assert await memory.search("coffee", ACME) == []


async def test_forget_removes_one_users_memories_and_list_shows_the_rest(memory):
    need(memory, "forget")
    assert supports(memory, Forgettable) and supports(memory, Listable)
    await memory.add([fact("Ann owns a bike")], ACME)
    await memory.add([fact("Bob drinks coffee")], ACME_BOB)
    assert len(await memory.list(ACME_BOB)) == 1
    assert await memory.forget(ACME) == 1  # forgets Ann's memory only
    assert await memory.search("bike", ACME) == []
    assert await memory.search("coffee", ACME_BOB)  # Bob is untouched


async def test_forgetting_a_tenant_removes_every_user_in_it(memory):
    need(memory, "forget")
    await memory.add([fact("one")], ACME)
    await memory.add([fact("two")], ACME_BOB)
    await memory.add([fact("three")], GLOBEX)
    assert await memory.forget(Scope(tenant="acme")) == 2
    assert await memory.search("three", GLOBEX)
    assert not await memory.search("one two", Scope(tenant="acme"), k=10)


def test_capabilities_are_reported_honestly():
    bare = CallableMemory(add=lambda e, s: None, search=lambda q, s, k: [])
    for cap in (Updatable, Deletable, Forgettable, Profiled, Listable):
        assert not isinstance(bare, cap)
    with pytest.raises(MissingMemoryCapability, match="Forgettable"):
        require(bare, Forgettable)
    assert isinstance(
        CallableMemory(add=lambda e, s: None, search=lambda q, s, k: [], forget=lambda s: 0),
        Forgettable,
    )


async def test_callable_memory_accepts_loose_return_shapes():
    stored: list[str] = []
    mem = CallableMemory(
        add=lambda entries, scope: stored.extend(e.text for e in entries),  # returns None
        search=lambda q, scope, k: [
            "plain string",
            {"memory": "dict form", "id": "m1", "score": 0.9, "source": "x"},
        ],
    )
    assert (await mem.add([fact("hello")], ACME)).ids == []
    assert stored == ["hello"]
    items = await mem.search("anything", ACME, k=5)
    assert [i.text for i in items] == ["plain string", "dict form"]
    assert items[1].id == "m1" and items[1].score == 0.9 and items[1].meta == {"source": "x"}
