"""The stand-ins in tests/helpers/memory_stubs.py are only worth anything if they match the real
libraries. This replays every call an adapter made on a stub against the REAL signature of the
method it targets. Skipped for libraries that are not installed; the `adapters` CI job installs
them, so a breaking change in Mem0, Graphiti or LangGraph fails CI instead of failing production."""

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa.memory import MemoryEntry, Scope
from aaa.memory.graphiti import GraphitiMemory
from aaa.memory.langgraph import LangGraphStoreMemory
from aaa.memory.mem0 import Mem0Memory
from helpers.memory_stubs import StubGraphiti, StubMem0, StubMem0Hosted, StubStore, graphiti_ops

ACME = Scope(tenant="acme", user="ann")
WIDE = Scope(tenant="acme")


async def exercise(memory) -> None:
    ids = (await memory.add([MemoryEntry(text="Ann likes jazz")], ACME)).ids
    await memory.add(
        [MemoryEntry(text="Bob likes tea", role="user")], Scope(tenant="acme", user="bob")
    )
    await memory.search("jazz", ACME, k=3)
    await memory.search("likes", WIDE, k=3)
    await memory.list(ACME, limit=10)
    if hasattr(memory, "update"):
        await memory.update(ids[0], "Ann likes blues", ACME)
    await memory.delete(ids, ACME)
    await memory.forget(WIDE)


def check_call(
    real_name: str, sig: inspect.Signature, bound_args: tuple, kwargs: dict, *, strict: bool
) -> None:
    try:
        sig.bind(*bound_args, **kwargs)
    except TypeError as e:
        raise AssertionError(f"{real_name}{sig} cannot take {bound_args} {kwargs}: {e}") from e
    if strict:
        # A `**kwargs` catch-all would accept a misspelt or renamed argument silently (and the real
        # library then ignores or rejects it at runtime), so every keyword must be a named parameter.
        unnamed = [k for k in kwargs if k not in sig.parameters]
        assert not unnamed, f"{real_name}{sig}: {unnamed} are only absorbed by **kwargs"


def assert_binds(real: type, calls, *, strict: bool = True) -> None:
    checked = 0
    for name, args, kwargs in calls:
        check_call(
            f"{real.__name__}.{name}",
            inspect.signature(getattr(real, name)),
            (None, *args),
            kwargs,
            strict=strict,
        )
        checked += 1
    assert checked, "the scenario made no calls to check"


async def test_mem0_open_source_calls_match_the_real_signatures():
    mem0 = pytest.importorskip("mem0")
    stub = StubMem0()
    await exercise(Mem0Memory(stub))
    assert_binds(mem0.AsyncMemory, stub.calls)


async def test_mem0_hosted_calls_match_the_real_signatures_and_option_models():
    mem0 = pytest.importorskip("mem0")
    types = pytest.importorskip("mem0.client.types")
    stub = StubMem0Hosted()
    await exercise(Mem0Memory(stub))
    # the hosted client takes its options as **kwargs by design; the option models below pin their names
    assert_binds(mem0.AsyncMemoryClient, stub.calls, strict=False)
    options = {
        "add": types.AddMemoryOptions,
        "search": types.SearchMemoryOptions,
        "get_all": types.GetAllMemoryOptions,
        "update": types.UpdateMemoryOptions,
        "delete_all": types.DeleteAllMemoryOptions,
    }
    for (
        name,
        _args,
        kwargs,
    ) in stub.calls:  # the client forwards **kwargs: they must be real options
        if name in options:
            unknown = set(kwargs) - set(options[name].model_fields)
            assert not unknown, f"hosted Mem0 {name}() has no option(s) {sorted(unknown)}"


async def test_graphiti_calls_match_the_real_signatures(monkeypatch):
    pytest.importorskip("graphiti_core")
    from graphiti_core import Graphiti
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EpisodeType, EpisodicNode

    g = StubGraphiti()
    monkeypatch.setattr("aaa.memory.graphiti._ops", lambda: graphiti_ops(g))
    await exercise(GraphitiMemory(g))
    for name, args, kwargs in g.calls:
        if name == "execute_query":
            continue  # the driver's own Cypher entry point; its shape varies by backend
        owner, _, method = name.rpartition(".")
        real = {"": Graphiti, "EntityEdge": EntityEdge, "EpisodicNode": EpisodicNode}[owner]
        bound = (None, *args) if owner == "" else args  # classmethods are already bound
        check_call(
            f"{real.__name__}.{method}",
            inspect.signature(getattr(real, method)),
            bound,
            kwargs,
            strict=True,
        )
    assert {EpisodeType.message.value, EpisodeType.text.value} == {"message", "text"}


async def test_langgraph_store_calls_match_the_real_signatures():
    base = pytest.importorskip("langgraph.store.base")
    store = StubStore()
    await exercise(LangGraphStoreMemory(store))
    assert_binds(base.BaseStore, store.calls)


async def test_the_adapter_works_on_langgraphs_real_in_memory_store():
    """Beyond signatures: LangGraph's own InMemoryStore (no embedding index) behind the adapter."""
    memory_store = pytest.importorskip("langgraph.store.memory")
    memory = LangGraphStoreMemory(memory_store.InMemoryStore())
    await memory.add([MemoryEntry(text="Ann is vegetarian")], ACME)
    await memory.add([MemoryEntry(text="Bob eats fish")], Scope(tenant="acme", user="bob"))
    assert [m.text for m in await memory.search("vegetarian food", ACME)] == ["Ann is vegetarian"]
    assert len(await memory.search("eats vegetarian fish", WIDE, k=10)) == 2
    assert await memory.forget(ACME) == 1
    assert await memory.search("vegetarian", ACME) == []
