"""The vector layer around the stores: EmbeddingIndex as a drop-in Index, overfetching, rerank, and per-backend specifics."""

import json
import math
from typing import ClassVar

import pytest

from aaa.data.index import Doc
from aaa.llm import FakeLLMClient
from aaa.llm.fake import text
from aaa.memory import MemoryEntry, NativeMemory, Scope
from aaa.vector import (
    EmbeddingIndex,
    F,
    Filter,
    InMemoryVectorStore,
    LLMReranker,
    Record,
    RerankingIndex,
    TenantVectorStore,
)
from aaa.vector.filters import And, Eq, In, Not, Or, Range


class WordEmbedder:
    """Deterministic bag-of-words embedder over a tiny vocabulary: semantically 'close' texts share words."""

    VOCAB: ClassVar[list[str]] = ["cat", "dog", "fish", "paris", "rome", "food", "tax", "law"]

    async def embed(self, texts):
        out = []
        for t in texts:
            words = t.lower().split()
            out.append([float(words.count(w)) + 0.01 for w in self.VOCAB])
        return out


def index(store=None, **kw):
    return EmbeddingIndex(
        store or InMemoryVectorStore(len(WordEmbedder.VOCAB)), WordEmbedder(), **kw
    )


async def seeded(**kw):
    idx = index(**kw)
    await idx.upsert(
        [
            Doc(id="1", text="cat cat food", meta={"kind": "pet", "year": 2020}),
            Doc(id="2", text="dog food", meta={"kind": "pet", "year": 2022}),
            Doc(id="3", text="paris rome", meta={"kind": "city", "year": 2021}),
            Doc(id="4", text="tax law", meta={"kind": "legal", "year": 2023}),
        ]
    )
    return idx


# -- filters ----------------------------------------------------------------------------------
def test_filter_builders_and_dict_parsing():
    assert isinstance(F("a") == 1, Eq)
    assert Filter.parse({"a": 1, "b": {"$gte": 2, "$lt": 5}}) == And(
        (Eq("a", 1), Range("b", gte=2), Range("b", lt=5))
    )
    assert Filter.parse({"$or": [{"a": 1}, {"b": 2}]}) == Or((Eq("a", 1), Eq("b", 2)))
    assert Filter.parse({"t": {"$in": ["x", "y"]}}) == In("t", ("x", "y"))
    assert Filter.parse({"$not": {"a": 1}}) == Not(Eq("a", 1))
    assert Filter.parse(None) is None


# -- EmbeddingIndex ---------------------------------------------------------------------------
async def test_embedding_index_is_a_drop_in_index():
    from aaa.data.index import Index

    idx = await seeded()
    assert isinstance(idx, Index) and await idx.count() == 4
    hits = await idx.search("cat food", k=2)
    assert (
        next(h.doc.id for h in hits) == "1"
        and hits[0].metric == "cosine"
        and hits[0].higher_is_better
    )
    assert hits[0].doc.text == "cat cat food" and hits[0].doc.meta["kind"] == "pet"
    await idx.delete(["1"])
    assert await idx.count() == 3


async def test_filters_push_down_and_predicates_overfetch_to_still_return_k():
    idx = index()
    # 40 'noise' docs closer to the query than the 3 'rare' ones
    await idx.upsert(
        [Doc(id=f"n{i}", text="cat cat cat", meta={"kind": "noise"}) for i in range(40)]
    )
    await idx.upsert([Doc(id=f"r{i}", text="fish", meta={"kind": "rare"}) for i in range(3)])
    pushed = await idx.search("cat cat cat", k=3, filter=F("kind") == "rare")
    assert {h.doc.id for h in pushed} == {"r0", "r1", "r2"}
    predicate = await idx.search("cat cat cat", k=3, filter=lambda d: d.meta["kind"] == "rare")
    assert {h.doc.id for h in predicate} == {
        "r0",
        "r1",
        "r2",
    }  # found despite 40 closer non-matches
    assert {h.doc.id for h in await idx.search("cat cat cat", k=3, filter={"kind": "rare"})} == {
        "r0",
        "r1",
        "r2",
    }


async def test_precomputed_embeddings_are_not_recomputed():
    class Counting(WordEmbedder):
        n = 0

        async def embed(self, texts):
            Counting.n += len(texts)
            return await super().embed(texts)

    idx = EmbeddingIndex(InMemoryVectorStore(8), Counting())
    await idx.upsert([Doc(id="a", text="cat", embedding=[1.0] * 8)])
    assert Counting.n == 0
    await idx.search("cat")
    assert Counting.n == 1


async def test_namespaced_indexes_share_a_store_without_seeing_each_other():
    store = InMemoryVectorStore(len(WordEmbedder.VOCAB))
    a, b = index(store, namespace="a"), index(store, namespace="b")
    await a.upsert([Doc(id="1", text="cat")])
    await b.upsert([Doc(id="1", text="dog")])
    assert [h.doc.text for h in await a.search("cat")] == ["cat"]
    assert [h.doc.text for h in await b.search("cat")] == ["dog"]


async def test_native_memory_runs_on_a_persistent_style_index():
    mem = NativeMemory(index=index())
    scope = Scope(tenant="acme", user="ann")
    await mem.add([MemoryEntry(text="ann likes cat food")], scope)
    await mem.add([MemoryEntry(text="bob likes dog food")], Scope(tenant="acme", user="bob"))
    got = await mem.search("food", scope)
    assert [i.text for i in got] == [
        "ann likes cat food"
    ]  # the scope filter (a predicate) works through overfetch


# -- rerank -----------------------------------------------------------------------------------
async def test_llm_reranker_reorders_and_survives_garbage():
    idx = await seeded()
    order = RerankingIndex(
        idx, LLMReranker(FakeLLMClient(lambda r: text(json.dumps({"order": [1, 0]})))), fetch_k=2
    )
    first = [h.doc.id for h in await idx.search("cat food dog", k=2)]
    reranked = [h.doc.id for h in await order.search("cat food dog", k=2)]
    assert reranked == first[::-1]
    broken = RerankingIndex(idx, LLMReranker(FakeLLMClient(lambda r: text("no idea"))), fetch_k=2)
    assert [h.doc.id for h in await broken.search("cat food dog", k=2)] == first


# -- tenant -----------------------------------------------------------------------------------
async def test_tenant_view_works_under_an_embedding_index():
    store = InMemoryVectorStore(len(WordEmbedder.VOCAB))
    acme = EmbeddingIndex(TenantVectorStore(store, "acme"), WordEmbedder())
    globex = EmbeddingIndex(TenantVectorStore(store, "globex"), WordEmbedder())
    await acme.upsert([Doc(id="1", text="cat")])
    await globex.upsert([Doc(id="1", text="cat secret")])
    assert [h.doc.text for h in await acme.search("cat", k=10)] == ["cat"]
    assert [h.doc.text for h in await globex.search("cat", k=10)] == ["cat secret"]


# -- score semantics --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("metric", "better"), [("cosine", True), ("dot", True), ("euclidean", False)]
)
async def test_in_memory_metrics(metric, better):
    st = InMemoryVectorStore(2, metric)
    await st.upsert([Record("near", [1.0, 0.0]), Record("far", [-1.0, 0.0])])
    ms = await st.query([2.0, 0.0], 2)
    assert st.higher_is_better is better and ms[0].id == "near"
    assert (ms[0].score > ms[1].score) is better
    if metric == "euclidean":
        assert ms[0].score == pytest.approx(1.0)
    if metric == "dot":
        assert ms[0].score == pytest.approx(2.0)
    if metric == "cosine":
        assert ms[0].score == pytest.approx(1.0) and math.isclose(ms[1].score, -1.0)
