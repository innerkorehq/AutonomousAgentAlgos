"""Hybrid search (similarity + keyword, fused by reciprocal rank) on every store that provides it."""

import math

import pytest
from test_vector_contract import DIM, backend, store  # noqa: F401  (fixtures: the same real backends as the contract suite)

from aaa.data.index import Doc
from aaa.vector import (
    EmbeddingIndex,
    F,
    HybridSearch,
    HybridUnavailable,
    InMemoryVectorStore,
    Record,
    TenantVectorStore,
)


def vec(i: int) -> list[float]:
    a = i * 0.37
    return [math.cos(a), math.sin(a), math.cos(2 * a), 0.5]


FAR = [-x for x in vec(0)]
CORPUS = [
    Record("code", FAR, "error code ZX4411 occurs when the pump overheats", {"lang": "en"}),
    Record("near1", vec(0), "the pump is quiet and runs smoothly", {"lang": "en"}),
    Record("near2", vec(1), "cooling fans keep the unit at a stable temperature", {"lang": "fr"}),
    Record(
        "near3", vec(2), "regular maintenance extends the life of the equipment", {"lang": "en"}
    ),
    Record("near4", vec(3), "the manual describes installation steps", {"lang": "fr"}),
]


@pytest.fixture
async def hs(store, backend):  # noqa: F811
    if not isinstance(store, HybridSearch):
        pytest.skip(f"{backend[0]} does not provide hybrid search")
    await store.upsert(CORPUS)
    return store


async def ids(coro):
    return [m.id for m in await coro]


async def test_an_exact_term_that_similarity_misses_is_found(hs):
    dense = await ids(hs.query(vec(0), 3))
    assert "code" not in dense  # the vector for this record is on the other side of the space
    fused = await ids(hs.query_hybrid(vec(0), "ZX4411", 3))
    assert "code" in fused and fused[0] in (
        "code",
        "near1",
    )  # present, and ranked with the best dense hit
    scores = [m.score for m in await hs.query_hybrid(vec(0), "ZX4411", 3)]
    assert scores == sorted(scores, reverse=True) and len(scores) == 3


async def test_a_query_without_keyword_matches_behaves_like_dense_search(hs):
    assert await ids(hs.query_hybrid(vec(0), "qqqzzz", 3)) == (await ids(hs.query(vec(0), 3)))[:3]
    assert await ids(hs.query_hybrid(vec(0), "", 3)) == (await ids(hs.query(vec(0), 3)))[:3]


async def test_filters_apply_to_both_legs(hs):
    got = await ids(hs.query_hybrid(vec(0), "ZX4411", 5, filter=F("lang") == "fr"))
    assert "code" not in got and set(got) <= {"near2", "near4"}


async def test_namespaces_are_still_walls(hs):
    assert await hs.query_hybrid(vec(0), "ZX4411", 5, namespace="elsewhere") == []
    await hs.upsert([Record("other", vec(0), "ZX4411 in another namespace")], namespace="elsewhere")
    assert await ids(hs.query_hybrid(vec(0), "ZX4411", 5, namespace="elsewhere")) == ["other"]
    assert "other" not in await ids(hs.query_hybrid(vec(0), "ZX4411", 5))


async def test_the_keyword_index_follows_updates_and_deletes(hs):
    await hs.upsert(
        [Record("code", FAR, "this record no longer mentions the term", {"lang": "en"})]
    )  # replace
    assert "code" not in await ids(hs.query_hybrid(vec(0), "ZX4411", 3))
    await hs.upsert([Record("new", FAR, "fresh ZX4411 note", {})])
    assert "new" in await ids(hs.query_hybrid(vec(0), "ZX4411", 3))
    await hs.delete(["new"])
    assert "new" not in await ids(hs.query_hybrid(vec(0), "ZX4411", 3))
    n = await hs.count()
    await hs.upsert([Record("tmp", FAR, "temporary ZX4411", {"tag": "x"})])
    await hs.delete_where(F("tag") == "x")
    assert "tmp" not in await ids(hs.query_hybrid(vec(0), "ZX4411", 3)) and await hs.count() == n


@pytest.mark.parametrize(
    "q",
    [
        'ZX4411 AND OR NOT "(" * : NEAR(a b)',
        "ZX4411'; DROP TABLE x; --",
        'zx4411 "unbalanced',
        "ZX4411 ^ - + ~ { } [ ]",
    ],
)
async def test_query_text_is_never_interpreted_as_search_syntax(hs, q):
    assert "code" in await ids(hs.query_hybrid(vec(0), q, 5))


async def test_tenant_views_get_hybrid_and_stay_walled(hs):
    acme, globex = TenantVectorStore(hs, "acme"), TenantVectorStore(hs, "globex")
    await acme.upsert([Record("a", FAR, "acme ZX4411 secret")])
    await globex.upsert([Record("g", FAR, "globex ZX4411 secret")])
    assert await ids(acme.query_hybrid(vec(0), "ZX4411", 5)) == ["a"]
    assert await ids(globex.query_hybrid(vec(0), "secret", 5)) == ["g"]


# -- through EmbeddingIndex ---------------------------------------------------------------------
class OneHot:
    """An embedder where unrelated texts are far apart: ranking by meaning alone cannot find a rare token."""

    async def embed(self, texts):
        return [[1.0 if (len(t) + i) % 4 == 0 else 0.1 for i in range(4)] for t in texts]


async def test_embedding_index_hybrid_modes():
    store_ = InMemoryVectorStore(4)
    idx_off = EmbeddingIndex(store_, OneHot())
    await idx_off.upsert(
        [Doc(id=str(i), text=f"filler document number {i} " + "x" * i) for i in range(12)]
        + [Doc(id="rare", text="needle ZX4411 here")]
    )
    off = await idx_off.search("ZX4411", k=3)
    on = await EmbeddingIndex(store_, OneHot(), hybrid=True).search("ZX4411", k=3)
    auto = await idx_off.search("ZX4411", k=3, hybrid="auto")
    assert "rare" in [h.doc.id for h in on] and [h.doc.id for h in auto] == [h.doc.id for h in on]
    assert on[0].metric == "rrf" and on[0].higher_is_better and off[0].metric == "cosine"


async def test_hybrid_true_is_an_error_where_the_store_cannot_but_auto_falls_back():
    class DenseOnly:
        metric, dimension = "cosine", 4
        higher_is_better = True

        def __init__(self):
            self.inner = InMemoryVectorStore(4)

        def __getattr__(self, n):
            if n == "query_hybrid":
                raise AttributeError(n)
            return getattr(self.inner, n)

    s = DenseOnly()
    idx = EmbeddingIndex(s, OneHot())  # type: ignore[arg-type]
    await idx.upsert([Doc(id="a", text="hello")])
    with pytest.raises(HybridUnavailable):
        await idx.search("hello", hybrid=True)
    assert [h.doc.id for h in await idx.search("hello", hybrid="auto")] == ["a"]


async def test_predicate_filters_work_with_hybrid():
    store_ = InMemoryVectorStore(4)
    idx = EmbeddingIndex(store_, OneHot(), hybrid=True)
    await idx.upsert(
        [
            Doc(id="a", text="alpha ZX4411", meta={"k": 1}),
            Doc(id="b", text="beta ZX4411", meta={"k": 2}),
        ]
    )
    assert [
        h.doc.id for h in await idx.search("ZX4411", k=5, filter=lambda d: d.meta["k"] == 2)
    ] == ["b"]
