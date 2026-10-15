"""Every vector store must behave the same. The in-memory store is the oracle; real backends join the parametrization.

`FEATURES` records what each backend does NOT provide, and `need()` skips (never silently passes) those checks."""

import math

import pytest

from aaa.vector import (
    And,
    DimensionMismatch,
    Eq,
    Exists,
    F,
    Filter,
    InMemoryVectorStore,
    Not,
    Or,
    Record,
    TenantVectorStore,
    VectorError,
    matches,
)
from aaa.vector.types import DeleteWhere, DropNamespace


def _no_pgvector(exc):
    """Skip where the extension is missing, but fail in the CI job that must have it."""
    import os

    if os.environ.get("AAA_REQUIRE_PGVECTOR"):
        raise AssertionError(f"pgvector is required here but unavailable: {exc}") from exc
    pytest.skip(f"pgvector extension not available: {exc}")


DIM = 4
FEATURES: dict[str, set[str]] = {
    "memory": set(),
    "pgvector": set(),
    "qdrant": {"string-range", "raw-vectors"},
    "sqlitevec": set(),
    "lancedb": set(),
    "chroma": {"string-range", "null-values", "exists"},
}


def need(request_param: str, feature: str) -> None:
    if feature in FEATURES.get(request_param, set()):
        pytest.skip(f"{request_param} does not provide {feature}")


@pytest.fixture(params=["memory", "pgvector", "qdrant", "sqlitevec", "chroma", "lancedb"])
def backend(request):
    """Sync fixture: the session-scoped Postgres admin DSN can be requested dynamically from here, but not from an async fixture."""
    return request.param, (
        request.getfixturevalue("pg_admin_dsn") if request.param == "pgvector" else None
    )


@pytest.fixture
async def store(backend):
    name, admin_dsn = backend
    if name == "memory":
        yield InMemoryVectorStore(DIM, "cosine")
        return
    if name == "lancedb":
        pytest.importorskip("lancedb")
        import tempfile

        from aaa.vector.lancedb import LanceDBStore

        with tempfile.TemporaryDirectory() as d:
            st = LanceDBStore(
                d, dimension=DIM, initial_batch=8
            )  # small first batch so the refine loop is exercised
            yield st
            await st.aclose()
        return
    if name == "chroma":
        pytest.importorskip("chromadb")
        import uuid as _uuid

        from aaa.vector.chroma import ChromaStore

        st = ChromaStore(dimension=DIM, prefix="c" + _uuid.uuid4().hex[:10])
        yield st
        await st.aclose()
        return
    if name == "sqlitevec":
        pytest.importorskip("sqlite_vec")
        from aaa.vector.sqlitevec import SqliteVecStore

        st = SqliteVecStore(":memory:", dimension=DIM)
        yield st
        await st.aclose()
        return
    if name == "qdrant":
        pytest.importorskip("qdrant_client")
        from aaa.vector.qdrant import QdrantStore

        st = QdrantStore(location=":memory:", dimension=DIM, collection="contract")
        yield st
        await st.aclose()
        return
    import uuid

    import asyncpg

    from aaa.vector.pgvector import PgVectorStore

    db = "v_" + uuid.uuid4().hex[:12]
    admin = await asyncpg.connect(admin_dsn)
    await admin.execute(f'CREATE DATABASE "{db}"')
    st = PgVectorStore(
        admin_dsn.rsplit("/", 1)[0] + "/" + db,
        dimension=DIM,
        metric="cosine",
        table="contract_vectors",
    )
    try:
        await st.setup()
    except asyncpg.PostgresError as exc:
        await admin.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
        await admin.close()
        _no_pgvector(exc)
    yield st
    await st.aclose()
    await admin.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
    await admin.close()


def vec(i: int) -> list[float]:
    a = i * 0.37
    return [math.cos(a), math.sin(a), math.cos(2 * a), 0.5]


META = [
    {
        "lang": "en",
        "year": 2019,
        "tag": "a",
        "score": 0.2,
        "flag": True,
        "author": {"country": "PT"},
    },
    {
        "lang": "en",
        "year": 2021,
        "tag": "b",
        "score": 0.7,
        "flag": False,
        "author": {"country": "US"},
    },
    {"lang": "fr", "year": 2021, "tag": "a", "score": 0.9, "author": {"country": "FR"}},
    {"lang": "fr", "year": 2023, "tag": "c", "score": 0.4},
    {
        "lang": "de",
        "year": "2021",
        "tag": "b",
        "flag": True,
    },  # year is a STRING here: 2021 must not match it
    {"year": 2020, "tag": None, "score": 1.0},
    {"lang": "en", "title": "2021", "year": 2022},
    {},
]


def corpus() -> list[Record]:
    return [Record(f"r{i}", vec(i), f"text {i}", dict(m)) for i, m in enumerate(META)]


FILTERS: list[tuple[str, Filter]] = [
    ("eq-str", F("lang") == "en"),
    ("eq-int", F("year") == 2021),
    ("eq-bool", F("flag") == True),
    ("eq-null", Eq("tag", None)),
    ("ne", F("lang") != "en"),
    ("in", F("tag").is_in(["a", "b"])),
    ("range-int", And((F("year") >= 2020, F("year") < 2023))),
    ("range-float", F("score") > 0.5),
    ("range-str", F("lang") >= "en"),
    ("exists", F("score").exists()),
    ("not-exists", ~F("score").exists()),
    ("and", (F("lang") == "en") & (F("year") >= 2021)),
    ("or", (F("lang") == "fr") | (F("tag") == "b")),
    ("not", Not(Or((F("lang") == "en", F("lang") == "fr")))),
    ("nested", F("author.country") == "PT"),
    ("nested-or", Or((F("author.country") == "PT", F("author.country") == "FR"))),
    ("empty-result", F("lang") == "xx"),
]


FILTER_FEATURE = {
    "range-str": "string-range",
    "eq-null": "null-values",
    "exists": "exists",
    "not-exists": "exists",
}


async def ids_for(store, f, **kw):
    ms = await store.query(vec(0), 100, filter=f, **kw)
    return {m.id for m in ms}


# -- basics -----------------------------------------------------------------------------------
async def test_nearest_first_and_score_direction(store):
    await store.upsert(corpus())
    ms = await store.query(vec(3), 3)
    assert ms[0].id == "r3" and len(ms) == 3
    scores = [m.score for m in ms]
    assert scores == sorted(scores, reverse=store.higher_is_better)
    assert ms[0].text == "text 3" and ms[0].meta == META[3]


async def test_get_roundtrips_text_meta_and_vector(store, backend):
    await store.upsert(corpus())
    [r] = await store.get(["r1"])
    assert r.id == "r1" and r.text == "text 1" and r.meta == META[1]
    want = vec(1)
    if (
        "raw-vectors" in FEATURES[backend[0]]
    ):  # cosine stores that keep unit vectors: compare directions
        n = math.sqrt(sum(x * x for x in want))
        want = [x / n for x in want]
    assert r.vector == pytest.approx(want, abs=1e-5)
    assert await store.get(["nope"]) == []


async def test_upsert_replaces_and_delete_removes(store):
    await store.upsert(corpus())
    n = await store.count()
    await store.upsert([Record("r0", vec(40), "new text", {"lang": "xx"})])
    assert await store.count() == n
    [r] = await store.get(["r0"])
    assert r.text == "new text" and r.meta == {"lang": "xx"}
    await store.delete(["r0", "r1", "missing"])
    assert await store.count() == n - 2
    assert await store.get(["r0", "r1"]) == []


async def test_namespaces_are_hard_partitions(store):
    await store.upsert([Record("x", vec(1), "a")], namespace="acme")
    await store.upsert([Record("x", vec(2), "b")], namespace="globex")
    assert [m.text for m in await store.query(vec(1), 5, namespace="acme")] == ["a"]
    assert [m.text for m in await store.query(vec(1), 5, namespace="globex")] == ["b"]
    assert await store.query(vec(1), 5, namespace="nobody") == []
    await store.delete(["x"], namespace="acme")
    assert await store.count(namespace="acme") == 0 and await store.count(namespace="globex") == 1


async def test_dimension_is_checked_on_upsert_and_query(store):
    await store.upsert(corpus()[:1])
    with pytest.raises(DimensionMismatch):
        await store.upsert([Record("bad", [1.0, 2.0], "x")])
    with pytest.raises(DimensionMismatch):
        await store.query([1.0, 2.0], 3)


# -- filters ----------------------------------------------------------------------------------
@pytest.mark.parametrize(("name", "f"), FILTERS, ids=[n for n, _ in FILTERS])
async def test_filters_agree_with_the_reference_semantics(store, backend, name, f):
    await store.upsert(corpus())
    feature = FILTER_FEATURE.get(name)
    if feature and feature in FEATURES[backend[0]]:
        from aaa.vector import UnsupportedFilter

        with pytest.raises(UnsupportedFilter):  # refused loudly, never approximated
            await ids_for(store, f)
        return
    want = {f"r{i}" for i, m in enumerate(META) if matches(f, m)}
    assert await ids_for(store, f) == want, name


async def test_a_selective_filter_still_returns_everything_it_matches(store):
    """Filtering must happen BEFORE the top-k cut: 3 matching records among 300, all far from the query."""
    recs = [Record(f"n{i}", vec(i), "noise", {"kind": "noise"}) for i in range(300)]
    recs += [
        Record(f"t{i}", [-vec(0)[0], -vec(0)[1], -vec(0)[2], 0.5], "target", {"kind": "rare"})
        for i in range(3)
    ]
    await store.upsert(recs)
    ms = await store.query(vec(0), 5, filter=F("kind") == "rare")
    assert {m.id for m in ms} == {"t0", "t1", "t2"}


async def test_dict_filters_parse(store):
    await store.upsert(corpus())
    f = Filter.parse({"lang": "en", "year": {"$gte": 2021}})
    assert await ids_for(store, f) == {"r1", "r6"}
    assert Filter.parse({"$or": [{"lang": "fr"}, {"tag": "c"}]}) is not None


# -- deletion capabilities --------------------------------------------------------------------
async def test_delete_where_and_drop_namespace(store, request):
    if not isinstance(store, DeleteWhere):
        pytest.skip("no delete_where")
    await store.upsert(corpus())
    n = await store.delete_where(F("lang") == "en")
    assert n == 3 and "r0" not in await ids_for(store, None)
    if isinstance(store, DropNamespace):
        await store.upsert([Record("z", vec(1), "z")], namespace="gone")
        await store.drop_namespace("gone")
        assert await store.count(namespace="gone") == 0


# -- tenant wrapper ---------------------------------------------------------------------------
async def test_tenant_view_cannot_see_or_touch_another_tenants_rows(store, backend):
    acme, globex = TenantVectorStore(store, "acme"), TenantVectorStore(store, "globex")
    await acme.upsert([Record("a1", vec(1), "acme secret", {"lang": "en"})])
    await globex.upsert([Record("g1", vec(1), "globex secret", {"lang": "en"})])

    assert [m.text for m in await acme.query(vec(1), 10)] == ["acme secret"]
    # an 'or' that would match every row must still be walled in
    wide = Or(
        (F("lang") == "en", F("lang") != "en")
        + (() if "exists" in FEATURES[backend[0]] else (Exists("lang"),))
    )
    assert [m.text for m in await acme.query(vec(1), 10, filter=wide)] == ["acme secret"]
    assert [
        m.text
        for m in await acme.query(
            vec(1), 10, filter=Or((Eq("tenant", "globex"), F("lang") == "en"))
        )
    ] == ["acme secret"]

    assert await acme.get(["g1"]) == []
    await acme.delete(["g1"])  # an id that is not this tenant's is not deleted
    assert [m.text for m in await globex.query(vec(1), 10)] == ["globex secret"]

    with pytest.raises(VectorError):
        await acme.query(vec(1), 5, namespace="globex")
    with pytest.raises(VectorError):
        await acme.upsert([Record("evil", vec(1), "x", {"tenant": "globex"})])
    assert await acme.count() == 1 and await globex.count() == 1


async def test_tenant_erase_removes_only_that_tenant(store):
    acme, globex = TenantVectorStore(store, "acme"), TenantVectorStore(store, "globex")
    await acme.upsert([Record("a1", vec(1), "a")])
    await globex.upsert([Record("g1", vec(2), "g")])
    if not isinstance(store, DeleteWhere | DropNamespace):
        pytest.skip("store has neither delete_where nor drop_namespace")
    await acme.erase()
    assert await acme.count() == 0 and await globex.count() == 1
