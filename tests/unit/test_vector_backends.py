"""Backend specifics beyond the shared contract: metrics, persistence, injection safety, refusals."""

import uuid

import pytest

from aaa.vector import F, Record, UnsupportedFilter


def _no_pgvector(exc):
    """Skip where the extension is missing, but fail in the CI job that must have it."""
    import os

    if os.environ.get("AAA_REQUIRE_PGVECTOR"):
        raise AssertionError(f"pgvector is required here but unavailable: {exc}") from exc
    pytest.skip(f"pgvector extension not available: {exc}")


DIM = 3


@pytest.fixture
async def pg(pg_admin_dsn):
    asyncpg = pytest.importorskip("asyncpg")
    from aaa.vector.pgvector import PgVectorStore

    db = "b_" + uuid.uuid4().hex[:12]
    admin = await asyncpg.connect(pg_admin_dsn)
    await admin.execute(f'CREATE DATABASE "{db}"')
    dsn = pg_admin_dsn.rsplit("/", 1)[0] + "/" + db

    async def make(**kw):
        kw.setdefault("dimension", DIM)
        st = PgVectorStore(dsn, **kw)
        try:
            await st.setup()
        except asyncpg.PostgresError as exc:
            _no_pgvector(exc)
        return st

    made = []

    async def tracked(**kw):
        made.append(st := await make(**kw))
        return st

    yield tracked
    for st in made:
        await st.aclose()
    await admin.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
    await admin.close()


class TestPgVector:
    @pytest.mark.parametrize(
        ("metric", "better"), [("cosine", True), ("dot", True), ("euclidean", False)]
    )
    async def test_scores_have_the_documented_meaning(self, pg, metric, better):
        st = await pg(metric=metric, table=f"m_{metric}")
        await st.upsert([Record("near", [1.0, 0.0, 0.0]), Record("far", [-1.0, 0.0, 0.0])])
        ms = await st.query([2.0, 0.0, 0.0], 2)
        assert st.higher_is_better is better and ms[0].id == "near"
        assert ms[0].score == pytest.approx(
            {"cosine": 1.0, "dot": 2.0, "euclidean": 1.0}[metric], abs=1e-5
        )
        assert (ms[0].score > ms[1].score) is better

    async def test_data_outlives_the_store_object(self, pg):
        a = await pg(table="persist")
        await a.upsert([Record("k", [1.0, 2.0, 3.0], "kept", {"n": 1})], namespace="ns")
        b = await pg(table="persist")  # a fresh object on the same database
        [r] = await b.get(["k"], namespace="ns")
        assert r.text == "kept" and r.meta == {"n": 1}

    async def test_two_tables_do_not_interfere(self, pg):
        a, b = await pg(table="t_a"), await pg(table="t_b")
        await a.upsert([Record("1", [1.0, 0.0, 0.0])])
        assert await b.count() == 0

    @pytest.mark.parametrize("bad", ["x; DROP TABLE users", 'a"b', "1abc", "", "a" * 64, "a-b"])
    def test_table_names_are_validated(self, bad):
        from aaa.vector.pgvector import PgVectorStore

        with pytest.raises(ValueError, match="invalid table name"):
            PgVectorStore("postgresql://x/y", dimension=3, table=bad)

    async def test_hostile_values_are_data_not_sql(self, pg):
        st = await pg(table="hostile")
        evil = "'; DROP TABLE hostile; --"
        await st.upsert(
            [Record(evil, [1.0, 0.0, 0.0], evil, {"k": evil, 'q"uote': "v"})], namespace=evil
        )
        ms = await st.query([1.0, 0.0, 0.0], 5, filter=F("k") == evil, namespace=evil)
        assert [m.id for m in ms] == [evil] and ms[0].meta["k"] == evil
        assert (await st.query([1.0, 0.0, 0.0], 5, filter=F('q"uote') == "v", namespace=evil))[
            0
        ].id == evil
        assert await st.count(namespace=evil) == 1

    async def test_constructor_needs_exactly_one_connection_source(self):
        from aaa.vector.pgvector import PgVectorStore

        with pytest.raises(ValueError, match="exactly one"):
            PgVectorStore(dimension=3)

    async def test_an_existing_pool_is_used_and_left_open(self, pg_admin_dsn, pg):
        import asyncpg

        from aaa.vector.pgvector import PgVectorStore

        base = await pg(table="seed")  # creates the database and extension
        pool = await asyncpg.create_pool(base.dsn, min_size=1, max_size=2)
        try:
            st = PgVectorStore(pool=pool, dimension=DIM, table="viapool")
            await st.upsert([Record("1", [1.0, 0.0, 0.0])])
            await st.aclose()
            assert await pool.fetchval("SELECT count(*) FROM viapool") == 1  # still open
        finally:
            await pool.close()


class TestQdrant:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("qdrant_client")

    @pytest.mark.parametrize(
        ("metric", "better"), [("cosine", True), ("dot", True), ("euclidean", False)]
    )
    async def test_scores_have_the_documented_meaning(self, metric, better):
        from aaa.vector.qdrant import QdrantStore

        st = QdrantStore(
            location=":memory:", dimension=DIM, metric=metric, collection=f"m_{metric}"
        )
        await st.upsert([Record("near", [1.0, 0.0, 0.0]), Record("far", [-1.0, 0.0, 0.0])])
        ms = await st.query([2.0, 0.0, 0.0], 2)
        assert st.higher_is_better is better and ms[0].id == "near"
        assert ms[0].score == pytest.approx(
            {"cosine": 1.0, "dot": 2.0, "euclidean": 1.0}[metric], abs=1e-4
        )
        assert (ms[0].score > ms[1].score) is better
        await st.aclose()

    async def test_unsupported_filters_fail_before_any_io(self):
        from aaa.vector.qdrant import QdrantStore

        class Exploding:
            def __getattr__(self, name):
                raise AssertionError(f"touched the client ({name})")

        st = QdrantStore(client=Exploding(), dimension=DIM)
        with pytest.raises(UnsupportedFilter):
            await st.query([1.0, 0.0, 0.0], 3, filter=F("name") >= "m")

    async def test_ids_are_mapped_to_uuids_and_round_trip(self):
        from aaa.vector.qdrant import QdrantStore, point_id

        st = QdrantStore(location=":memory:", dimension=DIM, collection="ids")
        weird = "not-a-uuid / with spaces ünïcode"
        await st.upsert([Record(weird, [1.0, 0.0, 0.0], "t")], namespace="n")
        assert uuid.UUID(point_id("n", weird))
        assert point_id("n", weird) != point_id(
            "m", weird
        )  # same id in two namespaces never collides
        [r] = await st.get([weird], namespace="n")
        assert r.id == weird
        await st.aclose()

    async def test_metadata_cannot_clobber_store_fields(self):
        from aaa.vector.qdrant import QdrantStore

        st = QdrantStore(location=":memory:", dimension=DIM, collection="clobber")
        await st.upsert(
            [Record("a", [1.0, 0.0, 0.0], "t", {"_ns": "other", "_id": "zzz", "_text": "x"})],
            namespace="mine",
        )
        [m] = await st.query([1.0, 0.0, 0.0], 5, namespace="mine")
        assert m.id == "a" and m.text == "t" and m.meta["_ns"] == "other"
        assert await st.query([1.0, 0.0, 0.0], 5, namespace="other") == []
        await st.aclose()
