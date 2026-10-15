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


@pytest.mark.parametrize(
    ("metric", "better"), [("cosine", True), ("dot", True), ("euclidean", False)]
)
class TestEmbeddedMetrics:
    """Score direction and magnitude for the embedded stores."""

    @staticmethod
    async def _check(st, metric, better):
        await st.upsert([Record("near", [1.0, 0.0, 0.0]), Record("far", [-1.0, 0.0, 0.0])])
        ms = await st.query([2.0, 0.0, 0.0], 2)
        assert st.higher_is_better is better and ms[0].id == "near"
        assert ms[0].score == pytest.approx(
            {"cosine": 1.0, "dot": 2.0, "euclidean": 1.0}[metric], abs=1e-4
        )
        assert (ms[0].score > ms[1].score) is better
        await st.aclose()

    async def test_sqlite_vec(self, metric, better):
        pytest.importorskip("sqlite_vec")
        from aaa.vector.sqlitevec import SqliteVecStore

        await self._check(SqliteVecStore(":memory:", dimension=DIM, metric=metric), metric, better)

    async def test_lancedb(self, metric, better, tmp_path):
        pytest.importorskip("lancedb")
        from aaa.vector.lancedb import LanceDBStore

        await self._check(LanceDBStore(str(tmp_path), dimension=DIM, metric=metric), metric, better)

    async def test_chroma(self, metric, better):
        pytest.importorskip("chromadb")
        from aaa.vector.chroma import ChromaStore

        await self._check(
            ChromaStore(dimension=DIM, metric=metric, prefix="m" + uuid.uuid4().hex[:8]),
            metric,
            better,
        )


class TestEmbeddedPersistence:
    async def test_sqlite_vec_file_survives_reopen(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        from aaa.vector.sqlitevec import SqliteVecStore

        a = SqliteVecStore(str(tmp_path / "v.db"), dimension=DIM)
        await a.upsert([Record("k", [1.0, 2.0, 3.0], "kept", {"n": {"deep": 1}})], namespace="ns")
        await a.aclose()
        b = SqliteVecStore(str(tmp_path / "v.db"), dimension=DIM)
        [r] = await b.get(["k"], namespace="ns")
        assert r.text == "kept" and r.meta == {"n": {"deep": 1}}
        await b.aclose()

    async def test_lancedb_directory_survives_reopen(self, tmp_path):
        pytest.importorskip("lancedb")
        from aaa.vector.lancedb import LanceDBStore

        a = LanceDBStore(str(tmp_path), dimension=DIM)
        await a.upsert([Record("k", [1.0, 2.0, 3.0], "kept", {"n": 1})], namespace="ns")
        await a.aclose()
        b = LanceDBStore(str(tmp_path), dimension=DIM)
        [r] = await b.get(["k"], namespace="ns")
        assert r.text == "kept" and r.meta == {"n": 1} and await b.count(namespace="ns") == 1
        await b.aclose()

    async def test_chroma_path_survives_reopen(self, tmp_path):
        pytest.importorskip("chromadb")
        from aaa.vector.chroma import ChromaStore

        a = ChromaStore(dimension=DIM, path=str(tmp_path))
        await a.upsert(
            [Record("k", [1.0, 2.0, 3.0], "kept", {"n": {"deep": 1}, "z": None})], namespace="ns"
        )
        await a.aclose()
        b = ChromaStore(dimension=DIM, path=str(tmp_path))
        [r] = await b.get(["k"], namespace="ns")
        assert r.text == "kept" and r.meta == {
            "n": {"deep": 1},
            "z": None,
        }  # exact metadata, nesting and null included
        await b.aclose()


class TestEmbeddedHostileInput:
    @pytest.mark.parametrize("backend", ["sqlitevec", "lancedb", "chroma"])
    async def test_odd_ids_namespaces_and_values_are_data(self, backend, tmp_path):
        evil = "'; DROP TABLE x; -- \" ünïcøde"
        if backend == "sqlitevec":
            pytest.importorskip("sqlite_vec")
            from aaa.vector.sqlitevec import SqliteVecStore

            st = SqliteVecStore(":memory:", dimension=DIM)
        elif backend == "lancedb":
            pytest.importorskip("lancedb")
            from aaa.vector.lancedb import LanceDBStore

            st = LanceDBStore(str(tmp_path), dimension=DIM)
        else:
            pytest.importorskip("chromadb")
            from aaa.vector.chroma import ChromaStore

            st = ChromaStore(dimension=DIM, prefix="h" + uuid.uuid4().hex[:8])
        await st.upsert([Record(evil, [1.0, 0.0, 0.0], evil, {"k": evil})], namespace=evil)
        ms = await st.query([1.0, 0.0, 0.0], 5, filter=F("k") == evil, namespace=evil)
        assert [m.id for m in ms] == [evil] and ms[0].text == evil
        assert await st.count(namespace=evil) == 1 and await st.count() == 0
        await st.delete([evil], namespace=evil)
        assert await st.count(namespace=evil) == 0
        await st.aclose()


class TestChromaRefusals:
    async def test_filters_chroma_cannot_express_fail_before_any_io(self):
        pytest.importorskip("chromadb")
        from aaa.vector import Exists, Not
        from aaa.vector.chroma import ChromaStore

        class Exploding:
            def __getattr__(self, name):
                raise AssertionError(f"touched the client ({name})")

        st = ChromaStore(dimension=DIM, client=Exploding())
        for bad in (F("name") >= "m", F("x") == None, Exists("x"), Not(F("n") > 3)):
            with pytest.raises(UnsupportedFilter):
                await st.query([1.0, 0.0, 0.0], 3, filter=bad)

    def test_negation_is_pushed_to_the_leaves(self):
        pytest.importorskip("chromadb")
        from aaa.vector import Not, Or
        from aaa.vector.chroma import to_where

        w = to_where(Not(Or((F("a") == 1, F("b").is_in(["x", "y"])))))
        assert w == {"$and": [{"a": {"$ne": 1}}, {"b": {"$nin": ["x", "y"]}}]}


class TestPinecone:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("pinecone")

    async def test_filters_pinecone_cannot_express_fail_before_any_io(self):
        from aaa.vector import Not
        from aaa.vector.pinecone import PineconeStore

        class Exploding:
            def __getattr__(self, name):
                raise AssertionError(f"touched the client ({name})")

        st = PineconeStore(index_host="x", dimension=DIM, client=Exploding())
        st._ix = Exploding()
        for bad in (F("name") >= "m", F("x") == None, Not(F("n") > 3)):
            with pytest.raises(UnsupportedFilter):
                await st.query([1.0, 0.0, 0.0], 3, filter=bad)

    async def test_oversized_records_are_rejected_before_the_network(self):
        from aaa.vector import VectorError
        from aaa.vector.pinecone import PineconeStore

        st = PineconeStore(index_host="x", dimension=DIM, client=object())
        with pytest.raises(VectorError, match="40 KB"):
            await st.upsert([Record("big", [1.0, 0.0, 0.0], "x" * 50_000)])

    def test_filter_compilation(self):
        from aaa.vector import Exists, Not, Or
        from aaa.vector.pinecone import to_filter

        assert to_filter(F("a") == 1) == {"a": 1}
        assert to_filter(F("a") != 1) == {"a": {"$ne": 1}}
        assert to_filter(Not(Exists("a"))) == {"a": {"$exists": False}}
        assert to_filter(F("t").is_in(["x", "y"])) == {"t": {"$in": ["x", "y"]}}
        assert to_filter(F("t").is_in(["x", 1])) == {
            "$or": [{"t": "x"}, {"t": 1}]
        }  # Pinecone rejects mixed-type $in
        assert to_filter(Not(Or((F("a") == 1, F("b") == 2)))) == {
            "$and": [{"a": {"$ne": 1}}, {"b": {"$ne": 2}}]
        }
        assert to_filter(F("n").is_in([])) == {"_aaa_never": {"$exists": True}}

    async def test_consistency_wait_polls_until_visible(self):
        from aaa.vector.pinecone import PineconeStore

        class Fetch:
            def __init__(self):
                self.calls = 0

            async def fetch(self, *, ids, namespace):
                self.calls += 1

                class R:
                    vectors = {i: object() for i in ids} if self.calls >= 3 else {}

                R.vectors = {i: object() for i in ids} if self.calls >= 3 else {}
                return R

            async def upsert(self, **kw):
                return None

        fake = Fetch()
        st = PineconeStore(index_host="x", dimension=DIM, client=object(), consistency_timeout_s=5)
        st._ix = fake
        await st.upsert([Record("a", [1.0, 0.0, 0.0])])
        assert fake.calls == 3  # not visible on the first two reads

        st2 = PineconeStore(
            index_host="x", dimension=DIM, client=object(), consistency_timeout_s=0.2
        )
        never = Fetch()
        never.calls = -10_000
        st2._ix = never
        from aaa.vector import VectorError

        with pytest.raises(VectorError, match="eventual consistency"):
            await st2.upsert([Record("a", [1.0, 0.0, 0.0])])

    def test_needs_an_index_name_or_host(self):
        from aaa.vector.pinecone import PineconeStore

        with pytest.raises(ValueError, match="index_name"):
            PineconeStore(dimension=DIM)
