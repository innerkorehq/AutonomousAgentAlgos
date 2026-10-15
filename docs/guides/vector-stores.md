# Persistent vector search

`aaa.data.InMemoryIndex` is fine for tests and small corpora; anything bigger, or anything that must survive a restart, needs a vector store.
AAA splits the job in two so each part stays small:

- a **`VectorStore`** holds vectors with a payload (text and metadata) and finds the nearest ones. It never embeds text.
- an **`EmbeddingIndex(store, embedder)`** adds the text side and implements the `Index` protocol, so `RAGAgent`, `search_tool` and
  `NativeMemory` run on a persistent store without changing.

```python
--8<-- "examples/vector_search.py"
```

```text
all: ['a', 'd', 'b', 'c'] (scores are cosine, higher is better: True)
english, 2023 or later: ['a', 'c']
acme sees: ['acme invoice terms']
```

## Stores

| Store | Install | Notes |
|---|---|---|
| `InMemoryVectorStore` | none | exact, brute force; the reference the shared tests compare the others to |
| `PgVectorStore(dsn, dimension=, metric=, table=)` | `pip install "autonomous-agent-algos[pgvector]"` + `CREATE EXTENSION vector` | one table, HNSW index, jsonb metadata, filters become SQL |
| `QdrantStore(url=… \| location=":memory:", dimension=)` | `pip install "autonomous-agent-algos[qdrant]"` | server, cloud, or in-process |
| `SqliteVecStore("vectors.db", dimension=)` | `[sqlitevec]` | one file; brute force and exact; every filter runs in SQL before the top-k cut; needs a Python whose `sqlite3` can load extensions |
| `LanceDBStore("./lance", dimension=)` | `[lancedb]` | embedded directory or object storage; metadata filters are exact but evaluated client-side over growing candidate batches (see below) |
| `PineconeStore(index_name=… \| index_host=…, dimension=)` | `[pinecone]` | Pinecone's own namespaces; metadata flat with a 40 KB limit; writes wait for visibility (`consistent=True`); no `delete_where` (serverless cannot) |
| `ChromaStore(path=… \| host=…, dimension=)` | `[chroma]` | embedded or server; one collection per namespace; flat metadata (nested objects flattened for filtering, exact original kept) |

Anything else can implement `VectorStore` (about six methods).

```python
from aaa.vector import EmbeddingIndex
from aaa.vector.pgvector import PgVectorStore
from aaa.llm import LLMEmbedder

store = PgVectorStore("postgresql://user:pw@host/db", dimension=1536)
await store.setup()                                   # creates the extension, table and indexes once
index = EmbeddingIndex(store, LLMEmbedder(client))
```

## Filters

`F("year") >= 2020`, `(F("lang") == "en") & F("tag").is_in(["a", "b"])`, `~F("archived").exists()`, dotted paths for nested metadata, or a dict:
`{"lang": "en", "year": {"$gte": 2020}}`. The meaning is identical in every store (and tested against all of them): `==` needs the field present, equal and
of the same type (`1` does not match `"1"`); `!=` also matches records without the field; ranges compare numbers with numbers and strings with strings.

Filters are applied **before** the top-k cut, so a rare match is found even when thousands of unrelated records are closer to the query. A store that cannot
express a filter raises `UnsupportedFilter` instead of fetching more and filtering afterwards (which silently returns fewer than `k`): Qdrant, for example, cannot do
ranges over strings. The old `Index` style filter, a Python function over the `Doc`, still works through `EmbeddingIndex`, which over-fetches and keeps asking
until it has `k` survivors (bounded by `max_overfetch`); prefer a `Filter` on a large index.

### What each store cannot do natively

Refused with `UnsupportedFilter` (never approximated):

| Store | Cannot express |
|---|---|
| Qdrant | ranges over strings |
| Chroma | ranges over strings, equality with `null`, `exists`, negated ranges |
| Pinecone | ranges over strings, equality with `null`, negated ranges; also no delete-by-filter |
| LanceDB | nothing is refused, but non-namespace filters are not pushed down: a very selective filter on a large namespace scans a lot. Choose pgvector or Qdrant if you need that at scale |
| pgvector, sqlite-vec, in-memory | everything in the filter language |

## Scores

`Match.score` is a similarity for `cosine` and `dot` (higher is better) and a distance for `euclidean` (lower is better); `Hit` carries `metric` and
`higher_is_better`. Scores from different stores, metrics or embedding models are **not comparable**. Qdrant keeps cosine vectors normalised to unit length.

## Several tenants in one store

```python
acme = TenantVectorStore(store, "acme")        # hand THIS to the code serving acme's requests
```

Two independent walls: every call is pinned to the namespace `acme` (a hard partition in the store), **and** every record carries `meta["tenant"]` while every query and delete
is ANDed with `tenant == "acme"` *outside* the caller's own filter, so an `or` in that filter cannot reach another tenant's rows. Addressing another namespace, or upserting a record that claims
another tenant, raises. `acme.erase()` removes everything of that tenant (tenant offboarding, erasure requests). The tests include filter-escape attempts against every store.

## Reranking

`RerankingIndex(index, LLMReranker(client), fetch_k=30)` fetches generously and lets a model reorder the best `k`; an unreadable reply keeps the original order. `CohereReranker(api_key)` calls Cohere's rerank endpoint instead (plain HTTP; built from Cohere's published request/response, mock-tested, not run live). Voyage and Jina rerankers are not included: their response formats could not be confirmed from the documentation.

## Things to know

- **Dimension.** The first upsert fixes it (pgvector and Qdrant are told up front); a later vector of another size raises `DimensionMismatch`, for writes and for queries. A different
  embedding model with the same dimension cannot be detected: re-index when you change models.
- **pgvector at scale.** One HNSW index serves the table, and namespace/filter conditions apply to its candidates, so a very selective filter on a huge table can return fewer than `k` rows
  unless you raise `ef_search`; for hard partitions at scale use a table per tenant (`table=`) or partition the table yourself. Table names are validated and quoted; values are always parameters.
- **Qdrant** keeps every namespace in one collection (an indexed payload field), which is Qdrant's recommended multi-tenant layout; the in-process mode ignores payload indexes.
- **Verified.** pgvector runs the shared contract suite against a real PostgreSQL with the extension, Qdrant against its real client in `:memory:` mode (not a server), sqlite-vec, LanceDB and Chroma against the real libraries (sqlite-vec 0.1.9, LanceDB 0.40, Chroma 1.5), and Pinecone against **Pinecone Local**, the official emulator, with both SDK 10 and SDK 7 (not a live Pinecone account; SDK 10's index-creation call is rejected by the emulator, so the tests create the index directly): filters, namespaces, tenant escape
  attempts, dimension checks, hostile values. Not run: Qdrant server/cloud, very large indexes, concurrent writers.
