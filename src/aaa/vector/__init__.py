"""Persistent vector search: stores that hold vectors, an index that adds text, and the pieces around them.

store = PgVectorStore(dsn, dimension=1536)                    # or QdrantStore, ... or InMemoryVectorStore
index = EmbeddingIndex(store, LLMEmbedder(client))            # an `aaa.data.Index`: RAGAgent, search_tool, NativeMemory use it
await index.upsert([Doc(id="1", text="...", meta={"lang": "en"})])
hits = await index.search("question", k=5, filter=F("lang") == "en")

tenant = TenantVectorStore(store, "acme")                    # one tenant's walled-off view
"""

from aaa.vector.filters import (
    And,
    Eq,
    Exists,
    F,
    Filter,
    In,
    Ne,
    Not,
    Or,
    Range,
    UnsupportedFilter,
    matches,
)
from aaa.vector.index import EmbeddingIndex, TenantVectorStore
from aaa.vector.memory import InMemoryVectorStore
from aaa.vector.rerank import LLMReranker, Reranker, RerankingIndex
from aaa.vector.types import (
    DeleteWhere,
    DimensionMismatch,
    DropNamespace,
    HybridSearch,
    Match,
    Metric,
    Record,
    VectorError,
    VectorStore,
)

__all__ = [
    "And",
    "DeleteWhere",
    "DimensionMismatch",
    "DropNamespace",
    "EmbeddingIndex",
    "Eq",
    "Exists",
    "F",
    "Filter",
    "HybridSearch",
    "In",
    "InMemoryVectorStore",
    "LLMReranker",
    "Match",
    "Metric",
    "Ne",
    "Not",
    "Or",
    "Range",
    "Record",
    "Reranker",
    "RerankingIndex",
    "TenantVectorStore",
    "UnsupportedFilter",
    "VectorError",
    "VectorStore",
    "matches",
]
