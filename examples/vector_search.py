"""Vector search: documents with metadata, filtered search, a per-tenant wall. Offline (swap in PgVectorStore / QdrantStore)."""

import asyncio

from aaa.data.index import Doc
from aaa.vector import EmbeddingIndex, F, InMemoryVectorStore, Record, TenantVectorStore

VOCAB = ["refund", "shipping", "invoice", "password", "returns", "login"]


class WordEmbedder:  # stands in for LLMEmbedder(client): counts vocabulary words
    async def embed(self, texts):
        return [[float(t.lower().split().count(w)) + 0.01 for w in VOCAB] for t in texts]


async def main() -> None:
    # In production: PgVectorStore("postgresql://...", dimension=1536)  or  QdrantStore(url="http://localhost:6333", dimension=1536)
    store = InMemoryVectorStore(dimension=len(VOCAB))
    index = EmbeddingIndex(store, WordEmbedder())
    await index.upsert(
        [
            Doc(id="a", text="refund policy for returns", meta={"lang": "en", "year": 2023}),
            Doc(id="b", text="shipping and returns", meta={"lang": "en", "year": 2021}),
            Doc(id="c", text="reset your password at login", meta={"lang": "en", "year": 2024}),
            Doc(id="d", text="refund policy for returns", meta={"lang": "fr", "year": 2024}),
        ]
    )
    hits = await index.search("refund returns", k=4)
    print(
        "all:",
        [h.doc.id for h in hits],
        f"(scores are {hits[0].metric}, higher is better: {hits[0].higher_is_better})",
    )
    recent = await index.search(
        "refund returns", k=4, filter=(F("lang") == "en") & (F("year") >= 2023)
    )
    print("english, 2023 or later:", [h.doc.id for h in recent])

    acme, globex = TenantVectorStore(store, "acme"), TenantVectorStore(store, "globex")
    await acme.upsert([Record("1", [1.0, 0, 0, 0, 0, 0], "acme invoice terms", {"lang": "en"})])
    await globex.upsert([Record("1", [1.0, 0, 0, 0, 0, 0], "globex invoice terms", {"lang": "en"})])
    wide_open = (F("lang") == "en") | (F("lang") != "en")  # a filter that matches every row...
    print(
        "acme sees:", [m.text for m in await acme.query([1.0, 0, 0, 0, 0, 0], 10, filter=wide_open)]
    )  # ...still only its own


asyncio.run(main())
