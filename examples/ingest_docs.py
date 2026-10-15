"""Ingestion: index a folder of documents, re-run after edits, and see exactly what was redone. Offline (a toy embedder)."""

import asyncio
import hashlib
import tempfile
from pathlib import Path

from aaa.ingest import DirectoryLoader, MarkdownChunker, Pipeline
from aaa.vector import EmbeddingIndex, F, InMemoryVectorStore


class ToyEmbedder:  # stands in for LLMEmbedder(client): a 4-number fingerprint of the text
    async def embed(self, texts):
        return [[1.0 + b / 255 for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts]


def show(label, r) -> None:
    print(
        f"{label}: added={r.added} updated={r.updated} unchanged={r.unchanged} removed={r.removed} embedded_chunks={r.embedded_chunks}"
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp)
        (docs / "install.md").write_text(
            "# Install\nRun pip install aaa.\n\n## Linux\nUse your package manager.\n"
        )
        (docs / "usage.md").write_text("# Usage\nCreate an engine and run an agent.\n")
        index = EmbeddingIndex(InMemoryVectorStore(4), ToyEmbedder())
        pipeline = Pipeline(index, MarkdownChunker(), embedder_id="toy-v1")
        loader = DirectoryLoader(docs)

        show("first sync ", await pipeline.sync(loader))
        show("second sync", await pipeline.sync(loader))  # nothing changed: nothing embedded
        (docs / "usage.md").write_text(
            "# Usage\nCreate an engine and run an agent.\n\n## Tips\nKeep tools small.\n"
        )
        show("after edit ", await pipeline.sync(loader))
        (docs / "install.md").unlink()
        show("after rm   ", await pipeline.sync(loader))

        for hit in await index.search("tools", k=10, filter=F("source") == "usage.md"):
            print(
                "  ",
                hit.doc.meta["source"],
                "|",
                hit.doc.meta.get("heading", "-"),
                "|",
                hit.doc.text.splitlines()[-1],
            )


asyncio.run(main())
