"""Ingestion: documents in, a searchable index out, and kept in step as the documents change.

pipeline = Pipeline(EmbeddingIndex(store, embedder), MarkdownChunker(), embedder_id="text-embedding-3-small")
report = await pipeline.sync(DirectoryLoader("./docs"))
"""

from aaa.ingest.chunkers import MarkdownChunker, RecursiveChunker
from aaa.ingest.loaders import DirectoryLoader, FileLoader, PDFLoader, html_to_text
from aaa.ingest.pipeline import Pipeline, SyncReport
from aaa.ingest.types import Chunk, Chunker, Document, Loader

__all__ = [
    "Chunk",
    "Chunker",
    "DirectoryLoader",
    "Document",
    "FileLoader",
    "Loader",
    "MarkdownChunker",
    "PDFLoader",
    "Pipeline",
    "RecursiveChunker",
    "SyncReport",
    "html_to_text",
]
