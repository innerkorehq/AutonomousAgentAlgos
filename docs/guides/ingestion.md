# Ingesting documents

Getting documents into a vector index is a pipeline: **load** files, **chunk** them into pieces a model can use, **embed** the pieces, **write** them, and keep the index in step
as the files change. `aaa.ingest` does that, incrementally, against any [vector store](vector-stores.md).

```python
--8<-- "examples/ingest_docs.py"
```

```text
first sync : added=2 updated=0 unchanged=0 removed=0 embedded_chunks=3
second sync: added=0 updated=0 unchanged=2 removed=0 embedded_chunks=0
after edit : added=0 updated=1 unchanged=1 removed=0 embedded_chunks=1
after rm   : added=0 updated=0 unchanged=1 removed=1 embedded_chunks=0
   usage.md | Usage | Create an engine and run an agent.
   usage.md | Usage > Tips | Keep tools small.
```

## The pieces

| | |
|---|---|
| `DirectoryLoader(root)` | every text file under a folder (by extension), `FileLoader(path)` for one file, `PDFLoader(path)` one document per page (`pip install "autonomous-agent-architecture[pdf]"`; text only, no OCR) |
| `RecursiveChunker(size, overlap)` | cuts on the largest natural boundary (paragraph, line, sentence, word) and carries `overlap` into the next chunk; `RecursiveChunker.by_tokens(256)` sizes by tokens |
| `MarkdownChunker()` | one chunk per section, each with `meta["heading"]` such as `Guide > Install > Linux`; code fences are not mistaken for headings |
| `Pipeline(index, chunker, scope=, embedder_id=)` | `await pipeline.sync(*loaders, prune=True)` -> `SyncReport` |

Write a loader for anything else: an object with `async def load()` yielding `Document(source, text, meta)`. A `source` is a stable name (a relative path, a URL):
the same source on the next sync is the same document. There is deliberately **no URL loader**: fetching untrusted URLs needs the guarded fetch tool.

## What `sync` does

- Chunk ids are deterministic, so running again writes the same records over themselves. Every chunk carries `source`, `chunk_idx`, `chunk_count`, `content_hash`, `doc_hash`, and the
  loader's `title` / `uri` / `page`, which flow straight into search results and the `[d1]` citations of `RAGAgent`.
- A **manifest** (per source: hash of its text and metadata, chunk count) lives in the same store, in a sibling namespace (`<namespace>.manifest`) so it never shows up in a search
  and is backed up with the data. An unchanged source is skipped without touching the index.
- A changed source re-embeds **only the chunks whose text changed**; a chunk whose text is the same but whose metadata moved keeps its stored vector; chunks beyond the new end are deleted.
- A source the loaders no longer return is removed (`prune=False` keeps it). `await pipeline.forget(source)` removes one.
- Changing the chunker, or passing a new `embedder_id` (you switched embedding model), re-ingests everything; stored vectors are never reused across embedding models.
- Embedding runs in batches (`batch_size=64`), `concurrency=4` at a time, and transient provider errors (rate limits, timeouts) are retried with backoff.
- **One bad source does not stop the run.** It is listed in `report.errors`, keeps its old manifest entry, and is retried by the next sync. A crash in the middle is safe for the same reason.
- Per-tenant: build the pipeline over a `TenantVectorStore`; its data and its manifest both stay inside that tenant's walls.

## Safe by default

`DirectoryLoader` does not follow symlinks, skips hidden files and directories (`.git`, `.env`), reads only the listed extensions, skips binary files and files over `max_bytes`
(all reported in `report.skipped` with the reason), and refuses a path that resolves outside the root. HTML is reduced to text (scripts and styles dropped). Ingested text still
reaches the model, so treat documents from untrusted sources as untrusted input: the [guardrails](guardrails.md) `tool_result` stage is the place to screen retrieved text.

## Limits

- The manifest is split over 256 records. Each holds its sources' entries as JSON, which is comfortable up to tens of thousands of sources on any store; on **Pinecone** (40 KB of
  metadata per record) the practical ceiling is around 50,000 sources.
- Images inside documents are not extracted; PDFs without a text layer yield nothing.
- Ingestion is one process at a time per scope. Two workers syncing the same scope concurrently race on the manifest (last write wins; the next sync repairs it).
- Verified against the in-memory store and the real pgvector, sqlite-vec, Qdrant (in-process), LanceDB, Chroma and Pinecone Local stores.
