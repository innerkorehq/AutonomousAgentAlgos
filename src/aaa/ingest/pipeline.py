"""Keep a vector index in step with a body of documents: load, chunk, embed what changed, write, and remove what disappeared.

    index = EmbeddingIndex(store, embedder)
    pipeline = Pipeline(index, MarkdownChunker(), embedder_id="text-embedding-3-small")
    report = await pipeline.sync(DirectoryLoader("./docs"))       # run it again whenever the files change

How it knows what changed (no separate database):

* chunk ids are deterministic (`<hash of scope+source>:<n>`), so re-running writes the same records over themselves;
* every chunk carries `source`, `chunk_idx`, `chunk_count`, `content_hash` and `doc_hash` in its metadata, which is what search results cite;
* a **manifest** lives in the same store, in a sibling namespace (`<namespace>.manifest`, so it never appears in search results, and it is backed up with the
  data): per source, the hash of its text and its chunk count. An unchanged source is skipped without reading the index; a changed one re-embeds only the chunks whose text
  changed (others keep their stored vector), and chunks beyond its new length are deleted; a source that a loader no longer returns is removed (`prune=True`).

A failure on one source is reported and the run continues; that source keeps its old manifest entry, so the next sync retries it. Changing the chunker, or passing a new `embedder_id`
(you changed embedding model), re-ingests everything."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from aaa.core.retry import RetryPolicy, is_transient, retrying
from aaa.ingest.chunkers import RecursiveChunker
from aaa.ingest.types import Chunk, Chunker, Document, Loader, digest
from aaa.vector.index import EmbeddingIndex
from aaa.vector.types import Record, VectorError, VectorStore

log = logging.getLogger("aaa.ingest")
SHARDS = 256
EMBED_RETRY = RetryPolicy(max_attempts=5, max_elapsed_s=120.0, base_delay_s=0.5, max_delay_s=10.0)


@dataclass
class SyncReport:
    added: int = 0  # sources new to the index
    updated: int = 0  # sources whose text (or the chunker / embedder) changed
    unchanged: int = 0
    removed: int = 0  # sources no longer returned by the loaders
    chunks_written: int = 0
    chunks_deleted: int = 0
    embed_calls: int = 0
    embedded_chunks: int = 0
    skipped: list[tuple[str, str]] = field(
        default_factory=list
    )  # (source, why) reported by loaders
    errors: list[tuple[str, str]] = field(
        default_factory=list
    )  # (source, error): left for the next sync

    @property
    def ok(self) -> bool:
        return not self.errors


def _chunker_fingerprint(c: Any) -> str:
    parts = [type(c).__name__]
    for name in ("size", "overlap", "separators", "max_chars"):
        if hasattr(c, name):
            parts.append(f"{name}={getattr(c, name)!r}")
    inner = getattr(c, "inner", None)
    if inner is not None:
        parts.append(_chunker_fingerprint(inner))
    ln = getattr(c, "length", None)
    if ln is not None and ln is not len:
        parts.append("length=" + getattr(ln, "__qualname__", "custom"))
    return "|".join(parts)


class _Manifest:
    """Per-source `(fingerprint, doc hash, chunk count)`, sharded over `SHARDS` records so each stays small."""

    def __init__(self, store: VectorStore, namespace: str, scope: str, vector: list[float]) -> None:
        self.store, self.ns, self.scope, self.vector = store, namespace, scope, vector
        self.shards: dict[str, dict[str, list[Any]]] = {}
        self.dirty: set[str] = set()

    def _id(self, shard: str) -> str:
        return f"m:{self.scope}:{shard}"

    @staticmethod
    def shard_of(source: str) -> str:
        return digest(source, 2)

    async def load(self) -> None:
        shards = [f"{i:02x}" for i in range(SHARDS)]
        for i in range(0, SHARDS, 128):
            ids = [self._id(s) for s in shards[i : i + 128]]
            for rec in await self.store.get(ids, namespace=self.ns):
                self.shards[rec.id.rsplit(":", 1)[1]] = json.loads(rec.meta.get("entries", "{}"))

    def get(self, source: str) -> list[Any] | None:
        return self.shards.get(self.shard_of(source), {}).get(source)

    def set(self, source: str, entry: list[Any]) -> None:
        sh = self.shard_of(source)
        self.shards.setdefault(sh, {})[source] = entry
        self.dirty.add(sh)

    def remove(self, source: str) -> None:
        sh = self.shard_of(source)
        if self.shards.get(sh, {}).pop(source, None) is not None:
            self.dirty.add(sh)

    def sources(self) -> list[str]:
        return [s for shard in self.shards.values() for s in shard]

    async def flush(self) -> None:
        upserts, deletes = [], []
        for sh in self.dirty:
            entries = self.shards.get(sh) or {}
            if entries:
                upserts.append(
                    Record(
                        self._id(sh),
                        self.vector,
                        "",
                        {
                            "scope": self.scope,
                            "entries": json.dumps(entries, separators=(",", ":")),
                        },
                    )
                )
            else:
                deletes.append(self._id(sh))
        if upserts:
            await self.store.upsert(upserts, namespace=self.ns)
        if deletes:
            await self.store.delete(deletes, namespace=self.ns)
        self.dirty.clear()


class Pipeline:
    def __init__(
        self,
        index: EmbeddingIndex,
        chunker: Chunker | None = None,
        *,
        scope: str = "default",
        embedder_id: str = "",
        batch_size: int = 64,
        concurrency: int = 4,
        max_chunks_per_document: int = 20_000,
    ) -> None:
        self.index, self.chunker = index, chunker or RecursiveChunker()
        self.scope, self.batch_size, self.max_chunks = scope, batch_size, max_chunks_per_document
        self._sem = asyncio.Semaphore(concurrency)
        self.fingerprint = digest(f"{_chunker_fingerprint(self.chunker)}|{embedder_id}", 12)

    # -- helpers ----------------------------------------------------------------------------
    @property
    def store(self) -> VectorStore:
        return self.index.store

    def _manifest_target(self) -> tuple[VectorStore, str]:
        store = self.index.store
        sibling = getattr(store, "sibling", None)
        if (
            sibling is not None
        ):  # a tenant view: its own sibling namespace, still inside the tenant wall
            return sibling(".manifest"), getattr(sibling(".manifest"), "namespace", ".manifest")
        return store, f"{self.index.namespace or 'default'}.manifest"

    def _key(self, source: str) -> str:
        return digest(f"{self.scope}\x00{source}", 20)

    def _chunk_id(self, source: str, i: int) -> str:
        return f"{self._key(source)}:{i}"

    async def _embed(self, texts: list[str], report: SyncReport) -> list[list[float]]:
        async def one(batch: list[str]) -> list[list[float]]:
            async with self._sem:
                report.embed_calls += 1
                return await retrying(
                    lambda: self.index.embedder.embed(batch),
                    EMBED_RETRY,
                    classify=lambda e: is_transient(e) or bool(getattr(e, "retryable", False)),
                    what="embedding",
                )

        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        out: list[list[float]] = []
        for vecs in await asyncio.gather(*(one(b) for b in batches)):
            out += vecs
        report.embedded_chunks += len(texts)
        return out

    # -- one document -----------------------------------------------------------------------
    async def _ingest(
        self, doc: Document, old: list[Any] | None, reuse: bool, report: SyncReport
    ) -> int:
        chunks: list[Chunk] = self.chunker.chunk(doc)
        if len(chunks) > self.max_chunks:
            raise VectorError(
                f"{doc.source}: {len(chunks)} chunks exceeds max_chunks_per_document={self.max_chunks}"
            )
        n = len(chunks)
        ids = [self._chunk_id(doc.source, i) for i in range(n)]
        have = (
            {r.id: r for r in await self._get(ids)} if old and reuse else {}
        )  # vectors are reused only under the same chunker and embedder
        records: list[Record] = []
        need: list[tuple[int, str]] = []
        for i, c in enumerate(chunks):
            meta = {
                **doc.meta,
                **c.meta,
                "source": doc.source,
                "chunk_idx": i,
                "chunk_count": n,
                "content_hash": digest(c.text, 20),
                "doc_hash": doc.hash[:20],
                "scope": self.scope,
            }
            prev = have.get(ids[i])
            if (
                prev is not None
                and prev.meta.get("content_hash") == meta["content_hash"]
                and prev.vector
            ):
                if prev.meta == meta:
                    continue  # identical record already stored
                records.append(
                    Record(ids[i], prev.vector, c.text, meta)
                )  # same text, new metadata: keep the vector
            else:
                need.append((i, c.text))
                records.append(Record(ids[i], [], c.text, meta))
        if need:
            vecs = await self._embed([t for _, t in need], report)
            fill = {ids[i]: v for (i, _), v in zip(need, vecs, strict=True)}
            for r in records:
                if not r.vector:
                    r.vector = fill[r.id]
        for i in range(0, len(records), self.batch_size):
            await self.store.upsert(
                records[i : i + self.batch_size], namespace=self.index.namespace
            )
        report.chunks_written += len(records)
        if old and old[2] > n:  # the document got shorter: drop the chunks past its new end
            stale = [self._chunk_id(doc.source, i) for i in range(n, old[2])]
            await self.store.delete(stale, namespace=self.index.namespace)
            report.chunks_deleted += len(stale)
        return n

    async def _get(self, ids: list[str]) -> list[Record]:
        out: list[Record] = []
        for i in range(0, len(ids), 200):
            out += await self.store.get(ids[i : i + 200], namespace=self.index.namespace)
        return out

    async def _remove(self, source: str, entry: list[Any]) -> None:
        ids = [self._chunk_id(source, i) for i in range(entry[2])]
        for i in range(0, len(ids), 200):
            await self.store.delete(ids[i : i + 200], namespace=self.index.namespace)

    # -- the sync ---------------------------------------------------------------------------
    async def sync(self, *loaders: Loader, prune: bool = True) -> SyncReport:
        report = SyncReport()
        mstore, mns = self._manifest_target()
        dim = self.store.dimension
        probe = None
        if dim is None:  # a brand-new, dimensionless store: the first embedding fixes it
            probe = (await self.index.embedder.embed(["dimension probe"]))[0]
            dim = len(probe)
        manifest = _Manifest(mstore, mns, self.scope, [1.0] + [0.0] * (dim - 1))
        await manifest.load()
        seen: set[str] = set()
        for loader in loaders:
            async for doc in loader.load():
                seen.add(doc.source)
                entry = manifest.get(doc.source)
                if entry and entry[0] == self.fingerprint and entry[1] == doc.hash:
                    report.unchanged += 1
                    continue
                try:
                    n = await self._ingest(
                        doc, entry, bool(entry and entry[0] == self.fingerprint), report
                    )
                except Exception as exc:
                    log.warning("ingest of %s failed", doc.source, exc_info=True)
                    report.errors.append((doc.source, f"{type(exc).__name__}: {exc}"))
                    continue
                manifest.set(doc.source, [self.fingerprint, doc.hash, n])
                report.updated += 1 if entry else 0
                report.added += 0 if entry else 1
            report.skipped += list(getattr(loader, "skipped", []))
        if prune:
            for source in set(manifest.sources()) - seen:
                entry = manifest.get(source)
                try:
                    if entry:
                        await self._remove(source, entry)
                except Exception as exc:  # noqa: BLE001
                    report.errors.append((source, f"{type(exc).__name__}: {exc}"))
                    continue
                manifest.remove(source)
                report.removed += 1
        await manifest.flush()
        return report

    async def forget(self, source: str) -> bool:
        """Remove one source's chunks and its manifest entry."""
        mstore, mns = self._manifest_target()
        dim = self.store.dimension or 1
        manifest = _Manifest(mstore, mns, self.scope, [1.0] + [0.0] * (dim - 1))
        await manifest.load()
        entry = manifest.get(source)
        if not entry:
            return False
        await self._remove(source, entry)
        manifest.remove(source)
        await manifest.flush()
        return True
