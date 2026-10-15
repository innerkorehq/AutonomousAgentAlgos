"""Ingestion: loaders, chunkers and the sync pipeline (incremental, resumable, and the same on every vector store)."""

import hashlib
import itertools
import re

import pytest
from test_vector_contract import backend, store  # noqa: F401  (fixtures: the real backends of the contract suite)

from aaa.ingest import (
    DirectoryLoader,
    FileLoader,
    MarkdownChunker,
    PDFLoader,
    Pipeline,
    RecursiveChunker,
    html_to_text,
)
from aaa.ingest.types import Document
from aaa.vector import EmbeddingIndex, F, InMemoryVectorStore, TenantVectorStore
from aaa.vector.index import TENANT_KEY  # noqa: F401


class CountingEmbedder:
    """A deterministic 4-d embedder that records every call."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.fail_on: str | None = None

    async def embed(self, texts):
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise ValueError("bad text")
        self.calls.append(list(texts))
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([1.0 + h[0] / 255, 1.0 + h[1] / 255, 1.0 + h[2] / 255, 1.0 + h[3] / 255])
        return out

    @property
    def embedded(self):
        return sum(len(c) for c in self.calls)


class ListLoader:
    def __init__(self, docs):
        self.docs, self.skipped = docs, []

    async def load(self):
        for d in self.docs:
            yield d


def docs(**texts):
    return ListLoader([Document(src, text, {"title": src}) for src, text in texts.items()])


LONG = "\n\n".join(f"Paragraph {i} talks about topic{i} in some detail. " * 6 for i in range(12))


# -- chunkers ---------------------------------------------------------------------------------
def test_recursive_chunker_respects_size_keeps_everything_and_overlaps():
    c = RecursiveChunker(size=300, overlap=60)
    chunks = c.split_text(LONG)
    assert len(chunks) > 3 and all(len(x) <= 300 for x in chunks)
    words = set(re.findall(r"topic\d+", LONG))
    assert words <= set(re.findall(r"topic\d+", " ".join(chunks)))  # nothing lost
    sentences = " ".join(
        f"Sentence number {i} says something." for i in range(40)
    )  # small pieces: room for overlap
    chunks = RecursiveChunker(size=200, overlap=50).split_text(sentences)
    assert len(chunks) > 3 and all(len(x) <= 200 for x in chunks)
    assert all(
        a.split(". ")[-1][:20] in b for a, b in itertools.pairwise(chunks)
    )  # each chunk starts with the end of the previous one


def test_recursive_chunker_edge_cases():
    assert (
        RecursiveChunker(50, 5).split_text("") == []
        and RecursiveChunker(50, 5).split_text("   \n ") == []
    )
    assert RecursiveChunker(50, 5).split_text("short") == ["short"]
    long_word = "x" * 130
    assert all(len(p) <= 50 for p in RecursiveChunker(50, 5).split_text(long_word))
    with pytest.raises(ValueError):
        RecursiveChunker(10, 10)


def test_token_sized_chunks():
    chunks = RecursiveChunker.by_tokens(40, 5).split_text(LONG)
    from aaa.context import HeuristicCounter

    assert all(HeuristicCounter().text(c) <= 40 for c in chunks)


def test_markdown_chunker_tracks_headings_and_ignores_code_fences():
    md = "intro line\n\n# Guide\nwelcome\n\n## Install\nrun it\n```\n# not a heading\nls\n```\n\n## Use\nuse it\n\n# Other\nend"
    chunks = MarkdownChunker().chunk(Document("g.md", md))
    heads = [c.meta.get("heading") for c in chunks]
    assert heads == [None, "Guide", "Guide > Install", "Guide > Use", "Other"]
    assert "# not a heading" in chunks[2].text
    assert (
        MarkdownChunker(max_chars=80, overlap=10)
        .chunk(Document("b.md", "# H\n" + "word " * 100))[0]
        .meta["heading"]
        == "H"
    )


def test_html_to_text():
    text, title = html_to_text(
        "<html><head><title>My Page</title><style>p{}</style></head><body><script>evil()</script><h1>Hi</h1><p>one &amp; two</p></body></html>"
    )
    assert (
        title == "My Page" and "evil" not in text and "one & two" in text and text.startswith("Hi")
    )


# -- loaders ----------------------------------------------------------------------------------
async def test_directory_loader_is_safe_by_default(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\nbody")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta")
    (tmp_path / "page.html").write_text("<title>T</title><p>html body</p>")
    (tmp_path / ".hidden.md").write_text("secret")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.txt").write_text("secret")
    (tmp_path / "bin.txt").write_bytes(b"abc\x00def")
    (tmp_path / "huge.txt").write_text("x" * 2000)
    (tmp_path / "skip.log").write_text("excluded by extension? no, .log is text")
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("do not read")
    (tmp_path / "link.txt").symlink_to(outside)
    loader = DirectoryLoader(tmp_path, max_bytes=1000, exclude=["*.log"])
    got = {d.source: d async for d in loader.load()}
    assert set(got) == {"a.md", "sub/b.txt", "page.html"}
    assert (
        got["a.md"].meta["title"] == "Alpha"
        and got["page.html"].meta["title"] == "T"
        and "html body" in got["page.html"].text
    )
    reasons = dict(loader.skipped)
    assert (
        reasons["bin.txt"] == "looks binary"
        and "larger" in reasons["huge.txt"]
        and reasons["link.txt"] == "symlink"
    )
    assert not any("secret" in d.text or "do not read" in d.text for d in got.values())


async def test_file_loader_reports_unreadable_files(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"\x00\x01")
    fl = FileLoader(p)
    assert [d async for d in fl.load()] == [] and fl.skipped == [("x.txt", "looks binary")]


async def test_pdf_loader_pages(tmp_path):
    pytest.importorskip("pypdf")
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
        None,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    streams = {
        4: "BT /F1 12 Tf 10 50 Td (First page text) Tj ET",
        6: "BT /F1 12 Tf 10 50 Td (Second page text) Tj ET",
    }
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        if body is None:
            s = streams[n]
            out += f"{n} 0 obj\n<< /Length {len(s)} >>\nstream\n{s}\nendstream\nendobj\n".encode()
        else:
            out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(
        f"{o:010d} 00000 n \n".encode() for o in offsets
    )
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path = tmp_path / "doc.pdf"
    path.write_bytes(out)
    pages = [d async for d in PDFLoader(path).load()]
    assert [(d.source, d.meta["page"]) for d in pages] == [
        ("doc.pdf#page=1", 1),
        ("doc.pdf#page=2", 2),
    ]
    assert "First page text" in pages[0].text and "Second page text" in pages[1].text


# -- the pipeline -----------------------------------------------------------------------------
def make(store_=None, **kw):
    emb = CountingEmbedder()
    idx = EmbeddingIndex(store_ or InMemoryVectorStore(4), emb)
    return Pipeline(idx, RecursiveChunker(300, 40), **kw), idx, emb


async def test_first_sync_indexes_second_sync_does_nothing():
    p, idx, emb = make(batch_size=5)
    r1 = await p.sync(docs(a=LONG, b="short note about foxes"))
    assert (r1.added, r1.updated, r1.unchanged, r1.removed) == (2, 0, 0, 0) and r1.ok
    assert r1.chunks_written == await idx.count() and r1.embedded_chunks == r1.chunks_written
    assert max(len(c) for c in emb.calls) <= 5 and r1.embed_calls == len(emb.calls)
    hit = (await idx.search("foxes", k=50, filter=F("source") == "b"))[0]
    assert (
        hit.doc.meta["source"] == "b"
        and hit.doc.meta["chunk_idx"] == 0
        and hit.doc.meta["chunk_count"] == 1
        and hit.doc.meta["content_hash"]
    )
    before = emb.embedded
    r2 = await p.sync(docs(a=LONG, b="short note about foxes"))
    assert (
        (r2.added, r2.updated, r2.unchanged) == (0, 0, 2)
        and r2.embed_calls == 0
        and emb.embedded == before
        and r2.chunks_written == 0
    )


async def test_an_edit_reembeds_only_changed_chunks_and_a_shorter_document_loses_its_tail():
    p, idx, _emb = make()
    await p.sync(docs(a=LONG))
    total = await idx.count()
    edited = LONG.replace("topic11", "topic-eleven")  # touches the last paragraph only
    r = await p.sync(docs(a=edited))
    assert (r.updated, r.unchanged) == (1, 0) and 0 < r.embedded_chunks < total
    short = LONG[:400]
    r2 = await p.sync(docs(a=short))
    assert r2.chunks_deleted > 0 and await idx.count() == len(
        RecursiveChunker(300, 40).split_text(short)
    )
    texts = " ".join(h.doc.text for h in await idx.search("topic", k=50))
    assert "topic11" not in texts and "topic-eleven" not in texts


async def test_removed_sources_are_pruned_unless_you_say_otherwise():
    p, idx, _ = make()
    await p.sync(docs(a="alpha text", b="beta text"))
    r = await p.sync(docs(a="alpha text"), prune=False)
    assert r.removed == 0 and await idx.count() == 2
    r = await p.sync(docs(a="alpha text"))
    assert r.removed == 1 and [h.doc.meta["source"] for h in await idx.search("text", k=10)] == [
        "a"
    ]
    assert await p.forget("a") and await idx.count() == 0 and not await p.forget("a")


async def test_changing_only_metadata_reuses_stored_vectors():
    p, idx, emb = make()
    await p.sync(ListLoader([Document("a", "same text", {"title": "one"})]))
    n = emb.embedded
    r = await p.sync(ListLoader([Document("a", "same text", {"title": "two"})]))
    assert r.updated == 1 and r.chunks_written == 1
    assert emb.embedded == n  # new title, same text: the vector was kept
    assert (await idx.search("same", k=1))[0].doc.meta["title"] == "two"


async def test_a_new_chunker_or_embedder_reingests_everything():
    store_ = InMemoryVectorStore(4)
    p, idx, emb = make(store_)
    await p.sync(docs(a=LONG))
    n = emb.embedded
    p2 = Pipeline(idx, RecursiveChunker(200, 20))
    r = await p2.sync(docs(a=LONG))
    assert r.updated == 1 and emb.embedded > n
    p3 = Pipeline(idx, RecursiveChunker(200, 20), embedder_id="model-v2")
    n2 = emb.embedded
    r3 = await p3.sync(docs(a=LONG))
    assert (
        r3.updated == 1 and emb.embedded - n2 == r3.chunks_written
    )  # nothing reused across embedding models


async def test_one_failing_source_is_reported_and_retried_next_time():
    p, _idx, emb = make()
    emb.fail_on = "POISON"
    r = await p.sync(docs(good="fine text", bad="this has POISON in it"))
    assert r.added == 1 and [s for s, _ in r.errors] == ["bad"] and not r.ok
    emb.fail_on = None
    r2 = await p.sync(docs(good="fine text", bad="this has POISON in it"))
    assert (r2.added, r2.unchanged, r2.errors) == (1, 1, [])


async def test_transient_embedding_errors_are_retried(monkeypatch):
    from aaa.core.retry import RetryPolicy
    from aaa.ingest import pipeline as mod
    from aaa.llm.errors import TransientError

    monkeypatch.setattr(
        mod, "EMBED_RETRY", RetryPolicy(max_attempts=4, base_delay_s=0.001, max_delay_s=0.002)
    )
    p, idx, emb = make()
    real, fails = emb.embed, [2]

    async def flaky(texts):
        if fails[0]:
            fails[0] -= 1
            raise TransientError("blip")
        return await real(texts)

    emb.embed = flaky  # type: ignore[method-assign]
    r = await p.sync(docs(a="some text"))
    assert r.ok and await idx.count() == 1


async def test_the_manifest_never_appears_in_search_and_scopes_do_not_collide():
    store_ = InMemoryVectorStore(4)
    emb = CountingEmbedder()
    idx = EmbeddingIndex(store_, emb)
    a, b = Pipeline(idx, scope="one"), Pipeline(idx, scope="two")
    await a.sync(docs(readme="from scope one"))
    await b.sync(docs(readme="from scope two"))
    assert await idx.count() == 2 and {h.doc.text for h in await idx.search("scope", k=10)} == {
        "from scope one",
        "from scope two",
    }
    assert (
        await store_.count(namespace="default.manifest") > 0
    )  # lives next door, invisible to search
    await a.sync(docs())  # scope one empties: only its chunks go
    assert [h.doc.text for h in await idx.search("scope", k=10)] == ["from scope two"]


async def test_a_tenant_view_keeps_its_data_and_its_manifest_inside_its_walls():
    base = InMemoryVectorStore(4)
    emb = CountingEmbedder()
    acme = Pipeline(EmbeddingIndex(TenantVectorStore(base, "acme"), emb))
    globex = Pipeline(EmbeddingIndex(TenantVectorStore(base, "globex"), emb))
    await acme.sync(docs(plan="acme secret plan"))
    await globex.sync(docs(plan="globex secret plan"))
    assert [h.doc.text for h in await acme.index.search("secret", k=10)] == ["acme secret plan"]
    await globex.sync(docs())  # globex removes its document; acme is untouched
    assert [h.doc.text for h in await acme.index.search("secret", k=10)] == ["acme secret plan"]
    assert await TenantVectorStore(base, "acme").count() == 1


# -- the same on every real vector store -------------------------------------------------------
async def test_sync_edit_and_prune_on_each_real_store(store, backend):  # noqa: F811
    emb = CountingEmbedder()
    p = Pipeline(EmbeddingIndex(store, emb), RecursiveChunker(300, 40))
    r1 = await p.sync(docs(a=LONG, b="foxes jump"))
    assert r1.ok and r1.added == 2
    assert (await p.sync(docs(a=LONG, b="foxes jump"))).unchanged == 2
    r3 = await p.sync(docs(a=LONG[:350]))
    assert r3.removed == 1 and r3.updated == 1
    ms = await store.query([1.0, 1.0, 1.0, 1.0], 100, filter=F("source") == "a")
    assert {m.meta["source"] for m in ms} == {"a"} and not await store.query(
        [1.0, 1.0, 1.0, 1.0], 100, filter=F("source") == "b"
    )
