import pytest

from aaa.data import BlobIntegrityError, Doc, FileBlobStore, InMemoryBlobStore, InMemoryIndex


@pytest.mark.parametrize("make", [lambda tp: InMemoryBlobStore(), lambda tp: FileBlobStore(tp)])
async def test_blob_contract(make, tmp_path):
    s = make(tmp_path)
    a = await s.put(b"hello", "text/plain")
    assert a == await s.put(b"hello", "text/plain")  # content-addressed: dedup
    assert await s.get(a) == b"hello" and await s.exists(a) and a.size == 5
    await s.delete(a)
    assert not await s.exists(a)
    with pytest.raises(FileNotFoundError):
        await s.get(a)


async def test_file_blob_detects_tampering(tmp_path):
    s = FileBlobStore(tmp_path)
    ref = await s.put(b"original")
    next(tmp_path.rglob(ref.sha256)).write_bytes(b"tampered")
    with pytest.raises(BlobIntegrityError):
        await s.get(ref)


class WordEmbedder:
    VOCAB = ("cat", "dog", "finance", "pet")

    async def embed(self, texts):
        return [[float(w in t.lower()) for w in self.VOCAB] for t in texts]


async def test_index_keyword_hybrid_filter_and_delete():
    docs = [
        Doc(id="1", text="the cat sat", meta={"t": "a"}),
        Doc(id="2", text="dog days", meta={"t": "b"}),
        Doc(id="3", text="quarterly finance report", meta={"t": "a"}),
    ]
    kw = InMemoryIndex()
    await kw.upsert(docs)
    assert [h.doc.id for h in await kw.search("cat")] == ["1"]
    assert [
        h.doc.id for h in await kw.search("report finance", filter=lambda d: d.meta["t"] == "a")
    ] == ["3"]
    hy = InMemoryIndex(WordEmbedder())
    await hy.upsert([Doc(id="1", text="cat"), Doc(id="2", text="dog"), Doc(id="3", text="finance")])
    top = await hy.search("pet cat", k=1)  # vector side catches 'cat'
    assert top[0].doc.id == "1"
    await hy.delete(["1"])
    assert await hy.count() == 2
