# Data model

AAA is meant to work over *your* data, whatever shape it has. This page explains exactly where your data enters the
system today, what the `Item`/`Codec` layer is for, and where the current boundaries are.

## Where your data enters

| Path | Use it for | Types |
|---|---|---|
| **Run input** | The task and its parameters. | Any JSON-serializable `dict` (`{"task": ..., "customer_id": ...}`). |
| **Tool arguments and results** | Everything the agent reads or writes in your systems. | Anything Pydantic can validate: models, dataclasses, `TypedDict`, lists, dicts, primitives. The tool's JSON schema is generated from the type hints. |
| **Indexed documents** | Knowledge the agent retrieves. | `Doc(id, text, meta)` in an `Index`. |
| **Events** | Anything that happens around a run. | JSON `data`, optionally versioned with a schema. |
| **`Item` + `Codec`** | Your own algorithms and tools that need to *render, persist or fingerprint* arbitrary values uniformly. | Any type, via a registered codec. |

!!! warning "A boundary to know about"
    The built-in algorithms take a plain-dict input and do their work through tools and the LLM; **none of them
    requires an `Item`**. `Item` and `Codec` are the uniform data layer you use in your own tools and algorithms
    (and that the [extension points](../guides/extending.md) are built around). Making the built-in algorithms
    consume `Item`s natively is on the [roadmap](../about/limitations.md).

## Items and codecs

An `Item[T]` wraps a user value in an envelope the library can reason about:

| Field | |
|---|---|
| `id` | ULID by default. |
| `kind` | Open string selecting the codec (`"ticket"`, `"invoice-row"`). |
| `payload` | Your value, untouched. |
| `meta` | Free-form dict. |
| `provenance` | `Provenance(source, parents, ts, version)`: where it came from. |
| `content_hash` | SHA-256 of the canonical JSON, for dedup and caching. |
| `embedding`, `blobs` | Optional vector and blob pointers. |

A **codec** teaches AAA four things about a type: render it to text for an LLM (`to_text`, with truncation), turn it
into JSON and back (`to_json`/`from_json`), describe it (`json_schema`), and fingerprint it.

```python
--8<-- "examples/data_codecs.py"
```

```text
ticket  f7199c89  '{\n  "id": 7,\n  "priority": "high",\n  "title": "…'
row     d578aa73  '{\n  "qty": 3,\n  "sku": "A-1"\n}'
event   81fa4071  '{\n  "ok": false,\n  "type": "login"\n}'
doc     2d87e6fc  'A long document body A long document body A lon…'
frame   78efaf7c  'col_a=1, col_b=x'
preflight: missing capabilities: scorer
```

- `TypeAdapterCodec` (used automatically when you register a *type*) covers Pydantic models, dataclasses, dicts,
  lists and primitives.
- `FunctionCodec` is the escape hatch for anything else: DataFrame rows, images, protobufs, ORM objects.
- Keys are sorted before hashing, so equal content gives equal hashes regardless of key order.

### Capability protocols and `preflight`

Algorithms and tools can declare what they need from your environment by name, and fail fast at startup instead of
mid-run:

```python
from aaa.data import preflight

preflight({"embedder", "scorer"}, {"embedder": my_embedder})  # raises MissingCapability: scorer
```

The protocols are `Embedder` (`async embed(texts) -> vectors`), `Scorer` (`async score(item) -> float`) and
`Keyer` (`key(item) -> Hashable` for dedup).

## Retrieval: `Index`

```python
from aaa.data import Doc, InMemoryIndex

index = InMemoryIndex(embedder)  # embedder optional
await index.upsert([Doc(id="d1", text="...", meta={"source": "wiki"})])
hits = await index.search("capital of France", k=5, filter=lambda d: d.meta["source"] == "wiki")
```

`InMemoryIndex` does BM25 keyword search and, when you give it an `Embedder`, fuses it with cosine similarity using
reciprocal-rank fusion. It is brute force: fine to roughly 10⁵ documents. For more, implement the small `Index`
protocol (`upsert`, `delete`, `search`, `count`) on top of pgvector, Qdrant or similar; `search_tool(index)` and
`RAGAgent` accept any implementation. A production vector-store adapter is not shipped yet.

## Large payloads

Values too big for an event live in a content-addressed blob store and the event carries a pointer. That is covered
in [Large payloads](../guides/large-payloads.md).

## Why events carry JSON, not your types

Events are written to a log that outlives your code. Storing JSON (validated against a versioned schema if you
register one) means a deployment can change a Python class without invalidating history, other languages can read
the log, and a `SELECT` in Postgres is a debugging tool.
