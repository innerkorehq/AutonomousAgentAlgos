# Large payloads

Events live in the log forever and are read often, so they should be small. When a payload (a tool result, an LLM response, a
document) exceeds a threshold, AAA moves it to a **blob store** and leaves a pointer in the event: the *claim-check* pattern.

```python
--8<-- "examples/large_payloads.py"
```

```text
got 60000 chars
stored event keys: ['$claim', 'call_id', 'name'] | blob: 60126 bytes
hydrated result length: 60000
```

## Turning it on

```python
from aaa.data import FileBlobStore, InMemoryBlobStore

engine = Engine(
    blobs=InMemoryBlobStore(), claim_threshold=16_384
)  # default threshold: 16 KiB of JSON
engine = Engine(blobs=FileBlobStore("/var/lib/aaa/blobs"))  # local disk
```

Without `blobs=`, nothing is claim-checked and large payloads stay in events.

## What is stored where

- The event keeps its envelope, a `"$claim": true` marker, and a few small **hot fields** (`usage`, `cost`, `tag`, `call_id`,
  `name`) so spend accounting and correlation never need to fetch the blob. The full `data` JSON is in the blob, referenced by `event.blobs[-1]`.
- `aaa.run.completed` is **never** claim-checked, so outcomes are always readable directly.
- Blobs are **content-addressed (SHA-256)** and every read verifies the hash (`BlobIntegrityError` on tampering or
  corruption). The engine namespaces each payload by its stream before storing it, so identical payloads in two runs are two
  different blobs: that is what lets [erasing one run](privacy.md#erasure) remove its blobs without touching anyone else's.

## Reading

Handlers and algorithms receive events **already hydrated**: you never see a stub. If you read the raw log yourself:

```python
full = await engine.hydrate(event)  # no-op for ordinary events
```

## Blob stores

| | |
|---|---|
| `InMemoryBlobStore()` | Tests and single-process, non-durable use. |
| `FileBlobStore(root)` | Sharded by hash prefix, atomic writes, verified reads. Put `root` on a durable (and in a cluster, **shared**) volume. |
| your own | Implement `put(data, media_type) -> BlobRef`, `get(ref)`, `exists(ref)`, `delete(ref)`; an S3 adapter is a small class. S3 is not shipped. |

## Operational notes

- **A blob must be reachable from every worker that may read the event.** In a cluster use shared storage (NFS, S3-style).
  With a non-shared store, an event written on worker A cannot be hydrated on B.
- **Blobs of erased runs are deleted** by [`engine.erase`](privacy.md#erasure) and retention policies. Orphans from other causes
  (a crash between writing a blob and appending its event) are not garbage-collected; use your storage layer's lifecycle rules.
- **Blobs are not encrypted** by AAA. Use an encrypted volume or bucket for sensitive data. See [Security](../operations/security.md).
- Projections read *hydrated* events, so folding over claim-checked events fetches their blobs (snapshots keep that to the tail since the last snapshot). Spend accounting needs only the hot fields and never fetches a blob.
