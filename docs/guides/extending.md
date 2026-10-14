# Extending AAA

Every integration point is a small protocol. Implement it, plug it in, and run the repository's contract tests against it to prove
it behaves.

| To add | Implement | Plug in with |
|---|---|---|
| An LLM backend | `LLMClient` (`complete`, `embed`, `aclose`; optional `stream`) | `register_provider(name, factory)` or pass directly to `LLMHandler` |
| A data type | `Codec` (`to_text`, `to_json`, `from_json`, `json_schema`, `fingerprint`) | `CodecRegistry.register(kind, codec)` |
| A vector store | `Index` (`upsert`, `delete`, `search`, `count`) | `RAGAgent(index)` / `search_tool(index)` |
| A blob store | `BlobStore` (`put`, `get`, `exists`, `delete`) | `Engine(blobs=...)` |
| An event store | `EventStore` (below) | `Engine(store=...)` |
| A bus backend | `BusBackend` (below) | `DistributedBus(backend)` |
| An algorithm | `Algorithm` | [Writing your own](algorithms/custom.md) |
| A tool | a function | `@tool` |
| A validator | `fn(candidate, context) -> Verdict` | `ValidationHandler({...})` |

## Route models to different backends

One engine has one `LLMHandler`. To send different models to different providers, give it a client that dispatches on `request.model`:

```python
--8<-- "examples/model_router.py"
```

```text
[small:fast-1] quick answer
[big:deep-9] careful answer
```

Use it with `ToolLoop(registry, model="big/deep-9")` etc. The same pattern implements fallback (try the primary, catch a
`TransientError`, try the secondary) or cost-based routing.

## A custom event store

The protocol (`aaa.events.EventStore`):

| Method | Contract |
|---|---|
| `append(stream, events, expected_seq=ANY_SEQ) -> list[Event]` | **Atomic.** Assign `seq` contiguously per stream. If `expected_seq` is given and the stream's last seq differs, raise `ConcurrencyError` and append nothing. **Idempotent:** an event whose `id` is already in the stream is not appended again; return the existing event with `dup=True`. |
| `read(stream, from_seq=0, limit=None)` | Events with `seq >= from_seq`, in order. |
| `subscribe(stream, from_seq=0)` | Async iterator: replay from `from_seq`, then yield new appends. |
| `last_seq(stream)` | Last seq, or `-1` for an empty/unknown stream. |
| `has(stream, event_id)` | O(1) existence check by id. |
| `run_index()` | `list[RunInfo(stream, started, completed)]` for every stream that contains `aaa.run.started`. |

Optional but recommended: `load_snapshot(stream, name)` and `save_snapshot(snap)` (monotonic by `seq`) to enable snapshots.

To use it in a cluster it must also enqueue to a bus **in the same atomic step as the append**, and expose `bus_backend()`.

!!! success "Prove it"
    `tests/unit/test_store_contract.py` is parametrized over stores through a fixture. Add yours and the contract (OCC,
    idempotency, ordering, subscribe, run index, snapshots, unicode and large payloads, concurrent appends, OCC races) runs
    against it.

## A custom bus backend

`DistributedBus` implements partitioning, consumer groups, leases, rebalancing and the acknowledgement window once; a backend
supplies storage primitives:

| Member | Contract |
|---|---|
| `partitions` | Fixed number of partitions. |
| `read(partition, after, limit, wait_s)` | Entries with id > `after`, ids strictly increasing and **gap-free in commit order**; block up to `wait_s`. |
| `tail(partition)` | Highest id. |
| `get_cursor` / `set_cursor(group, partition, cursor)` | Monotonic per `(group, partition)`. |
| `try_lease(group, partition, member, ttl_s)` | Acquire, or renew if `member` already holds it. Return whether held. |
| `release_lease`, `heartbeat`, `members`, `leave` | Group membership with expiry. |
| `abandon()`, `close()` | Crash simulation hook; cleanup. |

`MemoryBusBackend` is the reference implementation; the bus behavior tests (`test_distributed_bus.py`) take a `cluster` fixture
and run the same scenarios against every backend.

## A custom codec

```python
from aaa.data import FunctionCodec

codecs.register(
    "frame",
    FunctionCodec(
        to_text=lambda df: df.head(20).to_markdown(),
        to_json=lambda df: df.to_dict("records"),
        from_json=lambda rows: pd.DataFrame(rows),
    ),
)
```

## Guidelines for extensions

- Raise the library's typed errors (`aaa.llm.errors`, `ConcurrencyError`) so callers can react correctly.
- Keep handlers idempotent: they will be run more than once.
- Never block the event loop: use async I/O or `asyncio.to_thread`.
