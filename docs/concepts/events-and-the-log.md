# Events and the log

## The envelope

Every fact in AAA is an `Event`. The envelope is compatible with [CloudEvents 1.0](https://cloudevents.io/) and is
immutable (frozen Pydantic model).

| Field | Meaning |
|---|---|
| `id` | Unique within its stream. Time-sortable ULID for external events; **deterministic hash** for events derived from other events (see below). |
| `type` | Namespaced name such as `aaa.llm.completed`. `aaa.*` is reserved for the library; use your own prefix (`acme.ticket.created`). |
| `source` | Who emitted it (an algorithm name, `external`, `engine`). |
| `subject` | Optional entity the event is about. |
| `time` | UTC timestamp. |
| `stream` | The ordering and concurrency boundary. For runs this is the run id. |
| `run_id`, `trace_id`, `span_id` | Which run, and the trace the run belongs to (children share their root's `trace_id`). |
| `tenant` | Owner of the run, stamped at start and inherited by every derived event. See [Tenancy](../guides/privacy.md#tenancy). |
| `causation_id`, `correlation_id` | The event that directly caused this one; the first event of the chain. |
| `seq` | 0-based position in the stream, assigned by the store on append. |
| `schema_ref` | Optional `name/vN` payload schema for [versioned validation and upcasting](#schemas-and-versions). |
| `data` | The JSON payload. |
| `blobs` | [Claim-check](../guides/large-payloads.md) pointers for payloads that live in a blob store. |
| `meta` | Free-form string/number/bool tags (tenant, labels). |

```python
from aaa.events import Event

e = Event(type="acme.invoice.received", data={"invoice": 1042})
child = e.caused("acme.invoice.parsed", {"lines": 3})  # keeps stream, trace and causation
```

## Streams and ordering

A *stream* is an append-only sequence with its own `seq` counter. Within a stream, order is total. Across streams,
no order is promised. A run is one stream; a child run is another stream linked to its parent by events.

The store enforces two properties on `append`:

- **Optimistic concurrency.** `append(stream, events, expected_seq=n)` fails with `ConcurrencyError` unless the
  stream currently ends at `n`. `expected_seq=-1` means "the stream must be empty". The default (`ANY_SEQ`) skips the check.
- **Idempotency.** An event whose `id` is already in the stream is *not* appended again. The existing stored event is
  returned with `dup=True`. This single property underpins crash recovery, retries and multi-worker operation.

## Deterministic ids

`Event.caused(type, data)` derives the new event's id from `(parent id, type, canonical JSON of data)`. Deriving the
same event twice from the same parent therefore yields the same id, and the second append is absorbed by the store.
Results use a fixed id too: the result of request `R` is always `R:result`, so two workers racing on one request
produce exactly one result on the log.

!!! example "Why this matters"
    A worker crashes after appending three of the four events it was about to produce. On restart the same input
    event is redelivered. The algorithm runs again, produces all four events with the same ids, and the store keeps
    the new one and ignores the three duplicates. No de-duplication code exists in any algorithm.

## Reading the log

```python
events = await engine.store.read(run_id)  # whole stream
tail = await engine.store.read(run_id, from_seq=40)  # from a position
async for e in engine.handle(run_id).events():  # replay, then follow until the run ends
    ...
```

Payloads that were claim-checked into a blob store appear as `{"$claim": True, ...}` on the stored event; call
`await engine.hydrate(event)` for the full payload. Handlers and algorithms receive hydrated events automatically.

## Schemas and versions

Register a Pydantic model per versioned payload and migrate old events on read:

```python
from pydantic import BaseModel
from aaa.events import SchemaRegistry


class NoteV2(BaseModel):
    text: str
    lang: str = "en"


schemas = SchemaRegistry()
schemas.register("acme.note/v2", NoteV2)
schemas.upcaster("acme.note", 1, lambda d: {"text": d["body"]})  # v1 -> v2

engine = Engine(schemas=schemas)
```

`schema_ref` values without a registered schema pass through untouched, so unregistered event types cost nothing.

## CloudEvents interoperability

```python
from aaa.events.cloudevents import to_cloudevent, from_cloudevent

ce = to_cloudevent(event)  # a plain dict in CloudEvents 1.0 structured JSON mode
same = from_cloudevent(ce)  # round-trips exactly
```

The conversion has no dependency on the CloudEvents SDK. Fields CloudEvents cannot carry (float `meta` values, blob
pointers) travel in `aaa*` extension attributes.

## The event catalog

Every event type the library emits, with payload shapes, is listed in the [event catalog](../reference/event-catalog.md).
