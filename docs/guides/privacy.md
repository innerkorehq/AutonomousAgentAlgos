# Tenancy, redaction and retention

An event log keeps everything, including the things you would rather not keep: customers' personal data, secrets a tool
returned, a prompt someone pasted. AAA gives you three controls, which compose:

| Control | Question it answers | Where it acts |
|---|---|---|
| **Tenancy** | Whose run is this, and who may see it? | `Event.tenant`, `engine.tenant("acme")` |
| **Redaction** | What must never reach storage? | `Engine(redactor=...)`, on every append |
| **Erasure and retention** | How do we delete data later? | `engine.erase`, `engine.purge`, `Engine(retention=...)` |

```python
--8<-- "examples/privacy_tenancy.py"
```

```text
answer: found C-7: [REDACTED:email], card [REDACTED:card]
tenants on the log: ['acme']
stored e-mail visible: False
acme asking for globex's run: "no run 'globex/ticket-1'"
erased 9 events; last event: aaa.run.erased
payload left: False
still finished: True
globex untouched: True
```

## Tenancy

Every event has an optional `tenant` field. It is set once, when the run starts, and every event derived from the run
inherits it: LLM and tool events, signals you send, the completion event, and the runs a run starts as children.

The safe way to use it is a **tenant scope**:

```python
acme = engine.tenant("acme")

handle = await acme.start(algo, {"task": "..."}, run_id="order-1")  # stream id: "acme/order-1"
await acme.handle("order-1")      # ownership-checked lookup
await acme.runs()                 # only acme's runs
await acme.erase("order-1")       # only acme's runs
```

A scope does three things the raw `engine.start(..., tenant=...)` does not:

1. **Namespaces run ids** (`"acme/order-1"`), so two tenants may both use `order-1` without sharing a stream. Without this
   a clash would silently merge two customers' runs.
2. **Checks ownership on every lookup.** Asking for another tenant's run raises `KeyError`, **the same error as a run that does
   not exist**, so a tenant cannot probe which ids are in use.
3. **Scopes listings and bulk operations** (`runs()`, `purge()`, `erase_all()`) to the tenant.

`Engine(require_tenant=True)` makes an untenanted run an error, which is the right default for a multi-customer service: a
forgotten tenant becomes a loud failure instead of data nobody can attribute.

Tenant names are non-empty and contain no `/`.

!!! warning "A scope is bookkeeping, not authentication"
    The scope trusts the tenant name you give it. Derive that name from the authenticated caller, never from request input,
    and keep the raw `engine` out of request handlers. For hard isolation (regulated data, hostile tenants) use separate
    databases or key prefixes per tenant; see [Security](../operations/security.md#multi-tenancy).

## Redaction

A **redactor** rewrites an event before it is stored or delivered. It runs inside `Engine.emit`, after schema validation and
before the large-payload claim-check, so the redacted version is the *only* version: the log, the blob store, the bus, every
handler and every projection see it.

```python
from aaa.privacy import standard_redactor

engine = Engine(redactor=standard_redactor())
```

| Piece | Does |
|---|---|
| `PatternRedactor()` | Replaces regex matches in any string with `[REDACTED:<name>]`. Built in: `email`, `card` (Luhn-checked, so order numbers survive), `ssn` (US), `secret` (OpenAI/GitHub/AWS/Slack/Google key shapes), `jwt`, `bearer`. Add your own `Pattern(name, regex, check=None)`. |
| `KeyRedactor()` | Replaces the *value* of secret-bearing keys (`password`, `api_key`, `authorization`, `cookie`, `token`...) anywhere in the payload. Whole-key match, so `prompt_tokens` (accounting) is never touched. |
| `ChainRedactor(a, b, ...)` / `standard_redactor()` | Run several in order. `standard_redactor()` is keys plus the default patterns. |
| `only=[...]`, `skip=[...]` | Event-type globs: restrict a redactor to some events. |

Write your own by implementing `redact(event) -> Event`. It must be **pure, deterministic and idempotent**: the engine may
append the same event twice after a store outage and expects the same result.

### What redaction protects, and what it changes

!!! danger "The model sees the redacted text"
    Redaction happens before delivery, so a tool result handed back to the model is already redacted. `ann@example.com`
    becomes `[REDACTED:email]` *for the model too*. That is usually what you want (the secret never leaves your process),
    but if a run needs the real value to do its job, exclude those events with `skip=["aaa.tool.*"]`, or keep the value out
    of events entirely (keep an id in the event and look the value up inside the tool).

- Redaction rewrites **strings and secret-bearing keys**, never numbers, ids or the structural fields the library routes on
  (`call_id`, `usage`, `cost`, `budget`, `role`...). A user field that happens to be called `name` *is* scrubbed when it is
  nested; only the top-level tool `name` is structural.
- **Patterns are a safety net, not a guarantee.** A regex cannot recognise a name, an address or free-text medical detail.
  Keep sensitive data out of events where you can; use redaction to catch what slips through.
- `Event.meta` and the event's `data` are both redacted. The `subject` field and `source` are not: do not put personal data there.
- Original values passed to **your own code** (a tool's arguments as the model produced them, the data you give `start`) are not
  redacted in memory. Only what is persisted and delivered is.

## Erasure

The log is append-only, so erasing is done the event-sourcing way. You choose between two modes:

| Mode | What remains | Use when |
|---|---|---|
| **scrub** (default) | Each event's `id`, `type`, `seq`, `time`, causal ids, `tenant`, plus `algo`/`kind`/`parent` on start and completion events, and a final `aaa.run.erased` marker saying when and why. Payloads, `meta` and `subject` are gone. | You must keep an audit trail that something happened, but not what. |
| **delete** (`delete=True`) | Nothing. | The record itself must go. |

```python
report = await engine.erase("acme/order-1", reason="customer request #42")
report = await engine.erase("acme/order-1", delete=True)
```

Erasing a run also removes everything that held its data: its **snapshots**, its claim-checked **blobs**, and the bus's transient
**outbox copies** of its events. (Blobs are namespaced by stream, so two runs with identical payloads never share a blob and erasing
one cannot affect the other.)

- Only **finished** runs can be erased. An open run raises `RunNotFinished`; cancel it first, or pass `force=True`.
- It is **idempotent and crash-safe**: blobs go first, so a rerun after a crash finds the events and finishes the job.
- A scrubbed run still reports its verdict: `await handle.result()` returns the outcome kind (the *value* is gone).
- Children are separate runs. `acme.erase_all()` erases every finished run of the tenant.

!!! note "What AAA cannot erase"
    Anything outside its stores: **your own database backups**, logs your process wrote (`aaa.*` loggers include tracebacks),
    exports you made with `to_cloudevent`, the gateway's own logs, and the model provider's retention. Erasure is a promise
    about AAA's storage, not about the world.

## Retention

Retention is erasure on a schedule. Give the engine policies; `engine.sweep()` (which `Engine(sweep_interval_s=...)` runs for you)
applies them:

```python
from datetime import timedelta
from aaa.runtime import RetentionPolicy

engine = Engine(
    sweep_interval_s=60,
    retention=[
        RetentionPolicy(older_than=timedelta(days=365), tenant="acme"),   # acme: a year, then scrub
        RetentionPolicy(older_than=timedelta(days=30), delete=True),      # everyone else: 30 days, then delete
    ],
)
```

- The **first policy whose tenant matches** a run applies, so list tenant-specific policies first and a catch-all last.
- A policy is re-evaluated at most every `interval_s` (default one hour); evaluating scans the run index, so keep it modest on
  very large stores.
- Only runs that **finished** before the cutoff are touched. Runs already scrubbed are skipped (unless the policy deletes).
- Run it in any process, in any number of processes: erasure is idempotent.
- You can also call `await engine.purge(older_than=..., tenant=..., delete=...)` yourself, for a batch job or a one-off.

## Putting it together

For a multi-customer service:

```python
engine = Engine(
    store=await PostgresEventStore.create(dsn),
    blobs=FileBlobStore("/var/lib/aaa/blobs"),
    redactor=standard_redactor(skip=["aaa.run.completed"]),
    require_tenant=True,
    sweep_interval_s=60,
    retention=[RetentionPolicy(older_than=timedelta(days=90))],
)
scope = engine.tenant(authenticated_customer_id)      # from your auth layer, never from the request body
await scope.start(agent, {"task": task})
```

See also [Security](../operations/security.md), [Large payloads](large-payloads.md) and the [event catalog](../reference/event-catalog.md).
