# Persistence and recovery

By default an `Engine` keeps everything in memory. Give it a durable store and runs survive restarts; add `recover()` and
a crashed process's open runs are continued.

## Choosing a store

| Store | Durability | Processes | Use for |
|---|---|---|---|
| `InMemoryEventStore` | none | one | tests, notebooks, short-lived scripts |
| `SQLiteEventStore(path)` | disk (WAL) | **one engine** (readers are fine) | a service on one machine, local development |
| `PostgresEventStore.create(dsn)` | database | many | production, [clusters](clusters.md) |
| `RedisEventStore.create(url)` | your Redis persistence settings | many | low-latency clusters where Redis already runs |

All implement the same `EventStore` protocol and pass the same contract tests, so switching is a one-line change.

```python
from aaa.events import SQLiteEventStore

store = SQLiteEventStore("agent.db")  # creates the file and schema
async with Engine(store=store) as engine:
    ...
await store.aclose()
```

### SQLite specifics

- Pure standard library (no extra install). One dedicated thread owns the connection, so the event loop never blocks on disk.
- **WAL mode, `synchronous=NORMAL`**: crash-safe (the file cannot be corrupted), but a *power loss* may lose the last few
  commits (never reorders or tears them). For strict durability pass `SQLiteEventStore(path, synchronous="FULL")`.
- Appends are single `IMMEDIATE` transactions: the optimistic-concurrency check and the inserts are atomic, and a failed
  append leaves nothing behind.
- **One engine process per database file.** The in-process bus has no cross-process coordination; two engines on one
  file would both react to the same events. For multiple processes use Postgres or Redis.

## Recovering after a crash

```python
--8<-- "examples/persistence_recovery.py"
```

```text
process 1 died with the run open
process 2 recovered 1 run(s): result: value-of-x
```

`await engine.recover([algo, ...])` returns `{run_id: RunHandle}` for every run it re-adopted. Pass **every algorithm
that might have open runs** (children included, e.g. the `PlanAndExecute` *and* its executor).

What `recover()` does:

1. **Re-adopts** each open run: rebuilds its budget, spend and deadline from the log. A deadline keeps counting through the
   outage, so a run that should have timed out is completed as `timeout` immediately.
2. **Repairs** relationships: cancels orphaned children, re-announces a finished child's result that never reached its parent.
3. **Redelivers** every event that nothing reacted to (an event is "reacted to" if some other event names it as its cause).
   Because algorithms decide from state and derived ids are deterministic, repeating a reaction is absorbed.

`engine.last_recovery` is a `RecoveryReport`: `recovered`, `unknown_algo`, `orphans_cancelled`, `timed_out`, `announced`,
`redelivered`.

!!! note "Open runs whose algorithm you did not pass"
    They are left untouched and listed in `last_recovery.unknown_algo`. Recovery needs **unique algorithm names**
    (it matches by `name`) and raises `ValueError` if two different instances share one.

### What happens to work in flight

| In flight at the crash | After recovery |
|---|---|
| An LLM request with no response | Requested again (you pay for the repeated call). |
| A tool call that had **not** started | Runs. |
| A tool call that **had started** but had no result | Depends on its [delivery policy](tool-guard.md): re-run as attempt 2 (`at_least_once`) or reported as `OutcomeUnknown` (`at_most_once`). |
| A tool call that finished | **Not** repeated: its result is on the log. |
| A pending approval | Still pending; it is just events. |
| A running child run | Resumed with its parent. |

Clusters do not call `recover()`: the distributed bus redelivers by itself. See [Running a cluster](clusters.md).

## Growth and retention

- **Nothing is deleted automatically.** The log is append-only; completed runs stay. Plan retention for your volume:
  archive or delete finished streams with SQL on your own schedule (a finished run's stream is self-contained).
- For the distributed bus, trim the **outbox** periodically (`store.trim_outbox(keep_last=...)`; see
  [Running a cluster](clusters.md#maintenance)).
- Blob stores are content-addressed and never garbage-collected by AAA; see [Large payloads](large-payloads.md).
- Long conversations are linear in size ([Performance](../operations/performance.md#long-conversations)).

## Exporting events

`aaa.events.cloudevents.to_cloudevent(event)` yields a CloudEvents 1.0 dict you can ship to Kafka, a warehouse or a
tracing pipeline. See [Events and the log](../concepts/events-and-the-log.md#cloudevents-interoperability).
