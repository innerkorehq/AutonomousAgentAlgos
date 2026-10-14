# Delivery and idempotency

AAA delivers events **at least once**. Messages can arrive twice (after a crash, after a worker stalls past its
lease, after a rebalance). Rather than promising exactly-once delivery, which no distributed system can honestly give,
AAA makes duplicates harmless. This page lists the mechanisms so you can reason about your own code.

## What the library guarantees

| Mechanism | Guarantee |
|---|---|
| **Append before deliver** | A handler never sees an event that is not already durable (with a durable store). |
| **Idempotent append** | An event id appears at most once per stream. Re-appending returns the existing event with `dup=True`. |
| **Deterministic ids** | Events derived from a parent get ids computed from `(parent, type, data)`; re-deriving them after a crash produces the same ids. |
| **One result per request** | The result of request `R` has id `R:result`. Two workers racing on `R` yield one recorded result. |
| **Level-triggered algorithms** | An algorithm decides from *folded state*, not from "the event that just arrived", so seeing an old event again changes nothing. |
| **Serial per algorithm** | An algorithm handles its events one at a time, in log order. |
| **Atomic enqueue** | With the Postgres/Redis stores, the bus entry is written in the same transaction/script as the event, so a stored event cannot fail to be published. |
| **Durable acknowledgement** | Long-running work (a tool call) keeps its bus message unacknowledged until it finishes, so a crash redelivers it. |

## What it cannot guarantee

!!! danger "External side effects are not exactly-once"
    If a tool sends an email and the worker dies before recording the result, the call will be run again (under
    `at_least_once`) and the email goes out twice. No library can fix this from the outside, because the effect
    happened in a system AAA does not control. Your options, from strongest to weakest:

    1. **Make the effect idempotent** using `ctx.idempotency_key` (Stripe `Idempotency-Key`, SES dedupe,
       `INSERT ... ON CONFLICT`). Then repeats are free.
    2. **Declare `delivery="at_most_once"`**: a lost attempt is reported as `OutcomeUnknown` for a human or the
       algorithm to reconcile, and is never re-run.
    3. **Gate it with human approval** (`requires_approval=True`).

    The [tool execution guard](../guides/tool-guard.md) is the machinery behind 1 and 2.

## Reading the guarantees as a recipe

- Write **tools** to be idempotent, or declare `at_most_once`.
- Write **algorithms** as `(event, projected state) -> events`, with no hidden state and no I/O.
- Never rely on the *count* of times a handler runs.
- When you add an event type, give any event that could be re-derived a deterministic id (use `event.caused(...)`).

## Ordering, precisely

| Scope | Order |
|---|---|
| Within a stream (a run) | Total, by `seq`. |
| An algorithm's view of its own events | Log order, one at a time. |
| Across runs | None. |
| Capability handlers (LLM, tools) | A handler may process several requests of one run concurrently (that is the point of parallel tool calls); each result is its own event. |
| In a cluster | A run's events are consumed in order by whichever worker owns its partition. Ownership can move; the cursor guarantees continuity, and duplicates around a move are absorbed by idempotent appends. |

## In-process vs. durable

With the default in-memory store and in-process bus, nothing survives a crash and delivery is effectively once. Every
guarantee above that mentions durability starts to apply as soon as you use `SQLiteEventStore`, `PostgresEventStore`
or `RedisEventStore`. Redis durability additionally depends on your Redis persistence settings (see
[Running a cluster](../guides/clusters.md#redis)).
