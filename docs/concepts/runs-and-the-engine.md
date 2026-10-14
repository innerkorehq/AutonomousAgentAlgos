# Runs and the engine

The `Engine` is the runtime. It owns the [event store](events-and-the-log.md), the bus, and the bookkeeping that makes
a run bounded: spend, deadlines, cancellation, child runs.

```python
from aaa.runtime import Budget, Engine

async with Engine() as engine:  # in-memory store + in-process bus
    ...
async with Engine(store=SQLiteEventStore("agent.db")) as engine:  # durable
    ...
```

??? note "Engine constructor"
    | Argument | Default | Meaning |
    |---|---|---|
    | `store` | `InMemoryEventStore()` | Where events live. |
    | `bus` | `InProcessBus()` | How events are delivered. Pass a `DistributedBus` for a cluster, a `ClientBus` for a submit-only process. |
    | `schemas` | empty `SchemaRegistry` | Versioned payload validation and upcasting. |
    | `default_budget` | unlimited | Applied to runs started without an explicit `budget=`. |
    | `blobs` | none | A `BlobStore`; enables [claim-check](../guides/large-payloads.md). |
    | `claim_threshold` | `16384` | Bytes of JSON above which an event's payload moves to the blob store. |
    | `sweep_interval_s` | none | Run [`sweep()`](#sweeping) on this period. Recommended in a cluster. |

## Starting a run

```python
outcome = await engine.run(algo, {"task": "..."})  # start and wait

run = await engine.start(algo, {"task": "..."}, budget=Budget(deadline_s=60), run_id="job-42")
outcome = await run.result()
```

`run_id` is optional. **Starting an id that already exists is idempotent**: you get a handle to the existing run
instead of a duplicate. That makes `start` safe to retry after a timeout and gives you natural job keys.

A `RunHandle` is available for *any* run id, from any process: `engine.handle("job-42")`.

| `RunHandle` member | |
|---|---|
| `run_id` | The stream name. |
| `await result()` | The `Outcome`, whichever process hosted or finished the run. |
| `events(hydrate=False)` | Async iterator: replay the log from the start, then follow it until the run completes. |
| `await cancel(reason)` | Ask the engine to end the run as `cancelled`. |
| `await signal(type, data)` | Inject an external event (an approval, a webhook, a user message). |

## Outcomes

A run ends in exactly one `Outcome(kind, value, reason)`. The kinds are distinct on purpose so callers can react
correctly:

| `kind` | Meaning | Typical reaction |
|---|---|---|
| `success` | The algorithm produced a result in `value`. | Use it. |
| `no_solution` | It ran to its limit without an answer (e.g. `max_turns` reached, repair loop exhausted). `value` may hold the best attempt. | Show best-effort or escalate. |
| `timeout` | The deadline passed. | Retry with a bigger budget or give up. |
| `budget_exceeded` | Steps, tokens or cost exceeded their cap. | Same. |
| `invalid_input` | The input could not be handled (e.g. a router chose an unknown route). | Fix the caller. |
| `cancelled` | Someone cancelled, or the parent run ended. | None. |
| `failed` | An unrecoverable error (LLM auth failure, an algorithm exception). `reason` has the cause. | Alert. |

`outcome.ok` is `True` only for `success`. See [Outcomes and errors](../reference/outcomes-and-errors.md).

## Budgets

```python
Budget(max_steps=50, max_tokens=100_000, max_cost=1.50, deadline_s=120)
```

- **Steps** count completed LLM calls plus completed *or failed* tool calls.
- **Tokens** are summed from the `usage` the gateway reports.
- **Cost** is summed from `cost` on each completion: your `price_in_per_mtok`/`price_out_per_mtok` in the
  `LLMConfig`, or the cost LiteLLM computes. If neither exists, cost is `0` and `max_cost` cannot trip.
- **Deadline** is wall-clock seconds from when the run started. It survives restarts: a recovered run keeps the time
  it has already used.

Limits use strict `>`: `max_steps=20` allows 20 steps and trips on the 21st. When one trips, the engine appends
`aaa.budget.exceeded` and completes the run with `budget_exceeded` (or `timeout` for the deadline). An algorithm cannot
opt out.

!!! note "Enforcement is eventual, not preemptive"
    Budgets are checked when events are appended. A single very long LLM call that has already started will finish
    (and be counted) before the run stops. Combine budgets with a client timeout (`LLMConfig.timeout_s`).

## Child runs

Algorithms such as `PlanAndExecute` and `Supervisor` start *child runs*:

```python
await engine.start(worker, {"task": "..."}, parent=ctx.run_id, tag="step-a")
```

- The child is its own stream with its own outcome.
- Its **spend rolls up** into the parent's budget (via `aaa.child.completed`).
- The parent receives `aaa.child.completed` with the child's outcome.
- If the **parent finishes first** (cancelled, timed out, completed), still-open children are cancelled.
- `trace_id` is shared, so one trace covers the whole tree.

## Cancellation

```python
await run.cancel("user pressed stop")
```

appends `aaa.run.cancel`; the engine completes the run as `cancelled` and tools running for it are cancelled. A cancel
issued from *any* process works, because it is just an event.

## External events

`run.signal(type, data)` appends an arbitrary event to the run's stream. Algorithms that subscribe to it react. This is
how [approvals](../guides/human-approval.md), webhooks and [FSM](../guides/algorithms/workflow-fsm.md) transitions arrive.

## Sweeping

`await engine.sweep()` (or `Engine(sweep_interval_s=5)`) performs stateless, idempotent repairs that are safe to run
in any process at any time:

- complete runs whose **deadline** passed but whose owning process died (a dead process cannot fire its own timer);
- cancel **orphaned** children whose parent finished;
- re-announce a **finished child's result** that never reached its parent.

## Recovering after a restart

Single-process: `await engine.recover([algorithm, ...])` re-adopts every open run and redelivers whatever nothing
reacted to. Clusters do not need `recover()`; the bus redelivers on its own. See
[Persistence and recovery](../guides/persistence.md).

## Shutting down

`async with Engine()` drains the bus on exit. `await engine.abort()` is the opposite: a hard stop that models a crash
(nothing drained, no leases released). Use it in tests of your own recovery paths.
