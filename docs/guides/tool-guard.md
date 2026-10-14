# The tool execution guard

Tools are where an agent touches the real world: send the email, issue the refund, deploy the build. Delivery is
at-least-once, and a worker can die or stall (GC pause, VM freeze, network partition) while a call is running. This guide
shows what the library guarantees about such calls, and exactly what is left to your tool.

!!! abstract "The guarantee, stated precisely"
    For each tool call, AAA ensures **at most one worker runs it at a time while that worker is alive and holds its lease**,
    records **exactly one result**, and gives every run a **numbered, durable attempt record** (the fencing token). Under
    `at_most_once` it additionally **never starts a second attempt**. It cannot make an arbitrary external effect exactly-once:
    that requires the called system to deduplicate, which is what `ctx.idempotency_key` is for.

## Four layers

| Layer | What it does | Why |
|---|---|---|
| **Durable ack** | The bus message for a tool request stays unacknowledged until the tool finishes. | A worker dying mid-call has the request redelivered. (Without it the call would be silently lost.) |
| **Exclusion** | A per-call execution **lease** with a heartbeat. | Two live workers never run the same call concurrently. |
| **Fencing** | Each run is a numbered **attempt**, recorded (`aaa.tool.started`, id `<request>:attempt:<n>`) *before* it executes. A worker that loses its lease cancels its tool; a result from a superseded attempt is rejected. | A stale worker cannot overwrite or duplicate what the new attempt did in the log. |
| **Effect safety** | The tool receives `ctx.idempotency_key`, `ctx.attempt`, `ctx.fence`. | Lets the *external* system deduplicate. |

## Configuration

Set a default for all tools on the handler and override per tool.

```python
--8<-- "examples/tool_guard.py"
```

```text
refund  -> at_most_once | max_attempts = 3
search  -> at_least_once | max_attempts = 3
refunded A-100 (key=refund:A-100, attempt=1, fence=1)
```

```python
from aaa.tools import GuardConfig, ToolHandler, tool

ToolHandler(registry, guard=GuardConfig(lease_ttl_s=30, max_attempts=3)).attach(
    engine
)  # default for every tool


@tool  # inherits the default
async def search(q: str) -> list: ...


@tool(delivery="at_most_once")  # one-line override
async def send_email(to: str, body: str) -> str: ...


@tool(
    guard={"max_attempts": 1, "lease_ttl_s": 60},  # several overrides
    idempotency_key=lambda a: f"refund:{a['order_id']}",
)  # a business key
async def refund(order_id: str, ctx: ToolContext) -> str: ...
```

| `GuardConfig` field | Default | Meaning |
|---|---|---|
| `enabled` | `True` | `False`: no lease, no attempt records. The bus still redelivers a lost request, so a call can run twice and nothing notices. |
| `delivery` | `at_least_once` | `at_least_once`: re-run a lost call, up to `max_attempts`. `at_most_once`: never re-run; a lost attempt becomes an `OutcomeUnknown` failure. |
| `max_attempts` | `3` | Cap on executions of one call under `at_least_once`. Exceeding it fails the call with `ExecutionAbandoned`, so a call that keeps killing workers cannot loop forever. |
| `lease_ttl_s` | `15.0` | How long a silent worker keeps its claim. Smaller = faster takeover, more false alarms from slow heartbeats. |
| `heartbeat_s` | `lease_ttl_s / 3` | Renewal period. Must be smaller than the TTL. |
| `wait_poll_s` | `0.5` | How often a worker that found the call busy re-checks. |
| `on_lost_lease` | `cancel` | `cancel`: a worker that discovers it lost the lease cancels its tool and records nothing. `finish`: let it complete (its result is still subject to `fence_results`). |
| `fence_results` | `True` | Reject, and audit as `aaa.tool.stale_result`, a result from a superseded attempt. `False`: the first finisher wins. |

Configuration is validated early: an invalid value fails when the tool is *defined*; a combination that only conflicts
with the handler default (say `heartbeat_s >= lease_ttl_s`) fails at `attach()`, not at 3 a.m.

## What happens in each failure

```mermaid
sequenceDiagram
    participant A as Worker A
    participant L as Lease + log
    participant B as Worker B
    A->>L: acquire lease; append tool.started (attempt 1)
    A->>A: run tool...
    Note over A: A stalls or dies
    B->>L: request redelivered. Lease busy: wait
    Note over L: lease expires
    B->>L: acquire lease; append tool.started (attempt 2)
    B->>B: run tool (attempt 2)
    alt A wakes up while B runs (on_lost_lease=cancel)
        A->>L: heartbeat fails
        A->>A: cancel tool, append tool.fenced
    else A finishes first (fence_results=True)
        A->>L: attempt 2 exists: append tool.stale_result, no result
    end
    B->>L: append tool.completed (the only result)
```

| Situation | `at_least_once` | `at_most_once` |
|---|---|---|
| Worker dies mid-call | Another worker runs attempt 2 (up to `max_attempts`). | The call fails with `OutcomeUnknown`: it may already have taken effect; nothing re-runs it. |
| Worker stalls past its lease, then wakes | Its tool is cancelled (`cancel`), or its result is fenced out if a newer attempt exists. | Same; no newer attempt is ever started. |
| Call keeps killing workers | After `max_attempts`: `ExecutionAbandoned`. | Not applicable: only one attempt. |
| Normal run | One attempt, one result. | One attempt, one result. |

The `OutcomeUnknown` message the model receives explains that the action may have happened and must not simply be retried;
well-built prompts tell the model to verify (e.g. look the refund up) instead.

## Choosing a policy

| Tool kind | Recommended |
|---|---|
| Pure reads, searches, calculations | `at_least_once` (default). Repeats are free. |
| Idempotent writes (`PUT`, upserts, anything keyed) | `at_least_once`, and pass `ctx.idempotency_key` to the target. |
| Sends and payments you can deduplicate at the provider (Stripe `Idempotency-Key`, SES dedupe) | `at_least_once` **with** the key passed. Repeats become no-ops. |
| Sends and payments you cannot deduplicate | `at_most_once`, plus `requires_approval=True` if a human should confirm. |
| Destructive actions | `at_most_once` and `requires_approval=True`. |

## Idempotency keys

`ctx.idempotency_key` is **stable across attempts and workers** for the same logical call. The default is `<run id>/<call id>`.
Supply `idempotency_key=lambda args: ...` for a *business* key, which also survives the model re-issuing the call under a
new call id:

```python
@tool(idempotency_key=lambda a: f"refund:{a['order_id']}")
async def refund(order_id: str, ctx: ToolContext) -> str:
    return await stripe.refunds.create(order=order_id, idempotency_key=ctx.idempotency_key)
```

`ctx.attempt` and `ctx.fence` are the 1-based execution number, rising on each re-run. Use `fence` with systems that support
conditional writes ("only apply if my fence is the highest seen").

## Events

| Event | When |
|---|---|
| `aaa.tool.started` | An attempt is claimed, before the tool runs. `{call_id, name, attempt, worker, idempotency_key}`. |
| `aaa.tool.completed` / `aaa.tool.failed` | The one result. |
| `aaa.tool.fenced` | A worker lost its lease mid-call and cancelled its tool. |
| `aaa.tool.stale_result` | An attempt finished after a newer one began; its outcome is kept here for audit and not applied. |

## Backend semantics (important)

The execution lease uses the same primitive as partition ownership, so it inherits the backend's behavior:

| Backend | Lease lifetime | Frozen but connected worker |
|---|---|---|
| In-memory / Redis | Expires after `lease_ttl_s` unless renewed. | Replaced after the TTL; if it wakes, its renewal fails. |
| Postgres | Bound to the database **session** (advisory lock). Freed instantly when a worker dies or loses its connection. | **Keeps the lease**: nobody takes over, so no duplicate and no progress until it resumes or the connection dies. Use TCP keepalives / `idle_session_timeout`, or Redis, if you need to evict frozen processes. |

## Limits (be honest with yourself)

- **Not exactly-once for external effects.** If the effect happened and the worker died before recording the result,
  `at_least_once` re-runs it. Idempotency at the target is the only fix; `at_most_once` converts the risk into an explicit `OutcomeUnknown`.
- **A stalled worker's tool may keep running briefly.** Between waking up and its next heartbeat (up to `heartbeat_s`) the tool body can
  still execute. Keep `heartbeat_s` small relative to how bad a duplicate is, or make the tool idempotent.
- **The fence check is not one atomic store operation** (`has(attempt n+1)`, then append). Its window is microseconds and the result id is
  first-writer-wins, so the worst case is one recorded result, never two.
- A tool running with `enabled=False` has no lease, attempt record or fencing (the durable acknowledgement still applies, so a lost request is redelivered, and nothing detects a duplicate execution).
