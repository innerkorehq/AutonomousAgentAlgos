# AutonomousAgentAlgos (AAA)

Async, **event-driven** library of production-deployable agent algorithms that run over
**arbitrary user data**, with a pluggable LLM gateway layer.

**Documentation: <https://innerkorehq.github.io/AutonomousAgentAlgos/>** — quickstart, concepts, guides for every
algorithm, clustering, the tool execution guard, operations, security, API reference, and an honest
[status and limitations](https://innerkorehq.github.io/AutonomousAgentAlgos/about/limitations/) page.
The README below is a condensed overview; the docs are the reference.

```python
async with Engine() as eng:
    LLMHandler(build_client(gateways.portkey("gpt-4o"))).attach(eng)
    ToolHandler(registry).attach(eng)
    outcome = await eng.run(
        ToolLoop(registry), {"task": "..."}, budget=Budget(max_cost=0.50, deadline_s=60)
    )
```

## Design decisions (data first)
1. **The event log is the source of truth.** Every run is an append-only stream of immutable
   CloudEvents-compatible `Event`s. State, plans, conversation, spend are folds of the log.
   Algorithms hold *no* run state, so runs replay, resume and audit by construction.
2. **User data is a payload, never the algorithm's type.** `Item[T]` + per-`kind` `Codec`
   (Pydantic / dataclass / dict / str / custom `FunctionCodec`) render, serialize and
   fingerprint any type. Algorithms ask only for the capabilities they need.
3. **Large payloads are claim-checked** into a content-addressed `BlobStore`; events stay small.
   Handlers and algorithms see hydrated data transparently.
4. **Failures are facts, not exceptions.** `aaa.llm.failed`, `aaa.tool.failed` are events the
   algorithm reasons about. Outcomes are distinct: success / no_solution / timeout /
   budget_exceeded / invalid_input / cancelled / failed.
5. **Safety lives outside the model**: budgets, deadlines, approval gates, and idempotency are
   enforced by the Engine/ToolHandler, not by prompts.
6. **Ordering**: each algorithm sees its events serially in log order; delivery is in-process
   at-most-once (distributed at-least-once adapters are future work).
7. **Durable and crash-resumable.** `SQLiteEventStore` (stdlib, WAL, atomic OCC append,
   idempotent on event id) persists the log. After a crash, `engine.recover([...algorithms])`
   rebuilds budgets, spend and deadlines from the log, then redelivers events nothing reacted to.
   This is safe because algorithms decide from folded state, and every derived event has a
   deterministic id, so a repeated reaction is absorbed by the store instead of duplicating.

## Layout
`core` ids/outcomes/errors · `events` envelope, stores (memory/SQLite/Postgres/Redis), buses (in-process/distributed), projections+snapshots, schemas,
CloudEvents · `data` Item/Codec/BlobStore/Index · `llm` client protocol, OpenAI-compatible
adapter + gateway presets, LiteLLM, retry/breaker, streaming, fakes · `tools` typed registry +
handler · `validate` deterministic validators · `runtime` Engine/Budget · `algos`.

## Algorithms (v0.1)
ToolLoop, ReAct, PlanAndExecute (DAG, parallel child runs, replan), Router, GenerateTestRepair,
GeneratorCritic, WorkflowGraph (deterministic DAG), FSM, SelfConsistency, RAGAgent, Supervisor.

## Persistence, snapshots, and running as a cluster

| Store | Use for | Notes |
|---|---|---|
| `InMemoryEventStore` | tests, notebooks | nothing survives the process |
| `SQLiteEventStore` | one machine, durable | stdlib, WAL; single engine process |
| `PostgresEventStore` | production system of record | `pip install autonomous-agent-algos[postgres]` |
| `RedisEventStore` | low-latency, ephemeral-ish logs | `[redis]`; durability = your Redis AOF settings |

**Single process, restart safe** (`InProcessBus` + a durable store):
```python
async with Engine(store=SQLiteEventStore("agent.db")) as eng:
    LLMHandler(client).attach(eng)
    ToolHandler(registry).attach(eng)
    handles = await eng.recover([ToolLoop(registry), PlanAndExecute(...)])  # after a crash
```

**Multi-process** (`DistributedBus` over Postgres or Redis; run the same worker code N times):
```python
store = await PostgresEventStore.create(DSN, partitions=16)  # or RedisEventStore.create(URL, ...)
bus = DistributedBus(store.bus_backend())
async with Engine(store=store, bus=bus, sweep_interval_s=5) as eng:
    LLMHandler(client).attach(eng)
    ToolHandler(registry).attach(eng)
    eng.register(ToolLoop(registry))
    await asyncio.Event().wait()  # a worker
# anywhere else (API server, CLI): submit and observe only
client = Engine(store=store, bus=ClientBus())
outcome = await client.run(ToolLoop(registry), {"task": "..."})
```
How it works: an event is appended to the store *and* the bus outbox in one atomic step (a
transaction in Postgres, one Lua script in Redis), so nothing stored can fail to be published. Events
go to partition `crc32(run_id) % P`. Within a consumer *group* (an algorithm, the LLM handler, the
tool handler...) each partition is owned by exactly one process at a time, consumed in order, and
acknowledged with a cursor; processes joining or dying rebalance partitions. Delivery is
at-least-once; derived events have deterministic ids and results are one-per-request
(`<request id>:result`), so repeats are absorbed by the store's idempotent append. Any process can
host any run: engine state (spend, completion, open children) is a fold of the log, rebuilt on
demand from a snapshot plus the tail.

**Snapshots**: every algorithm's state is an incremental `Projection`; state is snapshotted every 64
events and a load costs O(new events). A 40-turn run does ~16x fewer state folds, and the gap
grows with run length. A snapshot is derived data: a corrupt, stale-versioned, or too-new one is
ignored and the state is replayed from the log. Bump a projection's `version` if you change its fold.

### Tool execution guard (stalled and crashed workers)
Bus delivery is at-least-once, and a worker can die or stall (GC pause, VM freeze, network partition)
while a tool call is running. Three layers protect against a repeated side effect; each is configurable.

| Layer | What it does |
|---|---|
| **Durable ack** | The request stays unacknowledged on the bus until the tool finishes, so a worker that dies mid-call has the call redelivered to another. |
| **Exclusion** | A per-call execution lease with a heartbeat: two workers never run the same call concurrently while the first is alive. |
| **Fencing** | Every run is a numbered attempt recorded *before* it executes (`aaa.tool.started`, id `<request>:attempt:<n>`). A worker that loses its lease cancels its tool and records nothing; a result from a superseded attempt is rejected (`aaa.tool.stale_result`). |
| **Effect safety** | The tool receives `ctx.idempotency_key` (stable across attempts and workers), `ctx.attempt` and `ctx.fence` to make the external effect idempotent. |

```python
from aaa.tools import GuardConfig, ToolContext, ToolHandler, tool

# global default for every tool
ToolHandler(registry, guard=GuardConfig(lease_ttl_s=30, max_attempts=3)).attach(engine)


@tool  # inherits the default: re-run a lost call, up to 3 attempts
async def search(q: str) -> list: ...


@tool(delivery="at_most_once")  # never re-run once an attempt started
async def send_email(to: str, body: str, ctx: ToolContext) -> str:
    return await ses.send(to, body, dedupe_key=ctx.idempotency_key)


@tool(
    idempotency_key=lambda a: f"refund:{a['order_id']}",  # business-level key, stable across
    guard={"max_attempts": 1, "lease_ttl_s": 60},
)  # the model re-issuing the call
async def refund(order_id: str, ctx: ToolContext) -> str: ...
```

| `GuardConfig` field | Default | Meaning |
|---|---|---|
| `enabled` | `True` | `False`: no lease, no attempt records (the bus still redelivers a lost request, so a tool can run twice and nothing notices). |
| `delivery` | `at_least_once` | `at_least_once`: re-run a lost call up to `max_attempts`. `at_most_once`: a lost attempt becomes an `OutcomeUnknown` failure (the effect may or may not have happened) for the algorithm or a human to reconcile. |
| `max_attempts` | `3` | Cap on executions of one call; beyond it the call fails with `ExecutionAbandoned` (a poison call must not loop forever). |
| `lease_ttl_s` / `heartbeat_s` | `15` / `ttl/3` | How long a silent worker keeps its claim / how often it renews. Lower = faster takeover but more false positives from slow heartbeats. |
| `wait_poll_s` | `0.5` | How often a worker that found the call busy re-checks. |
| `on_lost_lease` | `cancel` | `cancel`: a worker that wakes up and finds it lost the lease stops its tool. `finish`: let it complete (its result is still fenced). |
| `fence_results` | `True` | Reject (and audit) a result from a superseded attempt. `False`: the first finisher wins. |

Per-tool settings override the handler default; inconsistent combinations (e.g. `heartbeat_s >= lease_ttl_s`)
fail at startup. **Backend semantics differ**: Redis and in-memory leases expire by TTL, so a *frozen*
worker is replaced after `lease_ttl_s`. Postgres leases are advisory locks bound to the database session:
a worker that crashes or loses its connection frees them instantly, but one that is frozen *and still
connected* keeps them, so nobody takes over until it resumes or the connection dies (use TCP keepalives /
`idle_session_timeout`, or Redis, if you need TTL takeover of frozen processes). The fence check
(`has(attempt n+1)` then append) is not one atomic store operation; its residual window is microseconds
and the result id is first-writer-wins, so the worst case is one recorded result, never two.

### Guarantees and caveats (read these)
- **Tool execution is guarded, and the guard is configurable** (next section). The library makes a
  call run *at most once at a time* and *record exactly one result*; it cannot make an arbitrary
  external effect (an email, a payment) exactly-once. That part belongs to the tool, via the
  idempotency key the guard hands it.
- **Partition count is fixed** for the life of a cluster (checked on connect). Changing it needs a new
  cluster or migration. Pick headroom (default 16).
- **Algorithm names must be unique and identical on every worker**, and every worker in a group must
  register the same tools. Heterogeneous worker pools are not supported yet.
- **Outbox growth**: the bus log is not trimmed automatically. Call `store.trim_outbox(keep_last=...)`
  from a periodic job, sized to your longest tolerable consumer outage.
- **Postgres**: a dropped LISTEN connection degrades latency to polling, never correctness. Appends to
  the same partition serialize on one row lock (P-way write parallelism).
- **Redis**: with no persistence a restart loses the log. Works with Redis Cluster (hash-tagged keys)
  but is only tested on a single node.
- Each consumer group scans all events of its partitions (read amplification ~ number of groups).
- Algorithms' `Engine.sweep()` (set `sweep_interval_s`) enforces deadlines of dead owners and repairs
  lost child announcements; without it a run with a deadline whose host died waits for its next event.
- Tested with real separate OS processes and `SIGKILL` (including mid-tool-call) on Postgres and Redis;
  not yet load tested.

## Scaling with run length
A continuing LLM request stores only the *new* messages plus a pointer to the previous request
(`conv: {prev, delta}`); algorithm state holds a request id, not the transcript. The LLM handler
rebuilds the full transcript the provider needs from an in-memory cache, falling back to one pass
over the run's log after a restart/failover/different worker. Tool-call idempotency is an O(1)
`store.has(stream, id)` lookup. `scripts/bench_growth.py` (SQLite, fake LLM, ~400-byte tool results):

| turns | log size before -> after | wall time before -> after |
|---|---|---|
| 50 | 999 KB -> 170 KB | 0.22 s -> 0.07 s |
| 100 | 3,667 KB -> 337 KB | 1.17 s -> 0.14 s |
| 200 | 14,032 KB -> 673 KB | 7.53 s -> 0.28 s |
| 400 | (not measured) -> 1,344 KB | -> 0.56 s |

Log size, state size and per-turn work are now constant per turn (doubling the turns doubles the
totals). What is *not* reduced is the provider payload itself: each call must still send the whole
transcript, so token cost per call grows with the conversation. Use provider prompt caching or
summarization for very long conversations. Custom algorithms should build continuing requests with
`llm_request(..., prev=<previous request id>)` and pass only the new messages.

## Live gateway smoke test
`scripts/live_smoke.py` checks the library against a real OpenAI-compatible endpoint: completion +
usage, streaming, structured output, native tool calls (also streamed), a full ToolLoop through the
Engine, PlanAndExecute with a real planner, generate-test-repair, delta events, and error mapping
(unknown model, bad key, unreachable host). It spends a few thousand tokens. Configure with env vars:
```bash
AAA_LIVE_BASE_URL=http://127.0.0.1:11434/v1 AAA_LIVE_MODEL=gemma4:latest uv run python scripts/live_smoke.py
# a cloud gateway: use the base URL / headers from aaa.llm.gateways, e.g. Portkey
AAA_LIVE_BASE_URL=https://api.portkey.ai/v1 AAA_LIVE_MODEL=gpt-4o-mini AAA_LIVE_API_KEY=$OPENAI_API_KEY \
AAA_LIVE_HEADERS='{"x-portkey-api-key": "...", "x-portkey-provider": "openai"}' uv run python scripts/live_smoke.py
```
Verified so far: a local Ollama (`gemma4:latest`, 8B) — 11/11 applicable checks. Not yet verified
against any hosted gateway (Portkey, Cloudflare, Vercel, Bifrost, Kong, MLflow, LiteLLM proxy): their
base URLs and headers follow vendor docs and are covered only by mocked-transport tests.

## Known limitations
- No sandboxed code-execution validator is shipped (deliberately).
- Structured-output `strict` mode may need schema adjustments per provider.
- Only a local Ollama has been tested live; hosted gateways are untested (see above).

## License

[MIT](https://github.com/innerkorehq/AutonomousAgentAlgos/blob/master/LICENSE) © 2026 Innerkore Technologies.
