# AutonomousAgentAlgos (AAA)

Async, **event-driven** library of production-deployable agent algorithms that run over
**arbitrary user data**, with a pluggable LLM gateway layer.

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
    LLMHandler(client).attach(eng); ToolHandler(registry).attach(eng)
    handles = await eng.recover([ToolLoop(registry), PlanAndExecute(...)])   # after a crash
```

**Multi-process** (`DistributedBus` over Postgres or Redis; run the same worker code N times):
```python
store = await PostgresEventStore.create(DSN, partitions=16)      # or RedisEventStore.create(URL, ...)
bus = DistributedBus(store.bus_backend())
async with Engine(store=store, bus=bus, sweep_interval_s=5) as eng:
    LLMHandler(client).attach(eng); ToolHandler(registry).attach(eng); eng.register(ToolLoop(registry))
    await asyncio.Event().wait()                                  # a worker
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

### Guarantees and caveats (read these)
- **Tools run at-least-once.** Work in flight when a process dies is requested again; finished work
  is not redone. Make side-effecting tools idempotent or approval-gated.
- **No fencing tokens.** A worker that stalls past its lease can briefly overlap with the new owner;
  the duplicate effects are absorbed by idempotent appends, but a non-idempotent tool could run twice.
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
- Tested with real separate OS processes and `SIGKILL` on Postgres and Redis; not yet load tested.

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
- Each LLM request carries the full message list (quadratic log growth; claim-check bounds event size).
- No sandboxed code-execution validator is shipped (deliberately).
- Structured-output `strict` mode may need schema adjustments per provider.
- Only a local Ollama has been tested live; hosted gateways are untested (see above).
