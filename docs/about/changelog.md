# Changelog

## 0.0.1 (unreleased)

**Privacy: tenancy, redaction, retention** (see [the guide](../guides/privacy.md))

- `Event.tenant`, inherited by every derived event; `engine.tenant(name)` scopes that namespace run ids and check ownership on every lookup; `Engine(require_tenant=True)`.
- `Engine(redactor=...)` with `PatternRedactor` (e-mail, Luhn-checked cards, SSN, API keys, JWT, bearer), `KeyRedactor` and `standard_redactor()`: applied before persistence and delivery.
- `engine.erase` (scrub or delete), `engine.purge` and `Engine(retention=[RetentionPolicy(...)])` applied by `sweep()`. `EventStore` gained `scrub` and `delete_stream` (all four backends), which also remove
  snapshots and the bus outbox copies. Claim-checked blobs are now namespaced per stream so erasing one run cannot affect another.

**Quality gates**

- CI: ruff, `pyright` (0 errors over the library, examples and scripts) and the full suite on Python 3.11 and 3.12 against real Postgres and Redis.
- Type-correctness fixes found by pyright: the `EventBus` and `EventStore` protocols now declare what the engine and users rely on (`aclose`, `drain`, `abort`, `execution_guard`);
  `Algorithm.on_event` and `LLMClient.complete/stream` take positional-only parameters so any implementation naming (`e`, `request`, ...) type-checks (previously every built-in
  algorithm was rejected when passed to `Engine.run`); `Outcome.value` no longer forces narrowing before indexing.
- Python 3.11: long-lived loops could outlive a cancellation (the standard library's `asyncio.timeout`, used by redis-py, can swallow a cancel that races its deadline), which made
  `abort()`/shutdown hang with Redis. Loops now re-raise a swallowed cancel, and `aaa.core.wait_for`/`wait_event` replace `asyncio.wait_for` where tasks are shut down by cancellation.

**Resilience**

- Store outages no longer fail runs or lose reactions: bounded retry with backoff (`Engine(store_retry=...)`), `Redeliver` so the bus delivers an event again, a tool
  result retried until recorded (`ToolHandler(result_retry=...)`), and distributed-bus consumers that survive backend errors. Fixed a race where a tool task cancelled before its first step
  left its bus message unacknowledged. Tested with real dropped Postgres and Redis connections.

First public version. Alpha: see [Status and limitations](limitations.md).

**Runtime and data**

- Event-sourced engine: append-only log, deterministic ids, idempotent appends, budgets, deadlines, cancellation, child runs with spend roll-up, signals.
- Projections with versioned snapshots; incremental state for every built-in algorithm.
- Event stores: in-memory, SQLite, PostgreSQL, Redis, all passing one contract suite. CloudEvents 1.0 conversion. Versioned payload schemas with upcasters.
- Claim-check blob storage (in-memory, file) with integrity verification. In-memory hybrid (BM25 + vector) index.
- `Item`/`Codec` data layer with capability preflight.

**Delivery and clustering**

- In-process bus and a partitioned `DistributedBus` over Postgres (transactional outbox, `LISTEN/NOTIFY`, advisory-lock leases) and Redis (Streams + Lua), with consumer groups,
  cursors, fair-share rebalancing, durable acknowledgement and `ClientBus` for submit-only processes.
- Single-process crash recovery (`recover()`), cluster `sweep()`, adoption of any run by any worker.

**LLM layer**

- OpenAI-compatible client with streaming, tool calls and structured output; presets for LiteLLM proxy, Portkey, Cloudflare, Vercel, Bifrost, Kong, MLflow; LiteLLM SDK adapter;
  retry and circuit-breaker wrappers; typed errors; fake and replay clients; delta requests so long conversations stay linear.

**Algorithms**

- `ToolLoop`, `ReAct`, `PlanAndExecute`, `Router`, `Supervisor`, `GenerateTestRepair`, `GeneratorCritic`, `WorkflowGraph`, `FSM`, `SelfConsistency`, `RAGAgent`.

**Tools**

- Typed `@tool` with schema generation, pools, timeouts, human approval, and the **execution guard** (durable acknowledgement, execution lease, fencing attempts,
  `at_least_once`/`at_most_once`, idempotency keys), fully configurable globally and per tool.

**Docs and tooling**

- This documentation site, with every example executed by the test suite. License: MIT.
