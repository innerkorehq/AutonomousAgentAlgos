# Changelog

## Unreleased

**Observability** (see [the guide](../guides/observability.md))

- `aaa.observe.Observer`: derives traces from the event log (`aaa.trace.spans_from_events`) and exports them whole, parents first, once per cluster, without ever failing a run. Sinks: `InMemorySink`, `ConsoleSink`, `CallableSink`, `OTelSink` and presets `LangfuseSink`, `LangSmithSink`, `PhoenixSink` (extras `otel`, `langfuse`, `langsmith`, `phoenix`). Content capture is opt-in; sampling, tenant filter, incremental export of open sessions, `backfill`.
- Foundations: `aaa.score.Score`, `aaa.safe.safe_subscribe` (isolated, bounded observer workers).

## 0.1.1 (2026-10-15)

**Chat sessions and provider-side state** (see [the guide](../guides/sessions.md))

- `aaa.session.SessionManager` / `Session`: a durable multi-turn conversation. A session is a long-lived run; each user message starts a child run of your algorithm; the reply is recorded by whichever worker finishes the turn, so a turn survives the process that sent it. One turn at a time, per-tenant namespacing, whole-conversation budget, streaming, `context=` for long histories (summaries persisted and reused), and erasure/retention as for any run.
- Provider-held conversation state: `OpenAIResponsesClient` (`provider="openai_responses"`) plus a small `ServerStateClient` protocol. The `LLMHandler` continues from the provider's handle and sends only new messages (within a run and across session turns), falls back to a full request when the state is gone or the context policy rewrote the conversation, and `engine.erase` deletes provider-held state (`ErasureReport.provider_state`).
- `Usage.cached_tokens` and `price_cached_in_per_mtok`; `Engine.await_event`; `FakeStatefulClient` for tests.

**Release process**

- Releases publish to PyPI from GitHub through trusted publishing (no stored token); the release workflow skips a version that is already on PyPI.

## 0.1.0 (2026-10-15)

**Context management** (see [the guide](../guides/context.md))

- `aaa.context.ContextPolicy` + `LLMHandler(client, context=policy)`: each request is fitted to the model's window as a send-time *view*; the logical conversation, the delta requests and the log are untouched.
- Strategies: `TruncateToolResults`, `ClearToolResults`, `Summarize` (incremental; persisted as `aaa.context.summary` events, reused by content hash across restarts and workers, billed to the run), `DropOldest`. Tool calls stay with their results; the head and recent turns are never touched.
- Calibrated token estimates (`CalibratedCounter` learns from `usage.prompt_tokens`; optional `TiktokenCounter`); trigger/target hysteresis; one deeper retry when the provider still reports an overflow.

**MCP client** (see [the guide](../guides/mcp.md))

- `aaa.mcp.MCPClient`: connect to any MCP server over stdio, Streamable HTTP, or your own open `Client`/`ClientSession`, and use its tools as AAA tools (`SchemaTool`: argument validation against the server's JSON Schema, timeouts, approval, execution guard). Works with both the 1.x and 2.x `mcp` SDK.
- Safety policy from MCP annotations: read-only/idempotent tools are `at_least_once`, everything else `at_most_once`; `approval="destructive"` follows the spec's "no annotations means destructive"; `trust_annotations=False`, `include`/`exclude`, result truncation.
- `mcp_memory(client, add=..., search=...)`: any MCP memory server as an AAA `Memory`. Resources can be listed and read.
- Tested against real SDK servers in-process, over stdio and over HTTP with credentials, on both SDK versions in CI.

**Long-term memory** (see [the guide](../guides/memory.md))

- `aaa.memory`: a small `Memory` protocol (`add`, `search`) with optional capabilities (`Updatable`, `Deletable`, `Forgettable`, `Profiled`, `Listable`) and a tenant/user/agent `Scope`.
- Backends: `NativeMemory`, `Mem0Memory` (open source and hosted), `GraphitiMemory` (temporal, point-in-time recall), `SupermemoryMemory` (HTTP), `LangGraphStoreMemory` (any LangGraph store, LangMem records read natively), `CallableMemory` for any other library, `FakeMemory` for tests.
- Event-native: `MemoryHandler` (`aaa.memory.requested` → `completed`/`failed`, replayable, redacted, crash-safe writes), `ToolLoop(recall=True)` automatic recall, `WriteBack`, `memory_tools`.
- Erasure: `engine.forget_memory(scope)`, `TenantScope.forget_memory(user=...)`, and `erase_all()` reaches registered memory backends.
- Also: run metadata (`start(meta=...)`, `TenantScope.start(user=...)`), `ToolContext.tenant`/`meta`, `LLMEmbedder`, optional extras `mem0`, `graphiti`, `langgraph`, `langmem`.
- CI: an `adapters` job replays each adapter's calls against the real libraries' signatures and runs weekly against their newest releases.

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
