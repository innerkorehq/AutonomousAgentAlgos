# Status and limitations

AAA is **alpha (0.x)**. This page is deliberately blunt about what is proven, what is not, and what is missing, so you can decide
whether it fits your use.

## What has been verified

| Claim | Evidence |
|---|---|
| Type-correct public API | `pyright` (basic mode) reports 0 errors over the library, the examples and the scripts, in CI. The tests are not type-checked. |
| Core engine, algorithms, tools, projections, snapshots | Unit and integration tests (280+); every doc example is executed by the suite. |
| Four event stores obey one contract | One parametrized contract suite runs against memory, SQLite, Postgres and Redis (real servers, started on random ports). |
| Distributed bus: ordering, exactly-once competing consumers, broadcast, failover, rebalancing, durable ack | The same bus scenarios run against memory, Postgres and Redis backends. |
| Crash safety | Tests that `SIGKILL` real worker processes, including **mid-tool-call**, on Postgres and Redis, for both delivery policies. |
| Live LLM interaction | `scripts/live_smoke.py` passed 11/11 applicable checks against a **local Ollama** (`gemma4:latest`, 8B): completion, streaming, structured output, native and streamed tool calls, a full `ToolLoop`, `PlanAndExecute` with a real planner, generate-test-repair, delta events, error mapping. |
| Log growth is linear in turns | Benchmark in [Performance](../operations/performance.md). |

## What has **not** been verified

- **Hosted gateways.** Portkey, Cloudflare AI Gateway, Vercel AI Gateway, Bifrost, Kong AI Gateway, MLflow AI Gateway and the LiteLLM proxy
  are supported through documented URLs and headers and **mocked-transport tests only**. Bifrost's and MLflow's base paths vary by version.
  The LiteLLM SDK adapter has not been exercised live. OpenAI's strict structured-output mode has not been tested (a local Ollama may be more lenient).
- **Throughput.** Nothing has been load tested. Correctness under failure is tested; capacity and latency under load are not.
- **Algorithms against a real model:** `ReAct`, `Router`, `Supervisor`, `SelfConsistency`, `GeneratorCritic` and `RAGAgent` are tested with scripted models,
  not live ones.
- **Redis Cluster** (the key layout is cluster-safe by design; only a single node was tested) and **managed Postgres/Redis failover**.

### Verify a hosted gateway yourself

`scripts/live_smoke.py` takes a minute and a few thousand tokens. Point it at your gateway:

```bash
AAA_LIVE_BASE_URL=https://api.portkey.ai/v1 \
AAA_LIVE_MODEL=gpt-4o-mini \
AAA_LIVE_API_KEY="$OPENAI_API_KEY" \
AAA_LIVE_HEADERS='{"x-portkey-api-key": "...", "x-portkey-provider": "openai"}' \
uv run python scripts/live_smoke.py
```

It checks completion + usage, streaming, structured output, native and streamed tool calls, an end-to-end `ToolLoop`, `PlanAndExecute`,
generate-test-repair, delta events, and error mapping (unknown model, bad key, unreachable host). Optional: `AAA_LIVE_EMBED_MODEL` for
embeddings, `AAA_LIVE_CHAT_PATH` for non-standard paths (e.g. `""` for Kong), `AAA_LIVE_TIMEOUT`.

## Known reliability gaps

### Store outages during a run

A temporary outage of the event store (or blob store) does **not** fail a run or lose a reaction:

- Store operations on the handler path are retried with bounded exponential backoff and jitter (`Engine(store_retry=RetryPolicy(...))`,
  default: up to 30 s). Appends are idempotent on event id, so retrying one that actually committed is harmless.
- If an *algorithm or LLM* handler still cannot finish, the engine raises `Redeliver` instead of failing the run, and the bus delivers
  the event again later (in order) until it succeeds. Only a genuine bug in the handler fails the run.
- A *tool result* that cannot be recorded is retried **until it is** (`ToolHandler(result_retry=...)`, default: forever), so a tool that
  already ran is not re-run to get its result recorded. With a bounded policy, the handler gives up and leaves the bus message unacknowledged,
  so the call is redelivered once the store is back (and then follows its [delivery policy](../guides/tool-guard.md)).
- A distributed-bus consumer survives backend errors instead of dying (previously it could die silently while its lease kept renewing).

What this was tested against: scripted outages through a proxy store (in-process and cluster), a backend that errors on every read and
cursor write, and **real dropped connections**: all Postgres backends terminated, and every Redis client killed, repeatedly, while a run was
in flight. It has **not** been tested against a full database server restart or a managed-service failover, a stalled-but-reachable database,
or an outage longer than your retry budgets with a worker restart in the middle. Drill those in staging.

Things to know: while the store is down nothing progresses (events are the only record), a model call whose *result* could not be recorded
within `store_retry` is requested again (you pay for the repeat; raise the budget to avoid it), and `RunHandle.signal()` / `engine.start()`
raise `StoreUnavailable` to the caller after the budget so your API can answer 503.

### Other behavior to be aware of

- **Postgres cannot evict a frozen-but-connected worker.** Its advisory-lock leases live with the session; see [the guard](../guides/tool-guard.md#backend-semantics-important).
- **No exactly-once for external effects.** Only idempotent targets or `at_most_once` tools avoid duplicates after a crash. See [Delivery and idempotency](../concepts/delivery-and-idempotency.md).
- **The fence check is not one atomic store operation** (microsecond window; first-writer-wins on the result id bounds the damage to one recorded result).
- **Budgets are eventual,** not preemptive: a call already in flight finishes and is then counted.
- **Cost accounting needs prices** (or LiteLLM); otherwise `Budget(max_cost=...)` never trips. Token usage depends on the gateway reporting it (streams especially).
- **SQLite supports one engine process per file.**
- **Heterogeneous workers are unsupported:** every worker in a cluster registers the same algorithms and tools.
- **The partition count is fixed** once a cluster is created.
- **Nothing is garbage-collected:** event streams, the bus outbox (without `trim_outbox`), and blobs grow until you manage them.
- **`recover()` is for single-process restarts.** Clusters rely on bus redelivery plus `sweep()`.

## Features not built yet

| Gap | Notes |
|---|---|
| **Observability integration** | No OpenTelemetry spans or metrics yet. The event log carries trace/causation ids; an exporter is straightforward but unwritten. |
| **Hard tenant isolation, encryption** | Tenancy is application-level (shared store; run ids namespaced and ownership-checked). Redaction is pattern-based. Nothing is encrypted by AAA, and erasure does not reach your backups. See [Tenancy, redaction, retention](../guides/privacy.md). |
| **`Item`/`Codec` consumed natively by the built-in algorithms** | Today they are the rendering layer for your own tools/algorithms; built-ins take plain dict input. See [Data model](../concepts/data-model.md). |
| **Classical planning algorithms** | HTN, A\*/heuristic search over a user-defined domain, PDDL/CP-SAT/MILP solver adapters, MCTS. Deliberately deferred; the data contract for them was designed but not implemented. |
| **Provider-side state: OpenAI only, not run live** | Only the OpenAI Responses API is implemented (other providers can plug in via `ServerStateClient`); it is verified against the official SDK's types and a mock transport, not the live API. The provider retains the data (`store=True`). Sessions have no built-in rate limiting or per-user concurrency across sessions. See [Sessions](../guides/sessions.md#provider-side-state). |
| **Context windows are yours to configure** | `ContextPolicy` needs the model's real window (no built-in table, it would go stale). Token counts are calibrated estimates; images are not counted; summaries are lossy. See [Context management](../guides/context.md#limits). |
| **Memory backends verified live** | The Mem0, Graphiti, Supermemory and LangGraph adapters pass a shared contract against stand-ins, and their calls are checked against the real libraries' signatures in CI, but none has been run against a live service. `NativeMemory` is in-process and keyword-only unless given an embedder. See [Long-term memory](../guides/memory.md#what-is-and-is-not-verified). |
| **Vector database and S3 adapters** | `Index` and `BlobStore` are protocols; only in-memory (and file blobs) ship. |
| **Prompt-caching and summarization helpers** | The remaining cost driver in long conversations. |
| **Versioned docs** | Docs are published automatically from `master` and the suite runs in CI; there is one version of the docs, not one per release. The package is on PyPI (0.1.0, alpha). |

Not planned unless demand appears: reinforcement-learning policies, POMDP solvers, evolutionary search, distributed consensus
(DCOP). They are production-viable in specific domains but are poor defaults for LLM-agent workloads.

## Compatibility

- Python 3.11, 3.12 and 3.13, all exercised by the full test suite in CI. (3.11 needed two hardening fixes for cancellation; see the changelog.)
- Postgres 14+ is expected to work (tested on 18); Redis 6.2+ (Streams, Lua; tested on 8).
- Anything speaking the OpenAI chat-completions dialect.
