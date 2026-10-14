# Status and limitations

AAA is **alpha (0.x)**. This page is deliberately blunt about what is proven, what is not, and what is missing, so you can decide
whether it fits your use.

## What has been verified

| Claim | Evidence |
|---|---|
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
- **Type checking.** `pyright` is a dev dependency but has not been run over the code.

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

If the event store becomes unavailable while a handler is appending, **the reaction can be lost**:

- A tool whose result cannot be appended does not retry; its bus message is acknowledged anyway, so it is not redelivered.
- An exception in an algorithm handler marks the run `failed` (best effort); if that append fails too, the run is left open.

This was reproduced: a 0.5 s outage while a tool was running left the run at `aaa.tool.started` until its deadline. Always set a deadline; run
`sweep_interval_s` in a cluster; drill a database failover in staging. A fix (distinguish infrastructure errors from handler bugs, retry with backoff,
and leave the message unacknowledged when appends keep failing) has been scoped as follow-up work.

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
| **Tenancy, redaction, retention** | `Event.meta` can hold a tenant tag but nothing enforces isolation; no PII scrubbing hook; no automated erasure/retention. See [Security](../operations/security.md). |
| **`Item`/`Codec` consumed natively by the built-in algorithms** | Today they are the rendering layer for your own tools/algorithms; built-ins take plain dict input. See [Data model](../concepts/data-model.md). |
| **Classical planning algorithms** | HTN, A\*/heuristic search over a user-defined domain, PDDL/CP-SAT/MILP solver adapters, MCTS. Deliberately deferred; the data contract for them was designed but not implemented. |
| **Vector database and S3 adapters** | `Index` and `BlobStore` are protocols; only in-memory (and file blobs) ship. |
| **Prompt-caching and summarization helpers** | The remaining cost driver in long conversations. |
| **Automatic recovery from store outages** | See above. |
| **CI test workflow, PyPI release, versioned docs** | Docs are published automatically; the test suite is run locally. |

Not planned unless demand appears: reinforcement-learning policies, POMDP solvers, evolutionary search, distributed consensus
(DCOP). They are production-viable in specific domains but are poor defaults for LLM-agent workloads.

## Compatibility

- Python 3.11, 3.12 (tested on 3.12; the code uses no 3.12-only features).
- Postgres 14+ is expected to work (tested on 18); Redis 6.2+ (Streams, Lua; tested on 8).
- Anything speaking the OpenAI chat-completions dialect.
