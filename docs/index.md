<div align="center" markdown>

# AAA

## Autonomous Agent Architecture

**The architecture and runtime for autonomous agents.**

Build · Orchestrate · Execute · Recover · Manage

</div>

**An async, event-driven architecture for production agents: the algorithms, the durable runtime, tools, memory, evals and guardrails, over data of any shape, with any LLM gateway.**

`pip install autonomous-agent-architecture` · `import aaa` · "AAA" is the short name.

AAA is for the part of an agent system that is *not* the model: the loops, plans, routing, retries, budgets,
approvals, persistence and recovery that decide whether an agent is a demo or something you can put on call.
It gives you ready-made algorithms (tool loops, ReAct, plan-and-execute, routers, supervisors, generate-test-repair,
workflows, state machines, RAG, voting, Reflexion, debate, swarm handoffs, tree-of-thoughts) and the runtime that makes them durable, bounded and observable.

```python
--8<-- "examples/quickstart.py"
```

```text
success - 17 x 23 = 391
```

!!! info "That example runs offline"
    A scripted model stands in for the LLM so every example in these docs is executable and tested. The
    [LLM gateways guide](guides/llm-gateways.md) shows how to point the same code at a real model.

## Why AAA

<div class="grid cards" markdown>

-   **The log is the truth**

    Every run is an append-only stream of immutable events. State, plans, conversations and spend are *folds* of
    that log, so runs can be replayed, audited, resumed after a crash, and moved between processes.
    [:octicons-arrow-right-24: Events and the log](concepts/events-and-the-log.md)

-   **Your data, not ours**

    Algorithms see an `Item` envelope; you keep your own types (Pydantic, dataclasses, dicts, DataFrame rows,
    documents). A small codec teaches AAA to render, store and fingerprint them.
    [:octicons-arrow-right-24: Data model](concepts/data-model.md)

-   **Bounded by construction**

    Budgets (steps, tokens, cost), deadlines, cancellation, approval gates and tool timeouts are enforced by the
    engine, outside the model's control. Failures are facts on the log, not exceptions.
    [:octicons-arrow-right-24: Runs and the engine](concepts/runs-and-the-engine.md)

-   **Any LLM gateway**

    One OpenAI-compatible client with presets for LiteLLM, Portkey, Cloudflare, Vercel, Bifrost, Kong and MLflow,
    an in-process LiteLLM adapter, and a plug-in point for anything custom.
    [:octicons-arrow-right-24: LLM gateways](guides/llm-gateways.md)

-   **Durable and distributed**

    SQLite for one machine; Postgres or Redis plus a partitioned bus for many. Kill a worker mid-tool-call and
    another finishes the job.
    [:octicons-arrow-right-24: Running a cluster](guides/clusters.md)

-   **Safe side effects**

    Exclusive execution leases, numbered attempts as fencing tokens, at-most-once / at-least-once delivery, and
    idempotency keys for tools that send email or move money.
    [:octicons-arrow-right-24: Tool execution guard](guides/tool-guard.md)

</div>

## What is in 0.3

<div class="grid cards" markdown>

-   **More than text**

    Images, audio and files in messages, offloaded to a blob store; native Anthropic and Gemini clients.
    [:octicons-arrow-right-24: Multimodal](guides/multimodal.md) · [LLM gateways](guides/llm-gateways.md)

-   **Retrieval**

    Hybrid (similarity + keyword) search on pgvector, sqlite-vec and in memory; incremental document ingestion; Voyage reranking.
    [:octicons-arrow-right-24: Vector stores](guides/vector-stores.md) · [Ingestion](guides/ingestion.md)

-   **Tools that are safe to hand a model**

    An SSRF-hardened fetch tool, confined file access, web search, and a Docker code sandbox that needs human approval by default.
    [:octicons-arrow-right-24: Built-in tools](guides/builtin-tools.md)

-   **MCP both ways**

    Use any MCP server's tools, or serve yours to any MCP client with approval-gated tools hidden by default.
    [:octicons-arrow-right-24: MCP](guides/mcp.md)

-   **Evals that reach production**

    Scores delivered to Langfuse, Phoenix and LangSmith; datasets from real traces; red-team suites; simulated users.
    [:octicons-arrow-right-24: Evaluating agents](guides/evals.md)

-   **Run it for real**

    Backup and restore with a consistency check, measured throughput, a container image, an upgrade guide.
    [:octicons-arrow-right-24: Backup](operations/backup-restore.md) · [Upgrading](operations/upgrading.md)

</div>

!!! tip "Want an HTTP API?"
    The separate **[`aaa-server`](server/index.md)** package puts an engine behind a port: authentication, tenant isolation, rate limits and quotas, approvals over HTTP, resumable event streams, MCP and A2A endpoints.
    `pip install aaa-server` · [overview and docs](server/index.md) · [PyPI](https://pypi.org/project/aaa-server/)

## The algorithms

| Algorithm | Use it when |
|---|---|
| [`ToolLoop`](guides/algorithms/tool-loop-react.md) | The model should call your functions until it can answer. The default choice. |
| [`ReAct`](guides/algorithms/tool-loop-react.md) | The model/gateway has no native tool calling (text Thought/Action protocol). |
| [`PlanAndExecute`](guides/algorithms/plan-and-execute.md) | A task decomposes into steps with dependencies that can run in parallel. |
| [`Router`](guides/algorithms/router-supervisor.md) | Pick exactly one specialist: rules first, an LLM only for what rules do not decide. |
| [`Supervisor`](guides/algorithms/router-supervisor.md) | An LLM coordinates several workers, possibly in parallel. |
| [`GenerateTestRepair`](guides/algorithms/refine.md) | You can verify output deterministically (schema, tests, rules). |
| [`GeneratorCritic`](guides/algorithms/refine.md) | You can only verify with another model. |
| [`WorkflowGraph`](guides/algorithms/workflow-fsm.md) | The control flow is known; no LLM should decide it. |
| [`FSM`](guides/algorithms/workflow-fsm.md) | Long-lived, event-driven processes with explicit states (approvals, onboarding). |
| [`SelfConsistency`](guides/algorithms/ensembles-rag.md) | Answers are exactly comparable; majority vote beats one sample. |
| [`RAGAgent`](guides/algorithms/ensembles-rag.md) | Answers must be grounded in your documents, with citations. |
| [`Reflexion`](guides/algorithms/reasoning.md) | Retry from scratch, informed by written lessons from the failed attempts. |
| [`Debate`](guides/algorithms/reasoning.md) | Several answers that read each other's, then a judge or a vote. |
| [`Swarm`](guides/algorithms/reasoning.md) | Specialist agents hand a conversation to each other, each with only its own tools. |
| [`TreeOfThoughts`](guides/algorithms/reasoning.md) | Search over reasoning steps, scored and pruned, within a hard call budget. |

[Choosing an algorithm](guides/algorithms/index.md) walks through the decision, and
[Writing your own](guides/algorithms/custom.md) shows the 40-line recipe for a new one.

## Status

!!! warning "Alpha software: read the limits"
    AAA is version 0.x. The test suite is substantial (unit, store/bus contract tests across four backends,
    real multi-process tests that `SIGKILL` workers) but you should know what has and has not been proven:

    - **Verified live** against a real model server (a local Ollama running `gemma4`), including tool calls,
      streaming, structured output, plan-and-execute and repair loops.
    - **Not yet verified live** against any hosted gateway (Portkey, Cloudflare, Vercel, Bifrost, Kong, MLflow,
      LiteLLM proxy). Their URLs and headers follow vendor documentation and are covered by mocked-transport
      tests. A [smoke-test script](about/limitations.md#verify-a-hosted-gateway-yourself) is provided.
    - **Lightly load tested.** Correctness under failure is tested; throughput is [measured](operations/performance.md#measured-throughput)
      for one laptop (workers sharing its cores), not for a real multi-machine deployment.

    The full list is on [Status and limitations](about/limitations.md).

## Where to go next

1. [Install it](getting-started/installation.md) and run the [quickstart](getting-started/quickstart.md).
2. Read [the mental model](getting-started/mental-model.md) (ten minutes; everything else follows from it).
3. Pick an [algorithm](guides/algorithms/index.md), or go straight to [persistence](guides/persistence.md) and
   [clusters](guides/clusters.md) if you are deploying.
4. Need an API in front of it? See the [`aaa-server` overview](server/index.md).
5. Keep the [event catalog](reference/event-catalog.md) and [configuration reference](operations/configuration.md) open while you build.
6. See a real project built on it: [Built with AAA](about/built-with-aaa.md).

AAA is MIT licensed. Source: [github.com/innerkorehq/AutonomousAgentArchitecture](https://github.com/innerkorehq/AutonomousAgentArchitecture).
