# AutonomousAgentAlgos (AAA)

**An async, event-driven Python library of agent algorithms you can run in production, over data of any shape, with any LLM gateway.**

AAA is for the part of an agent system that is *not* the model: the loops, plans, routing, retries, budgets,
approvals, persistence and recovery that decide whether an agent is a demo or something you can put on call.
It gives you ready-made algorithms (tool loops, ReAct, plan-and-execute, routers, supervisors, generate-test-repair,
workflows, state machines, RAG, voting) and the runtime that makes them durable, bounded and observable.

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
    - **Not load tested.** Correctness under failure is tested; throughput numbers are only given for the
      single-process log-growth benchmark.

    The full list is on [Status and limitations](about/limitations.md).

## Where to go next

1. [Install it](getting-started/installation.md) and run the [quickstart](getting-started/quickstart.md).
2. Read [the mental model](getting-started/mental-model.md) (ten minutes; everything else follows from it).
3. Pick an [algorithm](guides/algorithms/index.md), or go straight to [persistence](guides/persistence.md) and
   [clusters](guides/clusters.md) if you are deploying.
4. Keep the [event catalog](reference/event-catalog.md) and [configuration reference](operations/configuration.md) open while you build.

AAA is MIT licensed. Source: [github.com/innerkorehq/AutonomousAgentAlgos](https://github.com/innerkorehq/AutonomousAgentAlgos).
