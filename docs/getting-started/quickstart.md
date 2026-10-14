# Quickstart

This page builds an agent in four steps. Every snippet is a real file from the repository's
[`examples/`](https://github.com/innerkorehq/AutonomousAgentAlgos/tree/master/examples) folder and is executed by
the test suite.

## 1. A tool and a model

```python
--8<-- "examples/quickstart.py"
```

Run it:

```bash
python examples/quickstart.py
```

```text
success - 17 x 23 = 391
```

What happened, line by line:

1. `@tool` turned a plain function into a tool. The JSON schema the model sees is generated from the type hints and
   the docstring becomes the description.
2. `ToolRegistry` holds the tools an agent may call.
3. `Engine()` is the runtime. It owns an event log, a message bus and the bookkeeping for budgets and deadlines.
4. `LLMHandler(...).attach(engine)` registers *who answers LLM requests*. `ToolHandler(...).attach(engine)`
   registers *who runs tool calls*. Algorithms never call a model or a tool directly; they emit request events and
   react to the results (see [the mental model](mental-model.md)).
5. `engine.run(algorithm, input, budget=...)` starts a run and waits for its outcome.

!!! tip "The scripted model"
    `FakeLLMClient(scripted_model)` calls your function with each `LLMRequest` and expects an `LLMResponse`
    back. `text(...)` and `tool_call(...)` are helpers that build those. It makes examples and tests deterministic.
    Replace it with a real client to go live:

    ```python
    from aaa.llm import build_client, gateways

    LLMHandler(build_client(gateways.portkey("gpt-4o-mini"))).attach(engine)
    ```

## 2. Look at what actually happened

A run is a list of events. After the run, read them:

```python
--8<-- "examples/inspect_log.py"
```

```text
0 aaa.run.started
1 aaa.llm.requested
2 aaa.llm.completed
3 aaa.tool.requested
4 aaa.tool.started
5 aaa.tool.completed
6 aaa.llm.requested
7 aaa.llm.completed
8 aaa.run.completed
outcome: success
```

This list *is* the run. The [event catalog](../reference/event-catalog.md) documents every type.

## 3. Bound it

`Budget(max_steps=20, deadline_s=30)` above is enforced by the engine, not by the model's good behavior. Exceeding
a limit ends the run with a distinct outcome (`budget_exceeded`, `timeout`), never an exception.

```python
Budget(max_steps=20, max_tokens=50_000, max_cost=0.50, deadline_s=60)
```

## 4. Make it durable

Pass a durable store and the same code survives restarts:

```python
from aaa.events import SQLiteEventStore

async with Engine(store=SQLiteEventStore("agent.db")) as engine:
    ...
```

See [Persistence and recovery](../guides/persistence.md) for resuming after a crash.

## Next steps

<div class="grid cards" markdown>

-   [**The mental model**](mental-model.md) - the five ideas everything else is built from.
-   [**Choosing an algorithm**](../guides/algorithms/index.md) - from `ToolLoop` to `PlanAndExecute`.
-   [**LLM gateways**](../guides/llm-gateways.md) - connect a real model.
-   [**Tools**](../guides/tools.md) - schemas, pools, timeouts, delegation.

</div>
