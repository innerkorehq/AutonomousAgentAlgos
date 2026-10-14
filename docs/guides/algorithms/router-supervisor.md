# Router and Supervisor

Two ways to use specialists: pick **one** (`Router`), or let a model coordinate **several** (`Supervisor`).

```python
--8<-- "examples/router_supervisor.py"
```

```text
Refund issued.
Report ready:
Findings: AAA is event-driven.
Draft: AAA logs everything as events.
```

## Router

```python
class Router:
    def __init__(self, routes, *, classifier=None, default=None, model=None): ...
```

Chooses exactly one route and returns **that specialist's outcome unchanged**.

1. If a `classifier(input) -> route name | None` is supplied, it runs first. Rules are cheap, instant and testable.
2. Otherwise (or if it returns `None`) one LLM call picks a route with a JSON schema whose `route` is an **enum of your
   route names**, using each `Route(algo, description)`'s description.
3. A choice that is not a known route falls back to `default`; with no default the run ends `invalid_input`.

`routes` values are an algorithm or a `Route(algo, "when to use me")`. The selected route is recorded as `aaa.route.selected`.

!!! tip "Keep the model out of easy decisions"
    A router that asks an LLM "is this about billing?" for every request spends money to rediscover what a keyword
    check knows. Put the obvious cases in `classifier` and let the model decide only the ambiguous remainder.

## Supervisor

```python
class Supervisor:
    def __init__(
        self, workers, *, extra_tools=None, **tool_loop_kwargs
    ): ...  # workers: {name: (algorithm, "what it is good at")}
```

A `Supervisor` is a `ToolLoop` whose tools are `delegate_<worker>(task)`. When the model calls one, the worker runs as
a **child run**; its answer comes back as the tool result.

- Parallel delegation works: the model can call several `delegate_*` tools in one turn.
- Delegations run in the `delegate` [pool](../tools.md#concurrency-pools) (default 8 at a time per worker process).
- A worker that does not succeed makes the delegate tool fail, `worker researcher ended failed: <reason>`, which the
  supervisor model reads and can handle.
- The workers' spend rolls up into the supervisor's budget.
- Add ordinary tools alongside delegates with `extra_tools=[...]`.

Wire it up by attaching the supervisor's tools (and each worker's tools, which merge into the same executor):

```python
ToolHandler(supervisor.tools).attach(engine)
ToolHandler(worker_registry).attach(engine)
```

??? note "Why delegation does not deadlock"
    A delegating tool waits for a child run, and that child needs tool executions of its own. If both used one
    bounded pool, enough waiting delegates would starve their own children. Tool calls therefore run in background
    tasks with a *separate* `delegate` pool, so waiting never blocks the capacity workers need.

## Router vs. Supervisor

| | Router | Supervisor |
|---|---|---|
| Specialists used | exactly one | zero to many, in parallel |
| Who decides | rules, then optionally one LLM call | an LLM, every turn |
| Cost | 0-1 LLM calls | a full tool loop |
| Use when | requests fall into categories | a task needs several kinds of work combined |
