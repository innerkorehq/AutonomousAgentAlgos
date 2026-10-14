# Tools

A tool is a function the model may call. AAA generates its JSON schema from your type hints, validates the model's
arguments, runs it with a timeout in a bounded pool, and reports failures back to the model as data.

```python
from aaa.tools import ToolContext, ToolRegistry, tool


@tool
def multiply(a: int, b: int = 1) -> int:
    """Multiply two integers."""
    return a * b


@tool(timeout_s=10, pool="db")
async def lookup_customer(customer_id: str, ctx: ToolContext) -> dict:
    """Fetch a customer record."""
    return await db.fetch_one("select * from customers where id = $1", customer_id)


registry = ToolRegistry([multiply, lookup_customer])
```

## Schemas come from your types

- Parameter types, defaults and the docstring become the schema and description the model sees.
- Parameters accept anything Pydantic validates: `int`, `list[str]`, nested models, `Literal[...]`, dataclasses.
- Unknown arguments are rejected (`additionalProperties: false`), so the model cannot invent parameters.
- A parameter named `ctx: ToolContext` is injected and hidden from the schema.
- The return value is converted to JSON (`BaseModel` and dataclasses included) before it enters the log.
- Sync functions run in a worker thread; `async def` functions are awaited. Neither blocks the event loop.

```python
print(multiply.spec().model_dump())
# {'name': 'multiply', 'description': 'Multiply two integers.',
#  'parameters': {'additionalProperties': False, 'properties': {'a': {'type': 'integer'},
#                 'b': {'default': 1, 'type': 'integer'}}, 'required': ['a'], 'type': 'object'}}
```

## `@tool` options

| Option | Default | |
|---|---|---|
| `name`, `description` | function name, docstring | |
| `requires_approval` | `False` | Wait for a human [approval event](human-approval.md) before running. |
| `timeout_s` | `60` | Cancel and report `Timeout` if exceeded. |
| `pool` | `"default"` | Concurrency pool (below). |
| `delivery` | handler default | `"at_least_once"` or `"at_most_once"`: see the [guard](tool-guard.md). |
| `idempotency_key` | `<run>/<call id>` | `lambda args: "refund:" + args["order_id"]`; passed to the tool as `ctx.idempotency_key`. |
| `guard` | handler default | A dict of per-tool [`GuardConfig`](tool-guard.md#configuration) overrides. |

Invalid options fail when the tool is *defined*, not on the first call.

## What the model sees when things go wrong

Tool failures are **data fed back to the model** so it can recover:

| Cause | `error.type` | Message to the model |
|---|---|---|
| The model sent bad arguments | `InvalidArguments` | `invalid arguments for multiply: a: Input should be a valid integer ...` |
| The model called a tool that does not exist | `UnknownTool` | `no tool named 'x'; available: [...]` |
| The tool raised | the exception class name | `str(exception)` |
| The tool exceeded `timeout_s` | `Timeout` | `timed out after 10s` |
| A human denied it | `ApprovalDenied` | the reviewer's reason |
| A lost at-most-once attempt | `OutcomeUnknown` | explains it may have taken effect |
| An at-least-once call kept dying | `ExecutionAbandoned` | gave up after N attempts |

`ToolLoop` renders these as `Error (InvalidArguments): ...` in the tool message. Raise `ToolError("a clear message")`
from your tool for a clean, model-readable failure without a traceback class name.

## Parallel calls

If the model returns several tool calls in one turn, they run concurrently and the results are returned **in the order
of the calls**, so the next model request is deterministic.

## Concurrency pools

```python
ToolHandler(registry, pools={"default": 16, "delegate": 8, "db": 4})
```

- Each tool runs in a named pool; a pool is a semaphore bounding concurrent executions *per worker*.
- `delegate` is reserved for tools that wait on child runs (used by `Supervisor`). It is separate so delegating tools
  can never starve the pool their children need.
- A tool naming a pool that does not exist uses `default`.

## One handler per engine

`ToolHandler(registry).attach(engine)` registers the tool executor. Attaching a second handler to the same engine
**merges its tools into the first** instead of competing for requests, so independent algorithms (a supervisor and its
workers) share one namespace:

```python
ToolHandler(supervisor.tools).attach(engine)
ToolHandler(worker_registry).attach(engine)  # merged
```

In a cluster, every worker must register the same tools.

## Exposing a subset

`ToolLoop(registry, tool_names=["search", "multiply"])` shows the model only those tools while the registry can hold more.

## `ToolContext`

Declare `ctx: ToolContext` to receive:

| Field | |
|---|---|
| `engine` | The engine (e.g. to start a child run). |
| `run_id`, `call_id` | Which run and which model call. |
| `attempt`, `fence` | 1-based execution number of this call (rises if the call is re-run). |
| `idempotency_key` | Stable across attempts and workers. Pass it to downstream systems. |

## Testing a tool

A tool is plain Python plus validation; call it directly:

```python
assert await multiply.invoke({"a": 6, "b": 7}) == 42
await multiply.invoke({"a": "six"})  # raises ToolArgsError, exactly what the model would be told
```

## Retrieval as a tool

`search_tool(index, name="search_documents", k=5, max_chars=800)` exposes any [`Index`](../concepts/data-model.md#retrieval-index)
as a tool. [`RAGAgent`](algorithms/ensembles-rag.md) is a `ToolLoop` preconfigured with it.
