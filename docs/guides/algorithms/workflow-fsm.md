# WorkflowGraph and FSM

Not every process needs a model to decide what happens next. These two put **you** in control of the flow while still
getting durability, parallelism, approval gates, budgets and an audit log.

```python
--8<-- "examples/workflow_fsm.py"
```

```text
workflow: 22
fsm path: draft -> review -> draft -> review -> done
```

## WorkflowGraph: a deterministic DAG

```python
class WorkflowGraph:
    def __init__(self, nodes, *, output=None): ...


Node(id, tool, after=[], args=None, when=None, on_error="fail")
```

Each node is **one tool call**, so nodes inherit everything tools have: schemas, timeouts, approval gates, concurrency
pools, the [execution guard](../tool-guard.md).

| `Node` field | |
|---|---|
| `id` | Unique name. |
| `tool` | Name of a registered tool. |
| `after` | Ids that must finish first. Nodes with no pending dependencies run **in parallel**. |
| `args` | A dict, or `fn(results, input) -> dict` computing arguments from upstream results and the run input. |
| `when` | `fn(results, input) -> bool`. If false the node is **skipped** (its result is `None`; downstream nodes still proceed). |
| `on_error` | `"fail"` (default): a failed node fails the run. `"continue"`: its result becomes `None`. |

`output` is a node id, a function of all results, or `None` (return every result as a dict). Construction raises
`ValueError` for duplicate ids, unknown dependencies and **cycles**, so a bad graph never starts.

Use it for: ETL-style pipelines, fan-out/fan-in over APIs, any process whose steps you can name in advance.

## FSM: states, events, loops

```python
FSM(initial, transitions, terminal, on_enter=None)
```

An explicit state machine driven by **events**: from your code, a webhook, a human, another run.

- `transitions[state][event_type]` is the next state, **or** `fn(event, input) -> state | None` (a guard; `None` means "not allowed now").
- An event with no valid transition in the current state is recorded as `aaa.fsm.ignored` and changes nothing.
- Entering a `terminal` state completes the run with `{"state", "path"}`.
- `on_enter[state](event, input) -> list[Event]` may emit events when a state is entered (request a tool, call an LLM,
  notify someone).
- Unlike a `WorkflowGraph`, **cycles are fine** (the example loops `review -> draft -> review`).
- The current state is a fold of `aaa.fsm.transitioned` events, so a machine waiting for a human can sleep for days and
  survive restarts.

Drive it with `run.signal(event_type, data)`:

```python
run = await engine.start(review, {"approvers": ["alice"]})
await run.signal("user.submit")
await run.signal("user.approve", {"by": "alice"})
```

!!! note "An FSM only ends when it reaches a terminal state"
    There is no implicit timeout. Give it a `Budget(deadline_s=...)` if an abandoned process should expire.

## WorkflowGraph or FSM?

| | WorkflowGraph | FSM |
|---|---|---|
| Shape | acyclic dependencies | any graph, including loops |
| Advances by | tool results | events |
| Parallelism | automatic | one transition at a time |
| Waits for outside input | only via approval-gated tools | natively |
| Typical use | pipelines | approvals, onboarding, order lifecycles |

They compose: an FSM state's `on_enter` can start a `WorkflowGraph` as a child run.
