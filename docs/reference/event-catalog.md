# Event catalog

Every event type the library emits. All are ordinary [`Event`](../concepts/events-and-the-log.md) envelopes; the table gives
the `data` payload. `aaa.*` is reserved; your own events should use your own prefix.

!!! info "Reading the tables"
    "Emitted by" names the component. *Derived* events (created with `event.caused(...)`) get deterministic ids, and results use the
    fixed id `<request id>:result`.

## Run lifecycle

| Type | `data` | Emitted by | Notes |
|---|---|---|---|
| `aaa.run.started` | `algo`, `input`, `parent`, `run_tag`, `budget` | `Engine.start` | id `<run>:started`; first event of every run. |
| `aaa.run.completed` | `kind`, `value`, `reason` | algorithm (`finish`) or `Engine.complete` | id `<run>:completed`; **exactly one per run**; never claim-checked. |
| `aaa.run.erased` | `reason`, `events`, `blobs` | `engine.erase` / retention | id `<run>:erased`; appended after a run is scrubbed; written straight to the log, not delivered. See [Erasure](../guides/privacy.md#erasure). |
| `aaa.run.cancel` | `reason` | `RunHandle.cancel` | The engine turns it into `run.completed` (`cancelled`). |
| `aaa.budget.exceeded` | `reason` | `Engine` | Followed by `run.completed` (`budget_exceeded`). |
| `aaa.timer.fired` | `kind: "deadline"` | `Engine` | Followed by `run.completed` (`timeout`). |
| `aaa.child.started` | `child_run_id`, `tag` | `Engine.start(parent=...)` | On the **parent's** stream. |
| `aaa.child.completed` | `child_run_id`, `tag`, `outcome`, `ledger` | `Engine` | On the parent's stream; `ledger` is the child's spend, rolled into the parent's. |

## LLM

| Type | `data` | Notes |
|---|---|---|
| `aaa.llm.requested` | `request` (an `LLMRequest`), `tag`, and optionally `conv: {prev, delta}` | With `conv`, `request.messages` is empty and the messages are `delta`, continuing the conversation after request `prev`'s reply. |
| `aaa.llm.completed` | `tag`, `response` (an `LLMResponse`), `usage`, `cost` | id `<request>:result`. `usage`/`cost` are top-level so spend accounting never needs the blob. Tool-call ids are rewritten to `call_<request id>_<n>`. |
| `aaa.llm.failed` | `tag`, `error: {type, message, retryable}` | id `<request>:result`. |
| `aaa.llm.delta` | `text`, `tag` | Only when streaming. Many per call; not part of any outcome. |

## Tools

| Type | `data` | Notes |
|---|---|---|
| `aaa.tool.requested` | `call_id`, `name`, `arguments` | Emitted by algorithms. |
| `aaa.tool.started` | `call_id`, `name`, `attempt`, `worker`, `idempotency_key` | id `<request>:attempt:<n>`; recorded **before** execution. The fencing token. Absent if the guard is disabled. |
| `aaa.tool.completed` | `call_id`, `name`, `result` | id `<request>:result`. |
| `aaa.tool.failed` | `call_id`, `name`, `error: {type, message}` | id `<request>:result`. Error types: `InvalidArguments`, `UnknownTool`, `Timeout`, `ApprovalDenied`, `OutcomeUnknown`, `ExecutionAbandoned`, or the tool's exception class. |
| `aaa.tool.fenced` | `call_id`, `attempt`, `worker`, `reason` | A worker lost its lease mid-call and cancelled its tool. |
| `aaa.tool.stale_result` | `call_id`, `attempt`, `superseded_by`, `worker`, `outcome`, `data` | A superseded attempt finished; kept for audit, not applied. |
| `aaa.approval.requested` | `call_id`, `name`, `arguments` | `ToolHandler`, for `requires_approval` tools. |
| `aaa.approval.granted` | `call_id` | **You** (`run.signal`). |
| `aaa.approval.denied` | `call_id`, `reason?` | **You**. Becomes a `tool.failed` (`ApprovalDenied`). |

## Validation

| Type | `data` |
|---|---|
| `aaa.memory.requested` | `op` (`recall`/`remember`/`forget`), `tag`, `query`/`k`/`filters`/`as_of`, `entries`, `ids` | An algorithm (`ToolLoop(recall=True)`) or you. Scope comes from the event's tenant and metadata, never the payload. See [Long-term memory](../guides/memory.md). |
| `aaa.memory.started` | `op` | `MemoryHandler`, id `<request>:attempt`. Written before a non-idempotent write so a worker that dies mid-write is reported, not repeated. |
| `aaa.memory.completed` | `op`, `tag`, `items` / `ids`, `pending`, `usage`, `cost` | `MemoryHandler`, id `<request>:result`. `usage`/`cost` count towards the run's ledger. |
| `aaa.memory.failed` | `op`, `tag`, `error` | `MemoryHandler`. A failure here never fails the run. |
| `aaa.validate.requested` | `candidate`, `validator`, `context` |
| `aaa.validate.completed` | `verdict: {ok, feedback, score}` (id `<request>:result`) |

## Algorithm events

| Type | `data` | Emitted by |
|---|---|---|
| `aaa.plan.created`, `aaa.plan.revised` | `steps: [{id, description, depends_on}]` | `PlanAndExecute` |
| `aaa.task.started` | `step_id`, `child_run_id` | `PlanAndExecute` |
| `aaa.route.selected` | `route` | `Router` |
| `aaa.node.skipped` | `node_id` | `WorkflowGraph` |
| `aaa.fsm.transitioned` | `from`, `to`, `on` | `FSM` |
| `aaa.fsm.ignored` | `state`, `event` | `FSM` (an event with no valid transition) |

## Your own events

Any other type is yours. They flow through the same store and bus; subscribe with `engine.subscribe("acme.*", handler, group="acme.handler")`,
or let an `FSM`/custom algorithm react via `subscriptions()`. Use `run.signal(type, data)` to inject one into a run.

## Consumer groups (cluster)

| Group | Consumes |
|---|---|
| `aaa.algo.<name>` | the events an algorithm subscribes to, for runs of that algorithm |
| `aaa.llm` | `aaa.llm.requested` |
| `aaa.tool.request` / `aaa.tool.granted` / `aaa.tool.denied` | tool requests / approvals / denials |
| `aaa.validate` | `aaa.validate.requested` |
| `aaa.engine.cancel` | `aaa.run.cancel` |
