# Outcomes and errors

## Run outcomes

Every run ends with one `Outcome(kind, value, reason)` (`aaa.core`). `outcome.ok` is `True` only for `success`.

| `OutcomeKind` | Produced when | `value` | `reason` |
|---|---|---|---|
| `success` | The algorithm finished. | the result (shape depends on the algorithm) | none |
| `no_solution` | A bounded loop ended without an answer (`max_turns`, `max_iterations`). | best attempt, if any | why |
| `timeout` | `Budget.deadline_s` elapsed. | none | `deadline Ns exceeded` |
| `budget_exceeded` | Steps, tokens or cost exceeded their cap. | none | e.g. `steps 21 > 20` |
| `invalid_input` | Input could not be handled (router chose an unknown route). | none | why |
| `cancelled` | `RunHandle.cancel`, or the parent run ended. | none | the reason given |
| `blocked` | A guardrail blocked the user input or the model's reply (`GuardrailBlocked`). See [Guardrails](../guides/guardrails.md). | none | `GuardrailBlocked: blocked by policy at input: <categories>` |
| `failed` | LLM failure the algorithm cannot recover from, an algorithm exception, an invalid plan, a failing workflow node. | none | `ErrorType: message` |

## LLM errors (`aaa.llm.errors`)

All derive from `LLMError` (which derives from `AAAError`) and carry a `retryable` flag.

| Error | `retryable` | Meaning |
|---|---|---|
| `TransientError` | yes | Timeouts, connection errors, 5xx, malformed gateway replies. |
| `RateLimited` | yes | 429; has `retry_after` seconds when the gateway sent `Retry-After`. |
| `GuardrailBlocked` | no | A guardrail blocked the request or reply; algorithms end the run as `blocked`. |
| `ContextOverflow` | no | Prompt exceeds the model's context window. |
| `ContentFiltered` | no | The provider blocked the request or response. |
| `AuthError` | no | 401/403. |
| `FatalError` | no | Any other client error (bad model name, 404, malformed request). |
| `CircuitOpen` | no | `CircuitBreakerClient` is rejecting calls. |
| `LLMConfigError` | no | Bad or incomplete configuration (missing `${ENV}`, unknown provider). |

They appear on the log as `aaa.llm.failed` with `error.type` set to the class name.

## Tool errors

On the log as `aaa.tool.failed` (`error.type`, `error.message`). Types: `InvalidArguments`, `UnknownTool`, `Timeout`, `ApprovalDenied`,
`OutcomeUnknown`, `ExecutionAbandoned`, or the class name of an exception your tool raised. In Python: `ToolError` (raise it for a clean model-readable
failure) and `ToolArgsError(ToolError)` (raised by `Tool.invoke` for bad arguments).

## Library exceptions

| Exception | Raised by |
|---|---|
| `aaa.core.AAAError` | Base class of all library errors. |
| `aaa.core.ConcurrencyError` | `EventStore.append(expected_seq=...)` mismatch. |
| `aaa.core.SchemaError` | Unknown/invalid/un-migratable event payload schema. |
| `aaa.core.RunNotFinished` | `engine.erase` on a run that is still open (pass `force=True`). |
| `aaa.core.TenantMismatch` | Chained cause of the `KeyError` a `TenantScope` raises for another tenant's run (reported as a missing run). |
| `aaa.data.MissingCapability` | `preflight()` when a required capability is absent. |
| `aaa.data.BlobIntegrityError` | A blob's content hash does not match its reference. |
| `ValueError` | Invalid configuration: bad `GuardConfig` combos, duplicate algorithm names, wrong partition count, cyclic `WorkflowGraph`. |

## What is *not* an exception

Failures inside a run (an LLM error, a crashing tool, a lost attempt) are **events**, not exceptions, and reach algorithms as data.
The exceptions above are for programming and configuration errors at the boundary of your code.
