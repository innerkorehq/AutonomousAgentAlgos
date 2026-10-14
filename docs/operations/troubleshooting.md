# Troubleshooting

## First: read the run's log

Almost every question is answered by the event list. Print it:

```python
for e in await engine.store.read(run_id):
    print(e.seq, e.type, e.time.isoformat(), str(e.data)[:100])
```

Look at the **last** event and ask what should have happened next.

| The log ends with... | Meaning |
|---|---|
| `aaa.run.started` | No algorithm picked it up: wrong/unregistered `name`, or (cluster) no worker running that algorithm. |
| `aaa.llm.requested` | No `LLMHandler` attached, the LLM call is still running, or the worker died (a cluster worker will redeliver). |
| `aaa.llm.failed` | The error is in `data["error"]` (`type`, `message`, `retryable`). The built-in loops end the run `failed`. |
| `aaa.tool.requested` | No `ToolHandler` attached on any worker. |
| `aaa.approval.requested` | Waiting for a human: signal `aaa.approval.granted` or `denied` with the `call_id`. |
| `aaa.tool.started` | A tool is running (or its worker died; see the guard). |
| `aaa.run.completed` | Done; read `data["kind"]` and `data["reason"]`. |

## Symptoms

### A run never finishes

1. Does the log end as above? Fix the missing piece.
2. Did you set a `Budget(deadline_s=...)`? Without one nothing bounds a run that is waiting on something that will never arrive.
3. Cluster: is a worker running that algorithm? Is `sweep_interval_s` set?
4. A brief **store outage** during a run can strand it ([known limitation](../about/limitations.md#store-outages-during-a-run)); the deadline is the backstop.

### `LLMConfigError: environment variable X is not set`

A `${X}` reference in `LLMConfig` (or a preset's default) has no value. Set the variable, pass your own `api_key=`/`headers=`, or
for an unauthenticated gateway pass `api_key=None` where the preset allows it.

### `LLMConfigError: unknown provider 'x'; known: [...]`

`provider` must be `openai_compat`, `litellm` (needs the extra), or a name you registered with `register_provider`.

### The run ends `failed` with `AuthError: HTTP 401`

The gateway rejected your key. Check the variable and, for presets with two keys (Portkey, Cloudflare), that the *gateway* key and the *provider* key are both set.

### The run ends `failed` with `FatalError: HTTP 404`

Usually a wrong model name or base URL. Compare with the gateway's docs (Bifrost and MLflow base paths vary by version).

### `ContextOverflow`

The conversation exceeds the model's window. Trim, summarize, or move to child runs. The built-in loops do not truncate for you.

### The model keeps calling a tool that fails

It sees `Error (InvalidArguments): ...` as the tool result. Improve the tool's docstring and parameter names, add examples to the description,
lower `max_turns`, and look at `ToolLoop(tool_names=[...])` to hide distractors.

### `UnknownTool` for a tool that exists

Cluster: a worker does not have that tool registered. Every worker must register identical tools. Single process: you attached a
`ToolHandler` with a different registry than the algorithm uses (attach all registries; they merge).

### `ValueError: consumer group 'aaa.algo.x' is already subscribed in this process`

Two algorithm instances share a `name` in one process with a distributed bus. Give each a unique `name=`.

### `ValueError: ambiguous algorithm name 'x'` (from `recover`)

You passed two different instances with the same name. Use unique names.

### `ValueError: this database was created with N partitions, not M`

The partition count is fixed per database (or Redis prefix). Use the original value.

### A tool ran twice

By design under `at_least_once` after a crash or stall. Make it idempotent with `ctx.idempotency_key`, or declare `delivery="at_most_once"`.
See [the guard](../guides/tool-guard.md). If it ran twice with *no* crash, check you did not start the same work under two different `run_id`s.

### `OutcomeUnknown` / `ExecutionAbandoned`

`OutcomeUnknown`: an `at_most_once` call's attempt was lost; verify the real-world state, then decide. `ExecutionAbandoned`: an `at_least_once` call
died on every one of `max_attempts` tries, so look for a crash in the tool.

### `ConcurrencyError`

You passed `expected_seq` to a store append and the stream moved. Reload and retry. The engine itself never uses it on shared streams.

### `BlobIntegrityError` / `FileNotFoundError` on hydrate

A blob is missing (not on shared storage for this worker, or deleted) or corrupt. Blobs must be reachable from every worker.

### Snapshots seem unused

They are created after `every` events (default 64) and ignored if `version` changed. Short runs never reach the threshold. This is normal.

### Cluster: events pile up / lag grows

Check workers are alive, partitions are owned (`bus.owned_partitions`), and that the lagging group's handler is not slow or failing.
Check you are trimming the outbox so it is not just an old backlog.

### Postgres cluster: a hung worker is not replaced

Postgres leases are session-bound. A frozen-but-connected worker holds them. Configure TCP keepalives / `idle_session_timeout`, or use Redis.

## Getting help

Open an issue with: the AAA version, store/bus, a minimal reproduction, and the event list of the failing run (redact sensitive data).
