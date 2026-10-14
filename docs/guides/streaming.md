# Streaming

Ask for streaming and the LLM handler publishes each piece of text as an `aaa.llm.delta` event while the model writes it.

```python
--8<-- "examples/streaming.py"
```

```text
Streaming works token by token.
outcome: success
```

## Turning it on

| Where | How |
|---|---|
| `ToolLoop`, `ReAct` | `ToolLoop(registry, stream=True)` |
| Your own algorithm | `LLMRequest(messages=..., stream=True)` |
| A client | it must implement `stream()`; the OpenAI-compatible client and `FakeLLMClient` do |

Clients without `stream()` still work: the handler falls back to a single completion and emits no deltas.

## Consuming deltas

Deltas are ordinary events in the run's stream, so any consumer works:

```python
async for event in run.events():  # replays from the start, then follows live
    if event.type == "aaa.llm.delta":
        websocket.send(event.data["text"])
```

`event.data` is `{"text": "...", "tag": "main"}`. The `tag` tells you which LLM call of a multi-call algorithm produced it.
Because deltas are in the log, a late-joining or reconnecting client replays everything it missed, and a finished run
can be re-streamed.

## Details

- **The final answer is still one `aaa.llm.completed` event** holding the whole message, tool calls and usage. Deltas are
  a live view, not the source of truth.
- **Tool calls stream too**, but their JSON arguments are assembled internally; they are not emitted as deltas.
- **Usage in streams** depends on the gateway honoring `stream_options.include_usage`. If it does not, tokens for that
  call are not counted toward `Budget(max_tokens=...)`.
- **No retry once streaming started.** `RetryingClient` and `CircuitBreakerClient` pass streams through, since the
  caller may already have shown text.
- **Volume.** Each delta is an event. For very chatty streams or very many concurrent runs, either set
  `LLMHandler(client, emit_deltas=False)` (the call still streams from the provider but nothing is published) or
  keep streaming for interactive runs only.
- **Mid-stream failure** becomes `aaa.llm.failed` like any other LLM error.
