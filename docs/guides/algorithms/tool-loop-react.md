# ToolLoop and ReAct

Both implement the classic agent loop: the model decides, tools act, results go back, repeat until the model answers.

```python
class ToolLoop:
    def __init__(
        self,
        registry,
        *,
        system="",
        model=None,
        max_turns=10,
        params=None,
        tool_names=None,
        name=None,
        stream=False,
    ): ...
```

## ToolLoop

`ToolLoop` uses the provider's **native function calling**. This is the default choice.

```python
--8<-- "examples/quickstart.py"
```

**Loop.** Each turn sends the conversation and tool specs to the model.

- If the reply contains tool calls: run them (concurrently), append their results, go again.
- If the reply is plain text: that is the answer; the run succeeds with `{"text": ..., "turns": n}`.
- If `max_turns` replies have asked for tools and there is still no answer: the run ends `no_solution`.
- If the LLM call fails (`aaa.llm.failed`): the run ends `failed` with the error type and message.

**Bad tool calls are conversation, not crashes.** Malformed JSON arguments, a missing tool, a failing tool, a timeout
or a denied approval all return to the model as an error message in the tool result, so it can correct itself. The
number of such round trips is bounded by `max_turns` and your `Budget`.

**Starting messages.** `{"task": "..."}` produces `[system?, user]`. Pass your own transcript instead:

```python
await engine.run(
    agent,
    {
        "messages": [
            {"role": "system", "content": "You are a careful analyst."},
            {"role": "user", "content": "Summarize ticket 481."},
        ]
    },
)
```

??? note "Parameters"
    | Parameter | Default | |
    |---|---|---|
    | `tools` | required | The `ToolRegistry`; its specs are sent with every request. |
    | `system` | `""` | System prompt (ignored if you pass `messages`). |
    | `model` | `None` | Model override for this algorithm. |
    | `max_turns` | `10` | Maximum model replies that request tools before giving up. |
    | `params` | `{}` | Passed to every request (`temperature`, `max_tokens`, `stop`, ...). |
    | `tool_names` | all | Expose only these tools. |
    | `name` | class default | Rename the instance (required to run two `ToolLoop`s side by side in a cluster). |
    | `stream` | `False` | Emit [`aaa.llm.delta`](../streaming.md) events. |

!!! tip "Long conversations stay cheap"
    Each continuing request stores only the new messages and a pointer to the previous request, and the loop's
    state is a handful of counters. A 400-turn run is ~1.3 MB of log, and cost per turn is constant. See
    [Performance](../../operations/performance.md#long-conversations).

## ReAct

`ReAct` is the text-protocol variant for models or gateways **without native tool calling**. It instructs the model to
write

```text
Thought: I need the population of Tokyo
Action: population
Action Input: {"city": "Tokyo"}
```

then *stops generation* (`stop=["\nObservation:"]`), runs the tool, and feeds back `Observation: ...`. It ends when the
model writes `Final Answer: ...`.

- The system prompt, including tool names, descriptions and argument schemas, is generated for you (override with `system=`).
- If both an action and a final answer appear, whichever comes first wins.
- If the reply has **no protocol markers at all**, it is treated as the final answer (models often just answer).
- `Action Input` that is not a JSON object is returned as an `Observation: Error ...` so the model can retry. It does not
  call a tool.
- One tool call per turn (the protocol has no parallel form).

```python
ReAct(registry, max_turns=8, params={"temperature": 0})
```

Prefer `ToolLoop` whenever native tool calling is available: it is more reliable, supports parallel calls and does not
depend on the model following a text format.

## Tuning checklist

- Set `params={"temperature": 0}` for repeatable behavior.
- Give tools clear docstrings. The model chooses tools from the descriptions.
- Keep `max_turns` small (5-10) and let the `Budget` be the real backstop.
- If the model keeps calling tools it does not need, shorten the registry with `tool_names`.
