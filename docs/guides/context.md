# Context management

A long run grows its conversation every turn: each tool result and each reply is appended and sent again. Sooner or later it no longer fits the
model's context window, or it fits but costs too much to resend. A **context policy** fits the conversation to a budget just before each
model call, so a run can go on for hundreds of steps.

```python
--8<-- "examples/context_management.py"
```

```text
without a policy the largest request was 13668 tokens (window 4000)
with a policy the largest request was 3363 tokens
the model was called 26 times and finished: Done: I looked up all 25 topics.
compaction was applied on 19 calls, using: ['clear_tool_results', 'stored_summary', 'summarize']
summaries written to the log: 4
the log still holds every message: True
```

```python
from aaa.context import ContextPolicy

policy = ContextPolicy(
    max_input_tokens=128_000,        # your model's real context window
    reserve_output=4_096,            # kept free for the reply
    summarizer=cheap_llm_client,     # optional: summarize old turns instead of dropping them
    summarizer_model="small-model",
)
LLMHandler(client, context=policy).attach(engine)
```

## How it works

- **It is a view, not an edit.** The handler keeps the *logical* conversation (what the algorithm and the event log hold) and gives the provider a
  fitted copy. Requests in the log are still small deltas, `conv.prev` still points at the previous request, and nothing is rewritten, so
  [replay and resume](persistence.md) behave exactly as before. A reply's `aaa.llm.completed` event carries a `context` report
  (`tokens_before`, `tokens_after`, `applied`) so you can see what was done and when.
- **It acts only when needed, and goes below the line.** Compaction starts when the estimate passes `trigger` (default 85 percent of the input budget)
  and stops under `target` (60 percent). That hysteresis means it runs now and then, not on every turn.
- **Estimates are calibrated.** `HeuristicCounter` estimates tokens from characters. `CalibratedCounter` (the default wrapper) compares each estimate
  with the provider's own `usage.prompt_tokens` and learns the correction per model. Exact counts are available for OpenAI-family models with
  `TiktokenCounter`. If the provider still reports an overflow, the handler retries once assuming the conversation is twice as big as estimated.

## Strategies

They run in order until the conversation fits, cheapest and least lossy first. Every one preserves three invariants: **tool calls stay with their results**
(providers reject orphans), **the head is untouched** (the system prompt, recalled memories and the original task), and **the most recent turns
stay verbatim** (`keep_recent`, default 3 turns).

| Strategy | What it does | Loses |
|---|---|---|
| `TruncateToolResults(max_chars=8000)` | Cuts any single huge tool result down to its start and end. | The middle of very large outputs. |
| `ClearToolResults(keep_last=4)` | Replaces old tool results with a placeholder; the calls stay, so the model remembers what it did. | Old tool output. |
| `Summarize(client, model)` | Replaces the oldest turns with a model-written, incremental summary, kept as `[head] [summary] [recent turns]`. Added when you pass `summarizer=`. | Detail, but keeps the gist. |
| `DropOldest()` | Last resort: deletes the oldest turns and leaves a one-line marker. | Everything it drops. |

Pass your own list with `ContextPolicy(..., strategies=[...])`, or write one: a strategy is an object with `name` and `async apply(work)`.

### Summaries are events

A summary costs a model call, so it is made once and kept: the handler writes it as an `aaa.context.summary` event holding the text, how many messages it covers
and a **hash of exactly that text**. A restart, a failover to another worker, or a later request finds it in the log and reuses it, but only if the conversation's
beginning still hashes the same (so a summary can never be applied to the wrong conversation). The next summary folds the previous one in, so it stays a fixed size.
What summarizing cost counts towards the run's budget ledger.

!!! note "Mind provider prompt caching"
    Many providers cache a conversation's unchanged *prefix*. Summaries are friendly to that: the head and the summary stay identical from turn to turn until the
    next compaction. `ClearToolResults` and `DropOldest` move as the conversation grows and so shift the prefix each turn; prefer a summarizer for long cached runs.

## Choosing numbers

- **`max_input_tokens` is yours to supply.** AAA does not carry a table of model windows (it would go stale). Too large means overflow errors (retried once, then failed); too small wastes context.
- Leave `reserve_output` at least as large as the replies you ask for (`max_tokens`).
- Use a **small, cheap model** for `summarizer_model`; the quality bar is "keeps names, numbers and decisions".
- `keep_recent` guards the turns the model is actively working on. Raise it if the agent loses its place.

## Limits

- The estimate is an estimate. Calibration and the overflow retry cover most of the gap, not all of it.
- Summaries are lossy by nature. The facts a run must never lose belong in the task, the system prompt, or [long-term memory](memory.md), not only in the transcript.
- The policy is applied by `LLMHandler`; a custom client that bypasses it is not managed. Images and other non-text content are not counted.
