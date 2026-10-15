# Chat sessions and provider-side state

Two ideas that are easy to confuse, and that work well together:

- A **session** is a durable conversation *thread* in AAA: a stable id, many user turns, an agent answering each one, surviving restarts and
  process boundaries.
- **Provider-side state** means the *LLM provider* keeps the conversation for you (OpenAI's Responses API), so each request carries only
  what is new instead of the whole history.

```python
--8<-- "examples/chat_session.py"
```

```text
you: Hi, my name is Ann.
bot: Noted (1 things said so far).
you: I work on payments.
bot: Noted (2 things said so far).
you: What is my name?
bot: Your name is Ann.
reattached: 3 turns on record
messages sent to the provider per turn: [2, 2, 2]  (resending the history would be [2, 4, 6])
turns continued from provider state: [False, True, True]
erased 4 runs; provider state deleted: 3
```

## Sessions

```python
sessions = SessionManager(engine, ToolLoop(registry, stream=True), system="You are a helpful analyst.")

chat = await sessions.create(user="ann")                    # untenanted; or sessions.tenant("acme").create(user="ann")
reply = await chat.send("What were Q3 sales?")              # TurnResult: .text .ok .kind .reason .run_id
reply = await chat.send("And Q4?")                          # the agent remembers the first question

async with await chat.stream("Summarize both") as s:        # token by token (the inner algorithm needs stream=True)
    async for piece in s:
        print(piece, end="")
    final = await s.result()

chat = await sessions.get(chat.id)                          # reattach later, from any process
await chat.history()                                        # [Turn(n, user, assistant, status, run_id)]
await chat.close()
```

### What a session is

A session is a **long-lived run**. Its stream holds `aaa.session.message` (the user speaking) and `aaa.session.reply` (the agent answering). Each message
starts a **child run** of your inner algorithm with the conversation so far as its input. Because it is built from the engine's own parts, a session gets the
rest for free:

| You get | Because |
|---|---|
| **Crash safety across processes** | The reply is recorded by whichever worker handles the child's completion. If the process that sent the message dies, `await chat.wait(n)` from anywhere returns the answer. |
| **Tenancy** | `sessions.tenant("acme")` namespaces ids (`acme/<id>`); another tenant's session is a plain `KeyError`. |
| **A whole-conversation budget** | `SessionManager(..., session_budget=Budget(max_cost=5))` rolls every turn's spend up. The turn that crosses the line is still delivered; the next `send` raises `SessionClosed`. `turn_budget` caps a single turn. |
| **Erasure and retention** | A closed session is an ordinary finished run, so `engine.tenant("acme").erase_all()` and retention policies remove it and its turns. Open sessions are reported in `skipped_open`: close them first. |
| **Memory** | Recall and write-back ([Long-term memory](memory.md)) work per turn, scoped by the session's tenant and `user`. |

### Rules of the road

- **One turn at a time.** A `send` while a turn is running raises `SessionBusy`; two senders racing for the same turn cannot both win.
- **A failed turn leaves no trace** in what the model sees next (`TurnResult.ok` is false and the turn is marked `failed` in `history()`); the user can simply try again.
- **Build the manager in every process that may handle a turn.** Workers need it to register the algorithms; pure clients (a `ClientBus`) just call it.
- `user` is trusted application data (it scopes memory); it is never taken from the model.

### Long conversations

```python
SessionManager(engine, algo, system="...", context=ContextPolicy(max_input_tokens=128_000, summarizer=cheap_client))
```

When the history outgrows the budget, old turns are summarized **once**, written as `aaa.session.summary` events, and reused on later turns by content hash (see
[Context management](context.md)). Without a `context`, the whole history is sent each turn, so a very long chat will eventually hit the model's limit.

## Provider-side state

By default AAA sends the full conversation on every model call (its own log stays small by storing only deltas, but the provider still receives everything).
A provider that **keeps the conversation** lets you send only what is new.

```python
from aaa.llm import LLMConfig, build_client

client = build_client(LLMConfig(provider="openai_responses", model="gpt-4o", base_url="https://api.openai.com/v1", api_key="${OPENAI_API_KEY}"))
LLMHandler(client).attach(engine)          # nothing else changes: ToolLoop, sessions and the rest use it automatically
```

What happens with such a client (`server_state = True`):

1. Every reply carries the provider's handle (`state_id`), recorded on the `aaa.llm.completed` event.
2. The next request in the same run, or the first request of the next session turn, tells the provider to continue from that handle and sends **only the new
   messages** (plus the system prompt, which the API does not carry over). Tool calls and results are matched by the provider's own call ids.
3. It falls back to a full request, transparently, when the provider has lost the state (expired or deleted), when the handle is unknown (for instance a different
   provider), or when the [context policy](context.md) had to rewrite the conversation: a conversation that was compacted is never continued from the provider's older copy.
4. `aaa.llm.completed` records `state`: `fresh`, `continued` or `fallback`, so the saving (or a surprise) is visible in the log.

### Trade-offs, honestly

- **The provider now holds your data.** With `store=True` (the default for this client) OpenAI retains responses under its own policy. `engine.erase(...)` and `erase_all()`
  delete the state ids they know about (`DELETE /responses/{id}`) and report the count (`ErasureReport.provider_state`), but only for an `LLMHandler` attached to the engine.
  If that is unacceptable for your data, use `store=False`: the client then behaves like any stateless one.
- **Cost is mostly latency and bandwidth, not always tokens.** Providers usually still bill the whole conversation as input tokens (at the cached rate when the prefix is cached).
  `Usage.cached_tokens` and `LLMConfig.price_cached_in_per_mtok` make that visible and priced; continuation chiefly saves what you send and re-process.
- **Other providers.** The mechanism is a small protocol (`ServerStateClient`: `server_state`, `LLMResponse.state_id`, `LLMRequest.continue_from`, `forget_state`), so another
  provider's server-held conversations (Gemini, Mistral) can plug in. Only OpenAI's Responses API is implemented. Gateways differ in how much of `/responses` they support.
- Prompt caching of an unchanged prefix (Anthropic `cache_control`, OpenAI automatic caching) is a different, complementary feature: it is reported (`cached_tokens`) but not controlled by AAA.

## What is and is not verified

Sessions are tested on memory, Postgres and Redis (a client that only submits and a worker that answers), through restarts, races, failures, budgets, tenancy and erasure.
The Responses client is tested over a mock transport and against the **official `openai` SDK's own types** (every request field and input item it builds is one the SDK
defines; CI runs this). It has **not** been run against the live OpenAI API; treat the first run against your account as part of adopting it.
