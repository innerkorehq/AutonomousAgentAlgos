# LLM gateways

Algorithms never talk to a provider. They emit `aaa.llm.requested`, and the **`LLMHandler`** you attach to the engine
answers by calling an **`LLMClient`**. Swapping providers or gateways means changing one line.

```python
from aaa.llm import LLMHandler, build_client, gateways

client = build_client(gateways.portkey("gpt-4o-mini", provider="openai"))
LLMHandler(client).attach(engine)
```

## The `LLMClient` protocol

Anything with these three async methods is an LLM backend:

```python
class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
    async def embed(self, texts: list[str], model: str | None = None) -> EmbedResponse: ...
    async def aclose(self) -> None: ...
```

A client may also implement `stream(request) -> AsyncIterator[StreamChunk]`; the handler detects it and streams when a
request asks for it ([Streaming](streaming.md)).

Requests and responses are normalized, whatever the provider:

| Type | Fields |
|---|---|
| `LLMRequest` | `model` (None = client default), `messages`, `tools`, `response_schema` (JSON Schema, for structured output), `params` (temperature, max_tokens, stop, ...), `stream` |
| `Message` | `role` (`system`/`user`/`assistant`/`tool`), `content`, `tool_calls`, `tool_call_id`, `name` |
| `ToolCall` | `id`, `name`, `arguments` (parsed dict), `raw_arguments` (set only when the model emitted invalid JSON) |
| `LLMResponse` | `message`, `usage`, `cost`, `model`, `finish_reason` |

## Configuring a client

`LLMConfig` is declarative. **Secrets are references, never values**: `${ENV_VAR}` is resolved when the client is built,
and a missing variable is a loud `LLMConfigError`.

| Field | Default | |
|---|---|---|
| `provider` | `"openai_compat"` | `"openai_compat"`, `"litellm"`, or any name you [registered](#custom-backends). |
| `model` | required | Default model; a request can override it. |
| `base_url` | `""` | Required for `openai_compat`. |
| `api_key` | `None` | Sent as `Authorization: Bearer ...`. Use `"${OPENAI_API_KEY}"`. |
| `headers` | `{}` | Extra headers; values may use `${VAR}`. |
| `chat_path` | `"/chat/completions"` | Appended to `base_url`. Use `""` when `base_url` already is the full endpoint. |
| `embeddings_path` | `"/embeddings"` | |
| `timeout_s` | `60.0` | Per-request timeout. Raise it for slow reasoning models. |
| `params` | `{}` | Default request params (merged under each request's own). |
| `price_in_per_mtok`, `price_out_per_mtok` | `None` | Enables [cost](../concepts/runs-and-the-engine.md#budgets) accounting and `Budget(max_cost=...)`. |

```python
from aaa.llm import LLMConfig, build_client

client = build_client(
    LLMConfig(
        model="gemma4:latest",
        base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
        timeout_s=240,
    )
)
```

## Gateway presets

All of these gateways speak the OpenAI chat-completions dialect, so a preset is only a base URL plus the auth and
routing headers that gateway documents. Every field can be overridden by passing it as a keyword.

```python
--8<-- "examples/llm_gateways.py"
```

| Preset | Base URL | Auth and routing | Model naming |
|---|---|---|---|
| `gateways.litellm_proxy(model)` | `http://localhost:4000/v1` | `Bearer ${LITELLM_API_KEY}` (`api_key=None` for none) | whatever the proxy routes |
| `gateways.portkey(model, provider="openai")` | `https://api.portkey.ai/v1` | `x-portkey-api-key: ${PORTKEY_API_KEY}`, `x-portkey-provider`; `Bearer ${PROVIDER_API_KEY}` is the *upstream* key | provider's model name |
| `gateways.cloudflare(model, account_id, gateway_id="default")` | `https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/compat` | `cf-aig-authorization: Bearer ${CLOUDFLARE_API_TOKEN}` (`authenticated=False` to omit); `Bearer ${PROVIDER_API_KEY}` | `provider/model` |
| `gateways.vercel(model)` | `https://ai-gateway.vercel.sh/v1` | `Bearer ${AI_GATEWAY_API_KEY}` | `creator/model`, case-sensitive |
| `gateways.bifrost(model)` | `http://localhost:8080/v1` | optional `api_key=` | `provider/model` |
| `gateways.kong(model, route_url)` | *your route URL*, used verbatim (`chat_path=""`) | optional `api_key=` | the model your `ai-proxy` allows |
| `gateways.mlflow(endpoint)` | `http://localhost:5000/gateway/mlflow/v1` | optional `api_key=` | the gateway *endpoint name* is the model |

!!! warning "Verify your gateway before relying on a preset"
    Preset URLs and headers follow each vendor's documentation, but **only a local Ollama has been exercised live**.
    Two presets depend on the gateway version: Bifrost's path (`/v1` vs `/openai`) and MLflow's base path
    (`/gateway/mlflow/v1` in current docs, `/v1` in older 3.x). Override `base_url` if yours differs. Run the
    [smoke test](../about/limitations.md#verify-a-hosted-gateway-yourself) against your gateway before production; it
    takes a minute and a few thousand tokens.

### Model choice per call

Every algorithm accepts `model=`; it is passed in each `LLMRequest`. A request with `model=None` uses the client's
default. There is one `LLMHandler` per engine, so to route different models to different backends, give the handler a
small client that dispatches on `request.model` (see [Extending AAA](extending.md#route-models-to-different-backends)).

## What the OpenAI-compatible client does

- **Tool calls**: your `ToolSpec`s become `tools=[{"type":"function",...}]`; calls come back as parsed `ToolCall`s. Arguments that
  are not valid JSON are kept in `raw_arguments` and reported back to the model as an error instead of guessed at.
- **Structured output**: `response_schema` becomes `response_format={"type":"json_schema","json_schema":{...,"strict":true}}`.
  Strict mode is demanding on some providers (all properties required, `additionalProperties: false`); built-in schemas
  follow those rules, but verify yours against your provider.
- **Streaming**: server-sent events with `stream_options.include_usage`. Not every gateway returns usage in the stream.
- **Cost**: computed from the prices in your config, if set.

### Error mapping

HTTP and transport failures become typed errors so retry logic can be correct:

| Condition | Error | Retryable |
|---|---|---|
| 429 | `RateLimited` (carries `retry_after` from the `Retry-After` header) | yes |
| 408, 409, 425, any 5xx; timeouts; connection errors; non-JSON bodies | `TransientError` | yes |
| 401, 403 | `AuthError` | no |
| `content_filter` / `content_policy` in the body, or finish reason `content_filter` | `ContentFiltered` | no |
| 400/413 mentioning context length / too many tokens | `ContextOverflow` | no |
| anything else 4xx | `FatalError` | no |
| the breaker is open | `CircuitOpen` | no |

`LLMHandler` converts a raised `LLMError` into an `aaa.llm.failed` event (with `error.type`, `error.message`,
`error.retryable`). The built-in loops end the run as `failed` on it; write your own algorithm to react differently
(shrink the prompt on `ContextOverflow`, fall back to a cheaper model on `RateLimited`).

## Retries and circuit breaking

Opt-in wrappers, applied around any client:

```python
from aaa.llm import CircuitBreakerClient, RetryingClient, build_client

client = RetryingClient(
    CircuitBreakerClient(build_client(cfg), failure_threshold=5, cooldown_s=30),
    max_attempts=3,
    base_delay=0.5,
    max_delay=20,
)
```

- `RetryingClient` retries only *retryable* errors, with exponential backoff and full jitter, and honors `Retry-After`.
- `CircuitBreakerClient` opens after `failure_threshold` consecutive transient failures, fails fast with `CircuitOpen`
  for `cooldown_s`, then lets one probe through (half-open). Caller mistakes (auth, bad request) never trip it.
- Streaming passes straight through both wrappers (no retry once bytes may have been seen).

!!! tip "Do not stack retries"
    If your gateway already retries and fails over (Portkey, LiteLLM, Cloudflare), leave `RetryingClient` out or set
    `max_attempts=1`, or one outage becomes a retry storm.

## LiteLLM in-process

```python
build_client(
    LLMConfig(provider="litellm", model="anthropic/claude-sonnet-4-5")
)  # pip install "...[litellm]"
```

Uses the LiteLLM SDK instead of HTTP, maps its exceptions onto the same error types, and reports LiteLLM's computed
cost. Not exercised live in this repository's test suite.

## The `LLMHandler`

```python
LLMHandler(client, concurrency=8, emit_deltas=True).attach(engine)
```

- `concurrency` bounds simultaneous requests per worker.
- It rewrites each tool call's id to be unique per call (`call_<request id>_<n>`): providers may omit or repeat ids.
- It reconstructs full conversations for [delta requests](../operations/performance.md#long-conversations) from a small
  cache, falling back to the log after a restart.
- Failures are *events*; the handler never raises into the bus.

## Custom backends

```python
from aaa.llm import LLMConfig, register_provider


class MyClient:
    def __init__(self, config: LLMConfig) -> None: ...
    async def complete(self, request): ...
    async def embed(self, texts, model=None): ...
    async def aclose(self): ...


register_provider("mine", MyClient)
client = build_client(LLMConfig(provider="mine", model="x"))
```

Raise the typed errors from `aaa.llm.errors` so retries and algorithms behave correctly.

## Fakes for tests

`FakeLLMClient(script)` takes a list of responses (or exceptions to raise) or a function of the request, and records
the requests it saw. `ReplayLLMClient(events)` serves the answers recorded in a run's log. See
[Testing your agents](testing.md).
