# Configuration reference

Every tunable in one place, with defaults. Names link to the guide that explains them.

## Engine and runs

| Setting | Where | Default | |
|---|---|---|---|
| `store` | `Engine(...)` | `InMemoryEventStore()` | [Persistence](../guides/persistence.md) |
| `bus` | `Engine(...)` | `InProcessBus()` | `DistributedBus`, `ClientBus` for [clusters](../guides/clusters.md) |
| `schemas` | `Engine(...)` | empty `SchemaRegistry` | versioned payload validation |
| `default_budget` | `Engine(...)` | unlimited | applies when a run has no `budget=` |
| `blobs`, `claim_threshold` | `Engine(...)` | none, `16384` | [Large payloads](../guides/large-payloads.md) |
| `sweep_interval_s` | `Engine(...)` | none | periodic [`sweep()`](../concepts/runs-and-the-engine.md#sweeping) |
| `max_steps`, `max_tokens`, `max_cost`, `deadline_s` | `Budget(...)` | none | per-run limits |
| `run_id` | `engine.start/run(...)` | generated ULID | makes starting idempotent |
| `parent`, `tag` | `engine.start(...)` | none | child-run linkage |

## LLM

| Setting | Where | Default |
|---|---|---|
| `provider`, `model`, `base_url`, `api_key`, `headers`, `chat_path`, `embeddings_path`, `timeout_s`, `params`, `price_in_per_mtok`, `price_out_per_mtok` | `LLMConfig` | see [LLM gateways](../guides/llm-gateways.md#configuring-a-client) |
| `concurrency`, `emit_deltas` | `LLMHandler(...)` | `8`, `True` |
| `max_cached_conversations` | `LLMHandler` attribute | `1024` |
| `max_attempts`, `base_delay`, `max_delay` | `RetryingClient` | `3`, `0.5`, `20.0` |
| `failure_threshold`, `cooldown_s` | `CircuitBreakerClient` | `5`, `30.0` |

## Tools

| Setting | Where | Default |
|---|---|---|
| `requires_approval`, `timeout_s`, `pool`, `name`, `description` | `@tool(...)` | `False`, `60`, `"default"` |
| `delivery`, `idempotency_key`, `guard` | `@tool(...)` | handler default |
| `pools` | `ToolHandler(...)` | `{"default": 16, "delegate": 8}` |
| `guard` | `ToolHandler(...)` | `GuardConfig()` |
| `enabled`, `delivery`, `max_attempts`, `lease_ttl_s`, `heartbeat_s`, `wait_poll_s`, `on_lost_lease`, `fence_results` | `GuardConfig` | `True`, `at_least_once`, `3`, `15.0`, `ttl/3`, `0.5`, `cancel`, `True` |

## Algorithms

| Algorithm | Settings (defaults) |
|---|---|
| `ToolLoop` / `ReAct` | `system=""`, `model=None`, `max_turns=10`, `params={}`, `tool_names=None`, `name`, `stream=False` |
| `PlanAndExecute` | `model=None`, `max_parallel=3`, `max_replans=1`, `synthesize=True`, `planner_system` |
| `Router` | `classifier=None`, `default=None`, `model=None` |
| `Supervisor` | `workers`, `extra_tools=None`, plus `ToolLoop` settings |
| `GenerateTestRepair` | `validator="default"`, `system=""`, `model=None`, `max_iterations=3`, `params={}`, `extract_code=True` |
| `GeneratorCritic` | `critic_system`, `critic_model=None`, plus the above |
| `SelfConsistency` | `n=5`, `temperature=0.8`, `model=None`, `system`, `extract` |
| `RAGAgent` | `index`, `k=5`, plus `ToolLoop` settings |
| `Projection` | `name=""`, `version=1`, `state_type=None`, `every=64` |

## Stores and bus

| Setting | Where | Default |
|---|---|---|
| `path`, `synchronous`, `poll_s` | `SQLiteEventStore` | required, `"NORMAL"`, `0.2` |
| `dsn`, `partitions`, `outbox`, `max_pool` | `PostgresEventStore.create` | required, `16`, `True`, `10` |
| `url`, `partitions`, `prefix`, `outbox` | `RedisEventStore.create` | required, `16`, `"aaa"`, `True` |
| `member`, `batch`, `poll_s`, `lease_ttl_s`, `tick_s`, `handover_timeout_s` | `DistributedBus` | random, `64`, `0.5`, `6.0`, `1.0`, `30.0` |
| `queue_size` | `InProcessBus` | `1024` |
| `keep_last` | `store.trim_outbox` | `100_000` |

## Environment variables

AAA reads **no** environment variables itself, except where *you* reference them as `${VAR}` in `LLMConfig` (`api_key`, `headers`). The
gateway presets reference these names, which you set:

| Preset | Variables |
|---|---|
| `litellm_proxy` | `LITELLM_API_KEY` |
| `portkey` | `PORTKEY_API_KEY`, `PROVIDER_API_KEY` |
| `cloudflare` | `CLOUDFLARE_API_TOKEN`, `PROVIDER_API_KEY` |
| `vercel` | `AI_GATEWAY_API_KEY` |

For the test suite and live smoke test: `AAA_TEST_POSTGRES_URL`, `AAA_TEST_REDIS_URL`, `AAA_TEST_PG_BIN`, and
`AAA_LIVE_BASE_URL`, `AAA_LIVE_MODEL`, `AAA_LIVE_API_KEY`, `AAA_LIVE_HEADERS`, `AAA_LIVE_CHAT_PATH`, `AAA_LIVE_EMBED_MODEL`, `AAA_LIVE_TIMEOUT`.
