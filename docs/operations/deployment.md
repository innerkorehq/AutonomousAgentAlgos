# Deployment checklist

A short list of the decisions that matter before you run AAA for real.

## 1. Pick your topology

| You have | Use |
|---|---|
| One process, losing state on restart is fine | defaults (`Engine()`) |
| One machine, runs must survive restarts | `SQLiteEventStore` + `recover()` at startup |
| Several workers, or an API process separate from workers | `PostgresEventStore` (or Redis) + `DistributedBus`; clients use `ClientBus` |

## 2. Before you go live

- [ ] **Every run has a `Budget`** with at least `deadline_s`, plus `max_steps` and `max_cost` (set `price_*_per_mtok` so cost counts).
      Set `Engine(default_budget=...)` as a safety net.
- [ ] **Every loop is bounded**: `max_turns`, `max_iterations`, `max_replans`.
- [ ] **LLM timeout** is set (`LLMConfig.timeout_s`) and appropriate for your slowest model.
- [ ] **Retries are in one place**: either your gateway or `RetryingClient`, not both.
- [ ] **Side-effecting tools** have a deliberate [delivery policy](../guides/tool-guard.md), idempotency keys where the target supports them, and
      `requires_approval=True` for anything destructive or irreversible.
- [ ] **Approval signals are authorized** in whatever service lets people approve (events are unauthenticated).
- [ ] **Secrets** come from environment variables via `${VAR}`, not literals in config.
- [ ] **You have run the live smoke test** against your actual gateway and model (`scripts/live_smoke.py`).
- [ ] **Persistence is on** and backed up (event tables are your ledger).
- [ ] **Sweeping is on** in a cluster: `Engine(sweep_interval_s=5)` on at least one worker.
- [ ] **Outbox trimming is scheduled** in a cluster.
- [ ] **Algorithm names and tool sets are identical** on every worker, and algorithm names are unique.
- [ ] **A kill-test passed in staging**: `kill -9` a worker mid-run and watch the run finish.

## 3. Sizing

| Setting | Guidance |
|---|---|
| `partitions` | Upper bound on parallel consumers per group. 16 is a sane default, 32-64 for larger fleets. **Cannot change later.** |
| Workers | Start with 2-3. Work is spread by partition (by run), so many concurrent runs, not one big run, is what scales. |
| `LLMHandler(concurrency=...)` | Simultaneous LLM calls per worker. Match your provider rate limits divided by workers. |
| Tool pools | `default` 16 and `delegate` 8 per worker; give slow or fragile backends (databases, third-party APIs) their own small pool. |
| Postgres connections | Each worker uses up to `max_pool + 2` (default 12). |
| `lease_ttl_s` (bus) | 6s default. Lower for faster failover; keep `tick_s` below a third of it. |
| Tool `lease_ttl_s` | 15s default. Must exceed your worst realistic event-loop stall; heartbeat is `ttl/3`. |

## 4. Health and observability

AAA exports traces from the event log with the [observer](../guides/observability.md) (OpenTelemetry, Langfuse, LangSmith, Phoenix); it does not yet
ship metrics (counters and histograms). The event log is the audit trail and the raw material:

- **Per run**: read the stream. `aaa.llm.completed.usage` has tokens, `aaa.run.completed` the outcome, `aaa.tool.started.worker` which worker.
- **Fleet**: count `aaa.run.completed` by `kind` over time; alert on `failed` and on runs open longer than expected.
- **Bus lag** (cluster): `backend.tail(p) - backend.get_cursor(group, p)` per partition.
- **Subscribe an exporter**: a handler on `aaa.run.completed` (with a `group=`) can push metrics to Prometheus/StatsD, and
  `to_cloudevent(event)` produces a portable record.
- **Logs**: the library logs under `aaa.*` (`aaa.engine`, `aaa.bus`, `aaa.dbus`, `aaa.recovery`, `aaa.projection`, `aaa.postgres`). Handler exceptions
  are logged with tracebacks and also surface as failed runs.

## 5. Upgrades

- **Projections** are versioned. A release that changes one bumps its `version`, which makes old snapshots be ignored and
  rebuilt from the log automatically. No migration step.
- **Event schemas** you register are upcast when an event is **written** (`SchemaRegistry.upcaster`); events already stored are never rewritten or
  upcast on read, so a fold must keep tolerating every shape the log has ever held. See the [stability policy](../policy/stability.md).
- **Workers on different versions**: during a rolling deploy, workers share one log. Keep algorithm names/tools compatible, or deploy
  with a short drain.
- The partition count and table/key layout are part of the data format; check release notes before upgrading across 0.x versions.
