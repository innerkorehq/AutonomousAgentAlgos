# Performance and scaling

What has been measured, what has not, and how to keep long runs cheap.

!!! warning "Scope of these numbers"
    The benchmark below measures **the library's own overhead** on one machine (Apple M4, SQLite, a scripted model, ~400-byte
    tool results). It says nothing about model latency, network, Postgres/Redis throughput or many-worker scaling, none of
    which have been load tested. Re-run it for your hardware: `uv run python scripts/bench_growth.py 50 100 200`.

## Long conversations

The naive design, writing the whole conversation into every LLM request event, makes log size and per-turn work grow with the
square of the number of turns. AAA avoids it three ways:

1. **Delta requests.** A continuing request stores only the *new* messages plus a pointer to the previous request
   (`conv: {prev, delta}`). The `LLMHandler` rebuilds the full transcript the provider needs from an in-memory cache; after a
   restart or when another worker takes over, it rebuilds once from the log.
2. **Tiny algorithm state.** `ToolLoop` keeps a request id and a few counters, not the transcript, so its snapshots are ~1 KB at any length.
3. **O(1) tool-call bookkeeping.** "Has this request been resolved?" is a keyed lookup, not a scan of the stream.

Measured, a tool loop with ~400-byte tool results:

| Turns | Log size before → after | Wall time before → after |
|---|---|---|
| 50 | 999 KB → 170 KB | 0.22 s → 0.07 s |
| 100 | 3,667 KB → 337 KB | 1.17 s → 0.14 s |
| 200 | 14,032 KB → 673 KB | 7.53 s → 0.28 s |
| 400 | n/a → 1,344 KB | n/a → 0.56 s |

Doubling the turns now doubles the totals. Snapshots complement this: for a 40-turn run, state folds drop from 6,724 to 407.

**What this does not reduce:** the provider payload. Each model call must still be sent the whole transcript, so *token cost per
call* grows with the conversation. Use provider prompt caching, trim or summarize old turns in your own algorithm, or move
work into child runs with fresh context (`PlanAndExecute` does this naturally).

## Measured throughput

`scripts/bench_cluster.py` starts N full engines (each with its own store connections and bus) **in one process**, a client submits runs through a `ClientBus`, and every run is a tiny `ToolLoop` (a scripted model, one
trivial tool call, about nine events). The model takes no time, so these numbers are the library's and the backend's overhead; a real model call (seconds) makes the backend irrelevant for latency and
leaves only the per-run cost. 100 runs per row, at most 20 in flight, 16 partitions, one Apple M4 laptop (10 cores) running the client, all workers and the backend (Redis 8 natively, Postgres 16 in Docker with 2 CPUs).
Workers share those cores, so **adding workers here shows coordination overhead, not the scale-out you would get from separate machines.**

| Backend | Workers | Runs/s | p50 latency | p95 latency |
|---|---|---|---|---|
| Redis | 1 | 116 | 162 ms | 227 ms |
| Redis | 2 | 96 | 193 ms | 287 ms |
| Redis | 4 | 83 | 225 ms | 315 ms |
| Postgres | 1 | 37 | 496 ms | 758 ms |
| Postgres | 2 | 38 | 506 ms | 752 ms |
| Postgres | 4 | 43 | 452 ms | 600 ms |

Read it as: on one machine a Postgres-backed worker completes tens of trivial runs a second and a Redis-backed one around a hundred; neither is the bottleneck of an agent that waits on a model. Your numbers will
differ: re-run the script against your own backend (`uv run python scripts/bench_cluster.py postgres $DSN 1 2 4`).

**Connections are the first limit you hit.** Each worker's Postgres store holds a pool (10 by default), and every client waiting on a run's result holds a connection while it waits. With Postgres's default
`max_connections=100`, four workers plus a client with 20 waiters already ran out (`too many clients already`, retried and logged). Budget connections as
`(workers + clients) * pool size + concurrent waiters`, raise `max_connections` or put PgBouncer in front, and bound how many runs a client awaits at once. Redis clients are bounded by `max_connections` on the
pool (see `RedisEventStore.create`): every consumer group blocks on each partition it owns.

## Where the time goes

- Per event: one append (store latency), one bus hop, one projection load (snapshot + a few events).
- Per LLM call: dominated by the model.
- Per tool call: your tool, plus lease/attempt bookkeeping when the [guard](../guides/tool-guard.md) is enabled (two to four small store operations).
- A step's cost rises with store latency, because events are appended before they are delivered. Keep the store close to the workers.

## Scaling out

- **Parallelism is across runs.** A run's events are consumed in order by one worker at a time, so one run does not get faster with more workers;
  many concurrent runs spread across partitions and workers.
- **Partitions bound parallelism per group.** With `partitions=16`, at most 16 workers consume a given group at once.
- **Postgres**: appends within a partition serialize on a row lock to keep the bus outbox gap-free and commit-ordered; that gives `partitions`-way write parallelism.
- **Read amplification**: each consumer group scans all events of the partitions it owns, so cost grows with the number of groups (six built-in groups plus one per algorithm).
- **Fan-out inside a run** (`PlanAndExecute`, `Supervisor`, `SelfConsistency`) runs concurrently; bound it with `max_parallel`, `LLMHandler(concurrency=...)` and tool pools.

## Practical tuning

| Symptom | Try |
|---|---|
| High latency between steps | Lower store latency; for Postgres confirm LISTEN/NOTIFY is working (otherwise polling adds up to `poll_s`). |
| Runs queue up under load | More workers; raise `LLMHandler(concurrency)` if the provider allows; check pool sizes. |
| Log growing fast | Streaming deltas (set `emit_deltas=False`), large payloads (enable [claim-check](../guides/large-payloads.md)), retention. |
| Slow failover | Lower `lease_ttl_s` and `tick_s` on the bus (Redis); remember Postgres frees leases at once on crash. |
| Tool timeouts too tight or loose | `@tool(timeout_s=...)` and, for guard takeover, `GuardConfig.lease_ttl_s`. |
