# Design decisions

The choices that shape AAA, with the alternative each one rejected and the price paid. Useful if you are evaluating the library or
about to extend it.

## 1. The event log is the source of truth

**Decision.** Every run is an append-only stream of immutable events. State is derived by folding events.
**Rejected.** A mutable per-run state dict (as in graph frameworks), with checkpoints as an add-on.
**Why.** Replay, resume after a crash, audit, deterministic tests and process migration all fall out of one mechanism instead of five.
**Cost.** Every transition is a write; reading state needs a fold (mitigated by snapshots); storage grows (mitigated by deltas, claim-check, your retention).

## 2. Algorithms hold no state

**Decision.** An algorithm is `(event, projected state) -> events`.
**Why.** Any worker can continue any run; tests do not need object lifecycles; recovery has nothing to restore.
**Cost.** You write a projection for non-trivial state, and avoid I/O in `on_event`.

## 3. Failures are facts, not exceptions

**Decision.** LLM and tool failures are events the algorithm reads; outcomes are seven distinct kinds.
**Why.** A bus cannot propagate a Python exception across processes, and "failed", "timed out" and "ran out of budget" need different reactions.

## 4. Idempotency instead of exactly-once

**Decision.** At-least-once delivery, made harmless by deterministic event ids, one result id per request, level-triggered algorithms and idempotent appends.
**Rejected.** Distributed transactions / exactly-once delivery claims.
**Why.** Exactly-once is not achievable across an LLM API and arbitrary tools. Making repeats harmless inside the system, and giving tools the means to be idempotent
(keys, fencing, at-most-once), is honest and testable.
**Cost.** Handlers must tolerate repeats; side-effecting tools need care.

## 5. Append, then deliver; enqueue atomically

**Decision.** Events are durable before any handler sees them, and the bus entry is written in the same transaction (Postgres) or Lua script (Redis) as the event.
**Why.** Removes the "stored but never published" crash window without a separate relay process.
**Cost.** The store and the bus are coupled: a custom store must also provide the bus backend to be clustered.

## 6. A partitioned log with leases, not a message queue

**Decision.** Events are partitioned by run; each (group, partition) has one leased owner consuming in order with a cursor.
**Rejected.** Per-message queues with visibility timeouts.
**Why.** A run's events need ordering, and an algorithm needs to see them serially. Partition ownership gives both with no per-message bookkeeping.
**Cost.** Parallelism is across runs, not within one; the partition count is fixed.

## 7. Delta requests for conversations

**Decision.** A continuing LLM request stores only the new messages plus a pointer; the handler reconstructs the full transcript.
**Why.** Storing the whole transcript in every request made log size and per-turn work quadratic (14 MB and 7.5 s at 200 turns, now 0.67 MB and 0.28 s).
**Cost.** A cache and a log-rebuild path in the handler; custom algorithms must opt in via `prev=`.

## 8. One OpenAI-compatible client plus presets

**Decision.** A single HTTP client for the OpenAI dialect, with gateway presets that only set a URL and headers; LiteLLM and custom providers as adapters.
**Why.** Gateways such as Portkey, Cloudflare, Vercel, Kong, Bifrost and MLflow converge on that dialect. One well-tested client beats seven thin ones.
**Cost.** Provider-native features outside that dialect need a custom client.

## 9. Standard-library SQLite on a dedicated thread

**Decision.** `sqlite3` with one worker thread owning the connection, WAL mode, explicit transactions.
**Rejected.** An async SQLite wrapper dependency.
**Why.** Zero extra dependencies, no event-loop blocking, predictable behavior.

## 10. The execution guard: a lease plus durable attempt records

**Decision.** Mutual exclusion via an execution lease with heartbeat; fencing via numbered attempt events recorded before running; policy as configuration.
**Rejected.** Pretending to offer exactly-once tool execution; a separate coordination service.
**Why.** It reuses the lease primitive the bus already needs, keeps the truth in the log, and states its limits (external effects need idempotency keys).

## 11. Tools and Pydantic are the main data path; `Item`/`Codec` are optional

**Decision.** Arbitrary user data flows through typed tool arguments/results, run inputs and indexed documents; `Item` + `Codec` provide a uniform render/serialize/fingerprint layer for your own code.
**Why.** It meets users where their data already has types. The built-ins do not yet *require* `Item`s; making them consume it natively is future work.

## 12. What was left out of the algorithm library on purpose

A survey of agent-algorithm families produced a long list: classical search, HTN, constraint solvers, MDP/POMDP, RL, evolutionary search, consensus.
AAA ships the patterns LLM-agent workloads actually use (loops, planning, routing, verification, deterministic workflows, voting, retrieval) and treats the rest as
*solvers to wrap*, not algorithms to reimplement. The infrastructure items on such lists (retries, circuit breakers, sagas, optimistic concurrency, checkpoint/resume)
live in the runtime where they belong, not in the algorithm catalogue.
