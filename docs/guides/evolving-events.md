# Evolving events and stores

The event log is your data and it outlives code. Three different things can change between releases, each with its own rule.

## 1. The shape of an event: additive only

Stored events are **never rewritten and never upcast on read**. So every fold (every `Projection`, every algorithm, every exporter) must keep understanding every
shape the log has ever held. That gives one rule for changing an event type:

- **Add** a field, with a default (`data.get("x", default)` in a fold; an optional field in a pydantic model).
- **Never remove or retype** a field, and never reuse an event type for a different meaning.
- If the meaning really changes, introduce a **new event type** and keep reading the old one.

`SchemaRegistry` (`aaa.events.SchemaRegistry`) can upcast events you register **when they are written** (v1 payload in, latest payload stored), which is useful for your own
event types; it does not touch what is already stored.

### The safety net

`tests/fixtures/events_v0_2/` contains real logs produced by v0.2.0 (24 scenarios covering every built-in event type: tools, approvals, memory, guardrails,
context summaries, child runs, plans, routing, workflows, state machines, chat sessions, streaming, budget stops, cancellation, timeouts, erasure, fencing), together with the
state every `Projection` folded out of them. `tests/unit/test_event_compat.py` replays those files with the **current** code and requires identical state, and checks
that the trace and CloudEvents exporters still read them. If you change how an existing event is read and this test fails, the change is not additive: make it so, or add a new event type.
The fixtures are frozen. A future format gets a new directory; `scripts/gen_event_fixtures.py` documents how they were made.

## 2. Projection state: bump the version

A `Projection` snapshots its state every N events. When you change the fold or the state shape, bump `Projection(version=...)`: snapshots with another version are ignored and
the state is rebuilt from the log. Nothing detects a forgotten bump, so make it part of the change (the compat test above will usually catch it).

## 3. The stores' own layout: versioned and migrated

The durable stores (SQLite, PostgreSQL, Redis) record the version of the **layout** they have been migrated to (tables, columns, key names), separate from event shapes:

| Store | Where |
|---|---|
| SQLite | `PRAGMA user_version` |
| PostgreSQL | `aaa_meta` row `schema_version` |
| Redis | key `<prefix>:meta:schema_version` |

On open:

- a store written by a **newer** release than the one running is refused (`SchemaTooNew`), so an old worker never guesses at a layout it does not know. Upgrade the library
  on every worker before (or together with) the first one that migrates.
- a store with **no version** was written by 0.1 or 0.2, whose layout is version 1; it is adopted and stamped.
- an **older** version is migrated forward one step at a time, each step in its own transaction where the backend has them (SQLite, PostgreSQL), under the same advisory lock that already serialises first-time setup, so workers starting together migrate once.
- a missing step raises `SchemaGap` instead of guessing.

Version 1 is the current layout; there are no migrations yet, and the release notes will say when there is one. `await store.schema_version()` reports it.

For a rolling upgrade across a layout change, the order is: stop or drain old workers (they will refuse the new layout), upgrade, start. Take a backup first (see the operations guide).
