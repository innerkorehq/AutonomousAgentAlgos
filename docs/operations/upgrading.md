# Upgrading

## The rules that make upgrades boring

- The [stability policy](../policy/stability.md) applies: patch releases never change APIs or the log format; minor releases may remove things that were deprecated for two minors, and say so under *Breaking* in the changelog.
- **The log format is additive.** New fields have defaults, existing ones are never removed, and stores never rewrite events. The test-suite folds frozen logs from older releases through every current projection.
- **Projections are versioned.** A release that changes one invalidates old snapshots, which are rebuilt from the log. No migration step.
- **Store layout is versioned** (`SCHEMA_VERSION`, recorded in the store). Opening an older store migrates it forward, one transaction per step, under the same lock first-time setup uses. A store written by a newer release is **refused** by an older one rather than guessed at.

## A safe procedure

1. Read the changelog's *Breaking* and *Deprecated* entries for every release you are skipping.
2. **Back up** ([how](backup-restore.md)) and run `aaa store verify` on the copy.
3. Roll workers one at a time: during the roll, old and new workers share one log. If the release changes an algorithm or tool in an incompatible way, drain first (stop starting runs, let open runs finish) instead.
4. After the roll, `aaa store check <store>` reports the layout version and the run count.
5. **Do not roll back across a layout change.** An older release refuses the migrated store. Rollback means restoring the backup.

## From 0.2 to 0.3

- Nothing in 0.2 code is removed. The new surface is additive: multimodal `Message.parts`, native Anthropic and Gemini clients, hybrid search and ingestion, built-in tools and the sandbox, the MCP server, the new algorithms, eval extras.
- **`aaa-server` is a separate package** with its own version (`pip install aaa-server`). The library has no web dependencies.
- **Store layout is unchanged** (version 1). Redis stores now size their client pool from the partition count (`max_connections`, default `max(100, 16 * partitions)`), which matters only for large clusters.
- `LoopState` gains optional `agent`/`handoffs` fields, omitted from its serialized form unless a `Swarm` uses them, so existing snapshots and logs are unaffected.
- Python 3.13 is supported.
