# Backup and restore

The event log is the source of truth, so a backup of the log (plus the blob store, if you use one) is a backup of everything: run state, sessions, scores and the audit trail. Projections and snapshots
are derived and are rebuilt from the log on demand.

## What to back up

| Piece | How |
|---|---|
| **Postgres event store** | `pg_dump -Fc` of the database (it holds events, the bus outbox, consumer cursors and snapshots). Point-in-time recovery via WAL archiving is the better tool for production; the dump below is the simple, verified path. |
| **Redis event store** | RDB/AOF persistence of the instance (`appendonly yes`). Redis is a fast store, not a durable archive: if the log must survive, use Postgres. |
| **SQLite** | The `.db` file (with its `-wal` file) copied while no writer is active, or `sqlite3 agent.db ".backup out.db"`. |
| **Blob store** (large payloads, attachments) | The `FileBlobStore` directory, or the bucket, copied **at the same time or after** the event store. Blobs are content-addressed and immutable, so copying them later is safe; copying them earlier can miss blobs that newer events refer to. |
| **Vector indexes and memory backends** | Their own backup procedures. They are not part of the event log; an ingestion `Pipeline` can rebuild an index from its sources. |

## Restore, then verify

```bash
pg_restore -d restored --no-owner agent.dump
aaa store verify postgresql://user:pw@host/restored --blobs /restored/blobs --deep
```

`aaa store verify` reads every run and checks that sequence numbers are contiguous, that each log starts with `aaa.run.started`, that event ids are unique, that each run completed at most once, that
every child run exists, and (with `--blobs`) that every blob an event refers to is present (`--deep` also re-hashes it). It prints the newest event time: compare it with the moment you expected the backup to
reach. It exits non-zero when it finds a problem, so it can gate a restore drill in CI or cron.

Then start workers against the restored store: open runs are picked up by the usual recovery (`engine.recover(...)`, or the sweeper in a cluster), and tool calls whose outcome is unknown follow the
[tool guard](../guides/tool-guard.md) rules (`at_most_once` calls are not repeated). Expect those runs to have lost any work done after the backup's last event.

## Things a backup does not do

- **Erasure does not reach backups.** Data erased with `engine.erase(...)` / `tenant.erase_all()` is still in older backups. Either keep backups short-lived, or record erasures and replay them after any restore.
- **A restore is a rollback.** Events after the backup are gone; external effects of those events (emails sent, payments made) are not undone.
- **Stores must match their layout version.** A restore into a newer release migrates forward; a store written by a *newer* release is refused by an older one (`SchemaTooNew`). See [Upgrading](upgrading.md).

## Verified, and not

Verified: `pg_dump -Fc` / `pg_restore` of a Postgres store holding 300 runs (2,700 events) into a second database, `aaa store verify` reporting it consistent, and the same command reporting a
deleted event as a sequence gap; blob checks against a `FileBlobStore`. **Not verified**: WAL/PITR recovery, Redis RDB/AOF restore, restores across Postgres major versions, or a restore drill under load.
Run your own drill before you need it.
