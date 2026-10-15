"""Persisted-layout versioning for the durable event stores.

`SCHEMA_VERSION` numbers the LAYOUT a store writes (tables, columns, key names) - not the shape of events, which is governed by the additive-only rule in
docs/policy/stability.md and never rewritten. Each durable store records the version it has been migrated to (SQLite `PRAGMA user_version`, Postgres
`aaa_meta('schema_version')`, Redis `<prefix>:meta:schema_version`) and on open:

* a store written by a NEWER release is refused (`SchemaTooNew`): an older release must not guess at a layout it does not know;
* an older (or unversioned, i.e. released before this mechanism existed: that layout IS version 1) store is brought forward one step at a time,
  each step in its own transaction where the backend has them, under the same lock the first-time setup already uses.

Version 1 is the layout of 0.1 and 0.2. A release that changes the layout adds `MIGRATIONS[n]` (n-1 -> n) to the store module and raises `SCHEMA_VERSION`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1


class SchemaTooNew(RuntimeError):
    """The store was written by a newer release than this one."""


class SchemaGap(RuntimeError):
    """No migration is registered for a step between the store's version and this release's."""


def pending(found: int, target: int, steps: Mapping[int, Any]) -> list[int]:
    """The migration steps to run, in order: `steps[v]` migrates version v-1 to v. Raises if the store is newer, or a step is missing."""
    if found > target:
        raise SchemaTooNew(
            f"this event store uses layout version {found}, but this release of aaa understands up to {target}; upgrade aaa (a store is never opened by an older release)"
        )
    need = list(range(found + 1, target + 1))
    if missing := [v for v in need if v not in steps]:
        raise SchemaGap(
            f"no migration registered for layout version(s) {missing} (store is at {found}, this release needs {target})"
        )
    return need
