from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aaa.core.ids import new_id, stable_id


class BlobRef(BaseModel):
    """Claim-check pointer to large payloads held in a BlobStore."""

    model_config = ConfigDict(frozen=True)
    uri: str
    sha256: str
    size: int
    media_type: str = "application/octet-stream"


ERASED = "$erased"
# Structural fields that survive erasure: enough to keep the causal skeleton and let a restart or
# `sweep()` reason about a finished run, but never user payload.
_KEEP_ON_ERASE: dict[str, tuple[str, ...]] = {
    "aaa.run.started": ("algo", "parent", "run_tag", "budget"),
    "aaa.run.completed": ("kind",),
}


class Event(BaseModel):
    """Immutable, CloudEvents-compatible envelope. The event log is the source of truth."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    type: str  # namespaced; "aaa.*" is reserved for the library
    source: str = "aaa"
    subject: str | None = None
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stream: str = "default"  # ordering + OCC boundary (usually the run id)
    run_id: str | None = None
    tenant: str | None = None  # owner of the run; inherited by every derived event
    trace_id: str | None = None
    span_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    seq: int = -1  # assigned by the EventStore on append (0-based, per stream)
    schema_ref: str = ""  # e.g. "aaa.llm.requested/v1"; empty = unversioned
    data: dict[str, Any] = Field(default_factory=dict)
    blobs: tuple[BlobRef, ...] = ()
    meta: dict[str, str | int | float | bool] = Field(default_factory=dict)
    # transient: set by a store on an event it returned because that id was already appended
    dup: bool = Field(default=False, exclude=True)

    def caused(self, type: str, data: dict[str, Any] | None = None, **kw: Any) -> Event:
        """Derive a follow-up event preserving the causal chain and stream. The id is a pure
        function of (parent, type, data), so re-deriving after a crash is idempotent."""
        base: dict[str, Any] = {
            "stream": self.stream,
            "run_id": self.run_id,
            "tenant": self.tenant,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id or self.id,
            "causation_id": self.id,
            "meta": self.meta,
        }
        base.update(kw)
        payload = data or {}
        if "id" not in base:  # deterministic: same parent + type + content => same id
            base["id"] = stable_id(self.id, type, json.dumps(payload, sort_keys=True, default=str))
        return Event(type=type, data=payload, **base)


def scrubbed(e: Event) -> Event:
    """The event with its payload erased: id, type, stream, seq, time, causal ids and tenant stay
    (the audit skeleton); `data`, blobs, `meta` and `subject` are gone. Idempotent."""
    keep = {k: e.data[k] for k in _KEEP_ON_ERASE.get(e.type, ()) if k in e.data}
    return e.model_copy(
        update={"data": {ERASED: True, **keep}, "blobs": (), "meta": {}, "subject": None}
    )
