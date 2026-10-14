"""CloudEvents 1.0 (structured JSON mode) <-> aaa Event. Dependency-free by design:
the output is spec-conformant and interoperates with the official SDK / any broker."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from aaa.core.errors import SchemaError
from aaa.events.event import BlobRef, Event

# CloudEvents extension names: lowercase alphanumerics only; values str/int/bool.
_EXT = {
    "stream": "stream",
    "run_id": "runid",
    "tenant": "tenant",
    "trace_id": "traceid",
    "span_id": "spanid",
    "causation_id": "causationid",
    "correlation_id": "correlationid",
    "seq": "seq",
}


def to_cloudevent(e: Event) -> dict[str, Any]:
    out: dict[str, Any] = {
        "specversion": "1.0",
        "id": e.id,
        "source": e.source,
        "type": e.type,
        "time": e.time.isoformat(),
        "datacontenttype": "application/json",
        "data": e.data,
    }
    if e.subject is not None:
        out["subject"] = e.subject
    if e.schema_ref:
        out["dataschema"] = f"aaa:{e.schema_ref}"
    for attr, name in _EXT.items():
        v = getattr(e, attr)
        if v is not None and not (attr == "seq" and v < 0):
            out[name] = v
    if e.meta:  # may contain floats, which CloudEvents extensions forbid -> JSON string
        out["aaameta"] = json.dumps(e.meta, sort_keys=True)
    if e.blobs:
        out["aaablobs"] = json.dumps([b.model_dump() for b in e.blobs], sort_keys=True)
    return out


def from_cloudevent(ce: dict[str, Any] | str | bytes) -> Event:
    if isinstance(ce, (str, bytes)):
        ce = json.loads(ce)
    assert isinstance(ce, dict)
    if ce.get("specversion") != "1.0":
        raise SchemaError(f"unsupported CloudEvents specversion {ce.get('specversion')!r}")
    for req in ("id", "source", "type"):
        if req not in ce:
            raise SchemaError(f"CloudEvent missing required attribute {req!r}")
    ds = ce.get("dataschema", "")
    kw: dict[str, Any] = {
        "id": ce["id"],
        "source": ce["source"],
        "type": ce["type"],
        "subject": ce.get("subject"),
        "schema_ref": ds.removeprefix("aaa:") if ds.startswith("aaa:") else "",
        "data": ce.get("data") or {},
    }
    if "time" in ce:
        kw["time"] = datetime.fromisoformat(ce["time"])
    for attr, name in _EXT.items():
        if name in ce:
            kw[attr] = ce[name]
    if "aaameta" in ce:
        kw["meta"] = json.loads(ce["aaameta"])
    if "aaablobs" in ce:
        kw["blobs"] = tuple(BlobRef(**b) for b in json.loads(ce["aaablobs"]))
    return Event(**kw)
