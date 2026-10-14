from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from aaa.core.ids import new_id
from aaa.events.event import BlobRef

T = TypeVar("T")


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str = "unknown"
    parents: tuple[str, ...] = ()
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = 1


class Item(BaseModel, Generic[T]):
    """Envelope around user data. Algorithms see Items; users own the payload type T."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=new_id)
    kind: str  # open string; selects the Codec
    payload: T
    meta: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    content_hash: str = ""
    embedding: list[float] | None = None
    blobs: tuple[BlobRef, ...] = ()


def fingerprint_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()
