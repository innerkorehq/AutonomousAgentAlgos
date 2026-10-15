"""What flows through ingestion: a loader yields `Document`s, a chunker cuts each into `Chunk`s, the pipeline turns chunks into index records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def digest(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


@dataclass
class Document:
    """One source's text. `source` is its stable identity (a relative path, a URL): the same source on the next sync is the same document."""

    source: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        """Identity of the document's content AND metadata: a changed title or page number re-syncs it (reusing vectors when the text is the same)."""
        return digest(self.text + "\x00" + json.dumps(self.meta, sort_keys=True, default=str), 32)


@dataclass
class Chunk:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Loader(Protocol):
    """Yields every document it can currently see. A source missing from one run's output and present in the last is treated as removed."""

    def load(self) -> AsyncIterator[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...
