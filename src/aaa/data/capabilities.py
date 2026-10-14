from __future__ import annotations

from collections.abc import Hashable
from typing import Protocol, runtime_checkable

from aaa.data.codec import Codec
from aaa.data.item import Item


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Scorer(Protocol):
    """Utility / heuristic / quality score for an item (higher is better)."""

    async def score(self, item: Item) -> float: ...


@runtime_checkable
class Keyer(Protocol):
    """Stable identity for dedup and search closed-sets."""

    def key(self, item: Item) -> Hashable: ...


class MissingCapability(Exception):
    pass


def preflight(required: set[str], available: dict[str, object]) -> None:
    """Fail fast at startup, listing everything an algorithm needs but wasn't given."""
    missing = sorted(required - available.keys())
    if missing:
        raise MissingCapability(f"missing capabilities: {', '.join(missing)}")


__all__ = ["Codec", "Embedder", "Keyer", "MissingCapability", "Scorer", "preflight"]
