"""Vector store vocabulary. A store holds vectors plus a payload (text, metadata); it never embeds text itself."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from aaa.vector.filters import Filter

Metric = Literal["cosine", "dot", "euclidean"]
DEFAULT_NAMESPACE = "default"


class VectorError(Exception):
    pass


class HybridUnavailable(VectorError):
    """The store cannot run hybrid search here (for example, its SQLite build has no FTS5)."""


class DimensionMismatch(VectorError):
    pass


@dataclass
class Record:
    id: str
    vector: list[float]
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Match:
    """One search result. `score` is a similarity for `cosine` and `dot` (higher is better) and a distance for `euclidean`
    (lower is better); see `VectorStore.higher_is_better`. Scores from different stores or metrics are not comparable."""

    id: str
    score: float
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@runtime_checkable
class VectorStore(Protocol):
    """Anything that stores vectors with a payload and finds the nearest ones.

    * `namespace` is a hard partition: records in one namespace are invisible to queries in another.
    * `filter` narrows by metadata (see `aaa.vector.filters`); a store that cannot express a filter raises `UnsupportedFilter`.
    * `dimension` may be `None` until the first upsert; after that every vector must match it (`DimensionMismatch`).
    """

    metric: Metric
    dimension: int | None

    @property
    def higher_is_better(self) -> bool: ...

    async def upsert(self, records: Sequence[Record], *, namespace: str | None = None) -> None: ...

    async def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]: ...

    async def get(self, ids: Sequence[str], *, namespace: str | None = None) -> list[Record]: ...

    async def delete(self, ids: Sequence[str], *, namespace: str | None = None) -> None: ...

    async def count(self, *, namespace: str | None = None) -> int: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class DeleteWhere(Protocol):
    """Optional: delete by metadata filter (returns how many were removed)."""

    async def delete_where(self, filter: Filter, *, namespace: str | None = None) -> int: ...


@runtime_checkable
class DropNamespace(Protocol):
    """Optional: remove a whole namespace (tenant offboarding, erasure)."""

    async def drop_namespace(self, namespace: str) -> None: ...


@runtime_checkable
class HybridSearch(Protocol):
    """Optional: dense + keyword search fused by the store (reciprocal-rank fusion, see `aaa.vector.hybrid`). May raise `HybridUnavailable`."""

    async def query_hybrid(
        self,
        vector: Sequence[float],
        text: str,
        k: int = 5,
        *,
        filter: Filter | None = None,
        namespace: str | None = None,
    ) -> list[Match]: ...


def check_vector(store_dim: int | None, vector: Sequence[float], what: str = "vector") -> int:
    """Raise `DimensionMismatch` unless `vector` fits; returns the dimension in force."""
    n = len(vector)
    if n == 0:
        raise DimensionMismatch(f"empty {what}")
    if store_dim is not None and n != store_dim:
        raise DimensionMismatch(f"{what} has {n} dimensions, the store holds {store_dim}")
    return n
