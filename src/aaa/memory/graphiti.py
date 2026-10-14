"""Graphiti adapter (pip install graphiti-core; the temporal-graph engine behind Zep).
Written against graphiti-core 0.30.

    from graphiti_core import Graphiti
    g = Graphiti("bolt://localhost:7687", "neo4j", "password")   # you configure LLM, embedder, database
    await g.build_indices_and_constraints()
    memory = GraphitiMemory(g)

Graphiti turns each episode into entities and *facts with validity windows* using its own LLM; the
facts it returns carry `valid_at` / `invalid_at`, so `search(as_of=...)` answers "what was true then".
By default only currently-valid facts are returned. The extraction LLM is Graphiti's own, so its spend
is outside the engine's budget and gateway.

Scoping: one Graphiti `group_id` per (tenant, user, agent, namespace) (`Scope.slug()`). Tenant-wide
and tenant+user scopes expand to every group with that prefix by asking the graph for its groups
(Cypher; Neo4j/FalkorDB/Neptune), so `forget(Scope(tenant=...))` erases all of a tenant's graphs.

Writes pass a deterministic episode `uuid`, so a repeated write is the same episode: `idempotent`."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from aaa.core.ids import stable_id
from aaa.memory.types import AddResult, MemoryEntry, MemoryItem, Scope

log = logging.getLogger("aaa.memory.graphiti")


def _ops() -> SimpleNamespace:
    """graphiti-core's own classes, imported lazily (tests substitute stand-ins)."""
    from graphiti_core.edges import EntityEdge  # pyright: ignore[reportMissingImports]
    from graphiti_core.nodes import EpisodeType, EpisodicNode  # pyright: ignore[reportMissingImports]

    return SimpleNamespace(
        EntityEdge=EntityEdge, EpisodeType=EpisodeType, EpisodicNode=EpisodicNode
    )


class GraphitiMemory:
    idempotent = True

    def __init__(
        self, graphiti: Any, *, source_description: str = "aaa", current_only: bool = True
    ) -> None:
        self.g = graphiti
        self.source_description, self.current_only = source_description, current_only
        self._locks: dict[str, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )  # episodes of a group are ordered

    # -- scope mapping ------------------------------------------------------------------------
    async def _groups(self, scope: Scope) -> list[str]:
        exact = scope.slug()
        if scope.agent or scope.namespace or not scope.tenant:
            return [exact]  # fully specified (or untenanted): exactly one group
        prefix = scope.slug()
        res = await self.g.driver.execute_query(
            "MATCH (n) WHERE n.group_id STARTS WITH $p RETURN DISTINCT n.group_id AS g", p=prefix
        )
        rows = (
            res[0] if isinstance(res, tuple) else res
        )  # drivers differ in what they wrap around the rows
        found = {r["g"] for r in rows if r["g"] == prefix or r["g"].startswith(prefix + "--")}
        return sorted(found | {exact})

    # -- Memory -------------------------------------------------------------------------------
    async def add(
        self, entries: Sequence[MemoryEntry], scope: Scope, *, key: str | None = None
    ) -> AddResult:
        ops = _ops()

        group = scope.slug()
        ids: list[str] = []
        async with self._locks[group]:
            for i, e in enumerate(entries):
                uid = e.id or stable_id(group, key or "", str(i), " ".join(e.text.lower().split()))
                body = f"{e.role}: {e.text}" if e.role else e.text
                res = await self.g.add_episode(
                    name=uid,
                    episode_body=body,
                    source_description=self.source_description,
                    reference_time=e.time or datetime.now(UTC),
                    source=ops.EpisodeType.message if e.role else ops.EpisodeType.text,
                    group_id=group,
                    uuid=uid,
                )
                ids.append(str(getattr(getattr(res, "episode", None), "uuid", uid)))
        return AddResult(ids=ids)

    @staticmethod
    def _item(edge: Any) -> MemoryItem:
        return MemoryItem(
            id=str(edge.uuid),
            text=str(edge.fact),
            kind="fact",
            created_at=getattr(edge, "created_at", None),
            valid_at=getattr(edge, "valid_at", None),
            invalid_at=getattr(edge, "invalid_at", None),
            meta={
                "relation": getattr(edge, "name", None),
                "group_id": getattr(edge, "group_id", None),
            },
            raw=edge,
        )

    async def search(
        self,
        query: str,
        scope: Scope,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        edges = await self.g.search(
            query, group_ids=await self._groups(scope), num_results=k * 3 if as_of else k
        )
        items = [self._item(e) for e in edges]
        if as_of is not None:
            items = [m for m in items if m.valid_on(as_of)]
        elif self.current_only:
            items = [m for m in items if m.invalid_at is None]
        if filters:
            items = [m for m in items if all(m.meta.get(f) == v for f, v in filters.items())]
        return items[:k]

    async def delete(self, ids: Sequence[str], scope: Scope) -> int:
        ops = _ops()

        groups = set(await self._groups(scope))
        mine: list[str] = []
        for uid in ids:
            try:
                edge = await ops.EntityEdge.get_by_uuid(self.g.driver, uid)
            except Exception as exc:  # noqa: BLE001 - graphiti raises its own EdgeNotFoundError
                log.debug("not deleting %s: %s", uid, exc)
                continue
            if edge.group_id in groups:
                mine.append(uid)
        if mine:
            await ops.EntityEdge.delete_by_uuids(self.g.driver, mine)
        return len(mine)

    async def forget(self, scope: Scope) -> int:
        ops = _ops()

        n = 0
        for group in await self._groups(scope):
            episodes = await ops.EpisodicNode.get_by_group_ids(self.g.driver, [group])
            n += len(episodes)
            await ops.EpisodicNode.delete_by_group_id(
                self.g.driver, group
            )  # entities, edges, episodes
        return n

    async def list(self, scope: Scope, *, limit: int = 100) -> list[MemoryItem]:
        episodes = await self.g.retrieve_episodes(
            datetime.now(UTC), last_n=limit, group_ids=await self._groups(scope)
        )
        return [
            MemoryItem(
                id=str(ep.uuid),
                text=str(ep.content),
                kind="episode",
                created_at=getattr(ep, "created_at", None),
                valid_at=getattr(ep, "valid_at", None),
                raw=ep,
            )
            for ep in episodes
        ]

    async def aclose(self) -> None:
        return None  # the caller owns the Graphiti object (call `await g.close()` yourself)
