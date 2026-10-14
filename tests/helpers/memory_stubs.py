"""Faithful-enough stand-ins for the external memory libraries, so the adapters are held to the shared
contract without servers, API keys or the libraries themselves.

Every stub records the calls it receives in `calls` as (method, args, kwargs). The optional
test_memory_signatures.py replays those exact calls against the REAL library's signatures
(`inspect.signature(...).bind`), so a stub cannot drift from reality unnoticed."""

from __future__ import annotations

import re
import uuid as uuidlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx

_W = re.compile(r"\w+")


def words(s: str) -> set[str]:
    return {w.lower() for w in _W.findall(s)}


def overlap(q: str, text: str) -> int:
    return len(words(q) & words(text))


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def rec(self, method: str, /, *args: Any, **kw: Any) -> None:
        self.calls.append((method, args, kw))


# ---- Mem0 (open-source AsyncMemory and hosted AsyncMemoryClient) ---------------------------------
class StubMem0(Recorder):
    hosted = False

    def __init__(self) -> None:
        super().__init__()
        self.rows: dict[str, dict[str, Any]] = {}

    def _match(self, row: dict, filters: dict) -> bool:
        for k, v in filters.items():
            if k == "user_id":
                if v != "*" and row["user_id"] != v:  # "*" is Mem0's wildcard entity filter
                    return False
            elif row["metadata"].get(k) != v:
                return False
        return True

    async def add(self, messages, *, user_id=None, metadata=None, infer=True, **kw):
        self.rec("add", messages, user_id=user_id, metadata=metadata, infer=infer, **kw)
        out = []
        for m in messages:
            mid = uuidlib.uuid4().hex
            self.rows[mid] = {
                "id": mid,
                "memory": m["content"],
                "user_id": user_id,
                "metadata": dict(metadata or {}),
                "created_at": datetime.now(UTC).isoformat(),
            }
            out.append({"id": mid, "memory": m["content"], "event": "ADD"})
        return {"results": out}

    async def search(self, query, *, top_k=20, filters=None, threshold=0.1, **kw):
        self.rec("search", query, top_k=top_k, filters=filters, threshold=threshold, **kw)
        hits = [
            {**r, "score": float(overlap(query, r["memory"]))}
            for r in self.rows.values()
            if self._match(r, filters or {}) and overlap(query, r["memory"])
        ]
        hits.sort(key=lambda r: -r["score"])
        return {"results": hits[:top_k]}

    async def get_all(self, *, filters=None, top_k=20, **kw):
        self.rec("get_all", filters=filters, top_k=top_k, **kw)
        return {"results": [r for r in self.rows.values() if self._match(r, filters or {})][:top_k]}

    async def update(self, memory_id, text=None, **kw):
        self.rec("update", memory_id, text=text, **kw)
        self.rows[memory_id]["memory"] = text
        return {"message": "updated"}

    async def delete(self, memory_id, **kw):
        self.rec("delete", memory_id, **kw)
        self.rows.pop(memory_id, None)
        return {"message": "deleted"}

    async def delete_all(self, user_id=None, **kw):
        self.rec("delete_all", user_id=user_id, **kw)
        for i in [i for i, r in self.rows.items() if r["user_id"] == user_id]:
            del self.rows[i]
        return {"message": "deleted"}


class StubMem0Hosted(StubMem0):
    hosted = True

    async def add(self, messages, *, filters=None, metadata=None, infer=True, **kw):
        self.rec("add", messages, filters=filters, metadata=metadata, infer=infer, **kw)
        out = []
        for m in messages:
            mid = uuidlib.uuid4().hex
            self.rows[mid] = {
                "id": mid,
                "memory": m["content"],
                "user_id": (filters or {})["user_id"],
                "metadata": dict(metadata or {}),
                "created_at": datetime.now(UTC).isoformat(),
            }
            out.append({"id": mid, "memory": m["content"], "event": "ADD"})
        return {"results": out}

    async def get_all(self, *, filters=None, page_size=20, **kw):  # hosted paginates with page_size
        self.rec("get_all", filters=filters, page_size=page_size, **kw)
        return {
            "results": [r for r in self.rows.values() if self._match(r, filters or {})][:page_size]
        }

    async def delete_all(self, *, filters=None, **kw):
        self.rec("delete_all", filters=filters, **kw)
        for i in [
            i for i, r in self.rows.items() if r["user_id"] == (filters or {}).get("user_id")
        ]:
            del self.rows[i]
        return {"message": "deleted"}


StubMem0Hosted.__module__ = "mem0.client.main"  # how Mem0Memory recognises the hosted client


# ---- Graphiti -----------------------------------------------------------------------------------
class StubGraphiti(Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.edges: list[SimpleNamespace] = []
        self.episodes: list[SimpleNamespace] = []
        self.driver = SimpleNamespace(execute_query=self._execute_query)

    async def _execute_query(self, query, **params):
        self.rec("execute_query", query, **params)
        groups = sorted({e.group_id for e in self.edges})
        return [{"g": g} for g in groups if g.startswith(params["p"])], None, None

    async def add_episode(
        self,
        name,
        episode_body,
        source_description,
        reference_time,
        source=None,
        group_id=None,
        uuid=None,
        **kw,
    ):
        self.rec(
            "add_episode",
            name=name,
            episode_body=episode_body,
            source_description=source_description,
            reference_time=reference_time,
            source=source,
            group_id=group_id,
            uuid=uuid,
            **kw,
        )
        if not any(e.uuid == uuid for e in self.episodes):  # same uuid = same episode
            self.episodes.append(
                SimpleNamespace(
                    uuid=uuid,
                    content=episode_body,
                    group_id=group_id,
                    created_at=reference_time,
                    valid_at=reference_time,
                )
            )
            self.edges.append(
                SimpleNamespace(
                    uuid=uuid,
                    fact=episode_body,
                    group_id=group_id,
                    name="MENTIONS",
                    created_at=datetime.now(UTC),
                    valid_at=reference_time,
                    invalid_at=None,
                )
            )
        return SimpleNamespace(episode=SimpleNamespace(uuid=uuid))

    async def search(self, query, center_node_uuid=None, group_ids=None, num_results=10, **kw):
        self.rec("search", query, group_ids=group_ids, num_results=num_results, **kw)
        hits = [e for e in self.edges if e.group_id in (group_ids or []) and overlap(query, e.fact)]
        hits.sort(key=lambda e: -overlap(query, e.fact))
        return hits[:num_results]

    async def retrieve_episodes(self, reference_time, last_n=3, group_ids=None, **kw):
        self.rec("retrieve_episodes", reference_time, last_n=last_n, group_ids=group_ids, **kw)
        return [e for e in self.episodes if e.group_id in (group_ids or [])][:last_n]


def graphiti_ops(g: StubGraphiti) -> SimpleNamespace:
    """Stand-ins for graphiti_core.edges.EntityEdge / nodes.EpisodicNode / EpisodeType."""

    class EntityEdge:
        @staticmethod
        async def get_by_uuid(driver, uuid):
            g.rec("EntityEdge.get_by_uuid", driver, uuid)
            for e in g.edges:
                if e.uuid == uuid:
                    return e
            raise LookupError(uuid)

        @staticmethod
        async def delete_by_uuids(driver, uuids):
            g.rec("EntityEdge.delete_by_uuids", driver, uuids)
            g.edges[:] = [e for e in g.edges if e.uuid not in uuids]

    class EpisodicNode:
        @staticmethod
        async def get_by_group_ids(driver, group_ids):
            g.rec("EpisodicNode.get_by_group_ids", driver, group_ids)
            return [e for e in g.episodes if e.group_id in group_ids]

        @staticmethod
        async def delete_by_group_id(driver, group_id):
            g.rec("EpisodicNode.delete_by_group_id", driver, group_id)
            g.episodes[:] = [e for e in g.episodes if e.group_id != group_id]
            g.edges[:] = [e for e in g.edges if e.group_id != group_id]

    class EpisodeType:
        message = "message"
        text = "text"

    return SimpleNamespace(
        EntityEdge=EntityEdge, EpisodicNode=EpisodicNode, EpisodeType=EpisodeType
    )


# ---- Supermemory (an httpx.MockTransport playing the HTTP API) -----------------------------------
class SupermemoryServer:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}  # customId -> document
        self.mems: dict[str, dict[str, Any]] = {}  # memory id -> memory
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle), base_url="https://sm.test"
        )

    def _remember(self, tag: str, text: str, metadata: dict, mid: str | None = None) -> dict:
        mid = mid or uuidlib.uuid4().hex
        mem = {
            "id": mid,
            "memory": text,
            "containerTag": tag,
            "metadata": metadata,
            "updatedAt": datetime.now(UTC).isoformat(),
            "createdAt": datetime.now(UTC).isoformat(),
        }
        self.mems[mid] = mem
        return mem

    def handle(self, request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content) if request.content else {}
        path, method = request.url.path, request.method
        self.requests.append((method, path, body))
        J = lambda data, status=200: httpx.Response(status, json=data)
        if (method, path) == ("POST", "/v3/documents"):
            custom = body.get("customId") or uuidlib.uuid4().hex
            for m in [m for m in self.mems.values() if m.get("customId") == custom]:
                del self.mems[m["id"]]  # same customId replaces the document and what it produced
            self.docs[custom] = {"id": custom, **body, "status": "done"}
            self._remember(body["containerTag"], body["content"], body.get("metadata", {}))[
                "customId"
            ] = custom
            return J({"id": custom, "status": "queued"})
        if method == "GET" and path.startswith("/v3/documents/"):
            return J({"id": path.rsplit("/", 1)[1], "status": "done"})
        if (method, path) == ("POST", "/v4/memories"):
            out = [
                self._remember(body["containerTag"], m["content"], m.get("metadata", {}))
                for m in body["memories"]
            ]
            return J({"documentId": None, "memories": out}, 201)
        if (method, path) == ("POST", "/v4/search"):
            q = body["q"]
            tags = body.get("containerTags") or [body["containerTag"]]
            hits = [
                {
                    "id": m["id"],
                    "memory": m["memory"],
                    "metadata": m["metadata"],
                    "updatedAt": m["updatedAt"],
                    "similarity": min(1.0, overlap(q, m["memory"]) / 5),
                }
                for m in self.mems.values()
                if m["containerTag"] in tags and overlap(q, m["memory"])
            ]
            hits.sort(key=lambda r: -r["similarity"])
            return J({"results": hits[: body["limit"]], "total": len(hits), "timing": 1})
        if (method, path) == ("PATCH", "/v4/memories"):
            m = self.mems.get(body["id"])
            if not m or m["containerTag"] != body["containerTag"]:
                return J({"error": "not found"}, 404)
            m["memory"] = body["newContent"]
            return J({"id": m["id"], "memory": m["memory"], "version": 2})
        if (method, path) == ("DELETE", "/v4/memories"):
            m = self.mems.get(body["id"])
            if not m or m["containerTag"] != body["containerTag"]:
                return J({"error": "not found"}, 404)
            del self.mems[body["id"]]
            return J({"id": body["id"], "forgotten": True})
        if (method, path) == ("POST", "/v4/memories/list"):
            tags = body["containerTags"]
            rows = [m for m in self.mems.values() if m["containerTag"] in tags]
            return J({"memoryEntries": rows[: int(body["limit"])], "pagination": {}})
        if (method, path) == ("POST", "/v4/profile"):
            facts = [
                m["memory"] for m in self.mems.values() if m["containerTag"] == body["containerTag"]
            ]
            return J({"profile": {"static": facts, "dynamic": []}})
        if (method, path) == ("GET", "/v3/container-tags/list"):
            return J({"containerTags": sorted({m["containerTag"] for m in self.mems.values()})})
        if method == "DELETE" and path.startswith("/v3/container-tags/"):
            tag = path.rsplit("/", 1)[1]
            gone = [i for i, m in self.mems.items() if m["containerTag"] == tag]
            if not gone:
                return J({"error": "not found"}, 404)
            for i in gone:
                del self.mems[i]
            return J(
                {
                    "success": True,
                    "containerTag": tag,
                    "deletedDocumentsCount": 0,
                    "deletedMemoriesCount": len(gone),
                }
            )
        return J({"error": f"unhandled {method} {path}"}, 500)


# ---- LangGraph BaseStore --------------------------------------------------------------------------
class StubStore(Recorder):
    """The slice of langgraph.store.base.BaseStore the adapter uses, with the same semantics:
    hierarchical tuple namespaces, prefix search, no embedding index (so `score` is None)."""

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[tuple[tuple[str, ...], str], SimpleNamespace] = {}

    async def aput(self, namespace, key, value, index=None, *, ttl=None):
        self.rec("aput", namespace, key, value)
        now = datetime.now(UTC)
        old = self.data.get((namespace, key))
        self.data[(namespace, key)] = SimpleNamespace(
            namespace=namespace,
            key=key,
            value=dict(value),
            created_at=old.created_at if old else now,
            updated_at=now,
            score=None,
        )

    async def asearch(
        self, namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None
    ):
        self.rec(
            "asearch", namespace_prefix, query=query, filter=filter, limit=limit, offset=offset
        )
        rows = [
            i for (ns, _), i in self.data.items() if ns[: len(namespace_prefix)] == namespace_prefix
        ]
        return rows[offset : offset + limit]

    async def adelete(self, namespace, key):
        self.rec("adelete", namespace, key)
        self.data.pop((namespace, key), None)
