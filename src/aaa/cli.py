"""`aaa`: read-only commands for looking at a durable event store from a shell.

    aaa store check sqlite:///agent.db     open the store (bringing an old layout forward) and report on it
    aaa store verify sqlite:///agent.db    check every run's log is consistent (use after a restore; --blobs DIR [--deep])
    aaa runs sqlite:///agent.db            one line per run
    aaa events sqlite:///agent.db RUN_ID   the run's events, in order
    aaa mcp serve mymodule:build           serve a ToolRegistry / MCPToolServer over MCP stdio (extra `mcp`)

Store URLs: `sqlite:///path.db` (or a bare `*.db` path), `postgresql://...` (extra `postgres`) and `redis://...` (extra `redis`). These commands never write events."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from aaa.events import EventStore, SQLiteEventStore
from aaa.events.migrate import SCHEMA_VERSION, SchemaTooNew


async def open_store(url: str) -> EventStore:
    if url.startswith(("postgres://", "postgresql://")):
        from aaa.events.postgres import PostgresEventStore

        return await PostgresEventStore.create(url)
    if url.startswith(("redis://", "rediss://")):
        from aaa.events.redis import RedisEventStore

        return await RedisEventStore.create(url)
    path = url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else url
    if not (url.startswith("sqlite:///") or path.endswith((".db", ".sqlite", ".sqlite3"))):
        raise SystemExit(
            f"unrecognised store {url!r}: use sqlite:///file.db, postgresql://... or redis://..."
        )
    return SQLiteEventStore(path)


def _clip(value: Any, width: int = 100) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= width else text[: width - 1] + "…"


async def _check(url: str) -> int:
    store = await open_store(url)
    try:
        runs = await store.run_index()
        open_runs = sum(r.completed is None for r in runs)
        print(
            f"ok: opened {type(store).__name__}; this release writes layout version {SCHEMA_VERSION}"
        )
        print(f"runs: {len(runs)} ({open_runs} open, {len(runs) - open_runs} finished)")
        tenants = sorted({r.started.tenant or "-" for r in runs})
        print(f"tenants: {', '.join(tenants) if tenants else '(none)'}")
        return 0
    finally:
        await store.aclose()


async def _verify(url: str, blobs: str | None, deep: bool) -> int:
    from aaa.data.blob import FileBlobStore
    from aaa.events.verify import verify_store

    store = await open_store(url)
    try:
        rep = await verify_store(store, FileBlobStore(blobs) if blobs else None, deep=deep)
    finally:
        await store.aclose()
    print(
        f"runs: {rep.runs} ({rep.open_runs} open); events: {rep.events}; blobs checked: {rep.blobs_checked}"
    )
    print(
        f"newest event: {rep.newest_event:%Y-%m-%dT%H:%M:%SZ}"
        if rep.newest_event
        else "newest event: (none)"
    )
    for p in rep.problems:
        print(f"PROBLEM: {p}")
    print("ok" if rep.ok else f"{len(rep.problems)} problem(s) found")
    return 0 if rep.ok else 1


async def _runs(url: str, open_only: bool) -> int:
    store = await open_store(url)
    try:
        for r in await store.run_index():
            if open_only and r.completed is not None:
                continue
            kind = r.completed.data.get("kind") if r.completed else "open"
            print(
                f"{r.stream}\t{r.started.data.get('algo')}\t{r.started.tenant or '-'}\t{kind}\t{r.started.time:%Y-%m-%dT%H:%M:%SZ}"
            )
        return 0
    finally:
        await store.aclose()


async def _events(url: str, run_id: str, show_data: bool, as_json: bool) -> int:
    store = await open_store(url)
    try:
        events = await store.read(run_id)
        if not events:
            print(f"no run {run_id!r}", file=sys.stderr)
            return 1
        for e in events:
            if as_json:
                print(e.model_dump_json())
            else:
                tail = f"  {_clip(e.data)}" if show_data else ""
                print(f"{e.seq}\t{e.time:%H:%M:%S.%f}\t{e.type}{tail}")
        return 0
    finally:
        await store.aclose()


async def _mcp_serve(spec: str) -> int:
    import importlib
    import inspect

    from aaa.mcp.server import MCPToolServer, serve_stdio
    from aaa.tools import ToolRegistry

    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise SystemExit("expected module:factory, e.g. myapp.tools:build")
    sys.path.insert(0, "")
    obj = getattr(importlib.import_module(module), attr)
    obj = obj() if callable(obj) else obj
    if inspect.isawaitable(obj):
        obj = await obj
    if isinstance(obj, ToolRegistry):
        obj = MCPToolServer(obj)
    if not isinstance(obj, MCPToolServer):
        raise SystemExit(f"{spec} must return a ToolRegistry or MCPToolServer")
    await serve_stdio(obj)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aaa", description="Inspect an AAA event store.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("store", help="store maintenance").add_subparsers(dest="sub", required=True)
    st.add_parser("check", help="open the store and report").add_argument("url")
    vf = st.add_parser("verify", help="check every run's log is consistent (after a restore)")
    vf.add_argument("url")
    vf.add_argument("--blobs", help="a FileBlobStore directory whose blobs the events refer to")
    vf.add_argument("--deep", action="store_true", help="also re-hash every blob")
    rn = sub.add_parser("runs", help="list runs")
    rn.add_argument("url")
    rn.add_argument("--open", action="store_true", help="only runs that have not finished")
    ev = sub.add_parser("events", help="print one run's events")
    ev.add_argument("url")
    ev.add_argument("run_id")
    ev.add_argument("--data", action="store_true", help="show (clipped) event payloads")
    ev.add_argument("--json", action="store_true", help="one JSON event per line")
    mc = sub.add_parser("mcp", help="MCP server").add_subparsers(dest="sub", required=True)
    sv = mc.add_parser("serve", help="serve tools over MCP stdio")
    sv.add_argument("factory", help="module:function returning a ToolRegistry or MCPToolServer")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "mcp":
            return asyncio.run(_mcp_serve(a.factory))
        if a.cmd == "store":
            if a.sub == "verify":
                return asyncio.run(_verify(a.url, a.blobs, a.deep))
            return asyncio.run(_check(a.url))
        if a.cmd == "runs":
            return asyncio.run(_runs(a.url, a.open))
        return asyncio.run(_events(a.url, a.run_id, a.data, a.json))
    except SchemaTooNew as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
