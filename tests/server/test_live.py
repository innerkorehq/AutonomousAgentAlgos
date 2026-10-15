"""Over a real socket (uvicorn): incremental SSE, keep-alives, resume, and cleanup when a client walks away."""

from __future__ import annotations

import asyncio
import socket

import httpx
import uvicorn
from aaa_server import create_app

from helpers.server_env import make_env


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            return server, task
        await asyncio.sleep(0.02)
    raise AssertionError("server did not start")


async def test_stream_is_incremental_keeps_alive_and_frees_its_slot_on_disconnect():
    env, engine = await make_env(max_streams=1)
    port = free_port()
    async with engine:
        server, task = await serve(create_app(env.cfg), port)
        try:
            base = f"http://127.0.0.1:{port}"
            h = {"Authorization": "Bearer key-acme-approver"}
            async with httpx.AsyncClient(base_url=base, headers=h, timeout=10) as c:
                rid = (
                    await c.post("/v1/runs", json={"algo": "agent", "input": {"task": "wire it"}})
                ).json()["run_id"]  # parks on approval, so the stream stays open
                seen = []
                async with c.stream("GET", f"/v1/runs/{rid}/events") as r:
                    assert r.status_code == 200
                    async for line in r.aiter_lines():
                        seen.append(line)
                        if any(x.startswith(": keepalive") for x in seen):
                            break  # arrives while the run is still open: the response really streams
                assert any("approval.requested" in x for x in seen)
                # the single stream slot is busy while a second client holds it open ...
                async with c.stream("GET", f"/v1/runs/{rid}/events") as first:
                    assert first.status_code == 200
                    second = await c.get(f"/v1/runs/{rid}/events")
                    assert second.status_code == 429
                # ... and is free again once that client disconnects
                status = 429
                for _ in range(100):
                    async with c.stream("GET", f"/v1/runs/{rid}/events") as probe:
                        status = probe.status_code
                    if status != 429:
                        break
                    await asyncio.sleep(0.05)
                assert status == 200
                await c.post(f"/v1/runs/{rid}/cancel")
        finally:
            server.should_exit = True
            await task
