"""The `aaa` store-inspection CLI and `aaa-server` factory loading."""

from __future__ import annotations

import asyncio
import sys

import pytest
from aaa_server import ServerConfig
from aaa_server.auth import ApiKeyAuthenticator
from aaa_server.cli import load_config

from aaa.algos import ToolLoop
from aaa.cli import main
from aaa.events import SQLiteEventStore
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolRegistry


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "agent.db"

    async def seed():
        store = SQLiteEventStore(path)
        async with Engine(store=store) as eng:
            LLMHandler(FakeLLMClient(lambda r: text("hi"))).attach(eng)
            await eng.tenant("acme").run(ToolLoop(ToolRegistry([])), {"task": "x"}, run_id="r1")

    asyncio.run(seed())
    return path


def test_store_check_runs_events(db, capsys):
    assert main(["store", "check", f"sqlite:///{db}"]) == 0
    out = capsys.readouterr().out
    assert "runs: 1 (0 open, 1 finished)" in out and "tenants: acme" in out
    assert main(["runs", str(db)]) == 0
    assert "acme/r1" in capsys.readouterr().out
    assert main(["events", str(db), "acme/r1", "--data"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split("\t")[2].startswith("aaa.run.started")
    assert main(["events", str(db), "acme/nope"]) == 1


def test_unknown_store_is_refused(capsys):
    with pytest.raises(SystemExit):
        main(["runs", "ftp://nope"])


def test_server_factory_loading(tmp_path, monkeypatch):
    (tmp_path / "myserver.py").write_text(
        "from aaa.runtime import Engine\n"
        "from aaa_server import ServerConfig\n"
        "from aaa_server.auth import ApiKeyAuthenticator\n"
        "def build():\n"
        "    return ServerConfig(engine=Engine(), authenticator=ApiKeyAuthenticator.from_keys({}))\n"
        "def wrong():\n    return 5\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert isinstance(load_config("myserver:build"), ServerConfig)
    with pytest.raises(SystemExit):
        load_config("myserver:wrong")
    with pytest.raises(SystemExit):
        load_config("nonsense")
    sys.modules.pop("myserver", None)
    _ = ApiKeyAuthenticator


def test_store_verify_finds_a_gap_a_missing_start_and_a_missing_blob(db, capsys):
    import sqlite3

    assert main(["store", "verify", str(db)]) == 0
    assert "ok" in capsys.readouterr().out
    c = sqlite3.connect(db)
    c.execute("DELETE FROM events WHERE stream='acme/r1' AND seq=1")
    c.commit()
    c.close()
    assert main(["store", "verify", str(db)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: acme/r1: sequence numbers" in out and "problem(s) found" in out


async def test_verify_checks_blobs(tmp_path):
    from aaa.data.blob import FileBlobStore
    from aaa.events import Event, InMemoryEventStore
    from aaa.events.verify import verify_store

    blobs = FileBlobStore(tmp_path)
    ref = await blobs.put(b"payload")
    store = InMemoryEventStore()
    await store.append(
        "r",
        [
            Event(
                id="r:started",
                type="aaa.run.started",
                stream="r",
                run_id="r",
                data={},
                blobs=(ref,),
            )
        ],
    )
    assert (await verify_store(store, blobs, deep=True)).ok
    await blobs.delete(ref)
    rep = await verify_store(store, blobs)
    assert not rep.ok and "missing" in rep.problems[0] and rep.blobs_checked == 1


async def test_an_async_factory_is_built_inside_the_servers_own_event_loop(tmp_path, monkeypatch):
    """A connection pool opened by the factory must belong to the loop that serves requests (a pool made in a
    throwaway loop fails with 'Event loop is closed' on the first request)."""
    import asyncio

    import httpx

    (tmp_path / "loopserver.py").write_text(
        "import asyncio\n"
        "from aaa.runtime import Engine\n"
        "from aaa_server import ServerConfig\n"
        "from aaa_server.auth import ApiKeyAuthenticator\n"
        "LOOPS = []\n"
        "async def build():\n"
        "    LOOPS.append(asyncio.get_running_loop())\n"
        "    return ServerConfig(engine=Engine(), authenticator=ApiKeyAuthenticator.from_keys({'k': ('acme', 'ann')}))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from aaa_server.cli import LazyApp, load_factory

    mod = __import__("loopserver")
    app = LazyApp(load_factory("loopserver:build"))
    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    await inbox.put({"type": "lifespan.startup"})
    life = asyncio.create_task(app({"type": "lifespan"}, inbox.get, outbox.put))
    assert (await outbox.get())["type"] == "lifespan.startup.complete"
    assert mod.LOOPS == [asyncio.get_running_loop()]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/healthz")).json() == {"ok": True}
        assert (await c.get("/v1/runs", headers={"Authorization": "Bearer k"})).status_code == 200
    await inbox.put({"type": "lifespan.shutdown"})
    assert (await outbox.get())["type"] == "lifespan.shutdown.complete"
    await life
    sys.modules.pop("loopserver", None)
