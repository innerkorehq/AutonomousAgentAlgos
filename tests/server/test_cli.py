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
