"""The MCP client against a REAL MCP server built with the official SDK (tests/helpers/mcp_server.py):
in-process, and as a stdio subprocess. Skipped when the `mcp` package is not installed; the CI
`adapters` job runs it on both major versions of the SDK."""

import asyncio
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.mcp import MCPClient, mcp_memory, mcp_tool_name
from aaa.memory import Forgettable, MemoryEntry, Scope
from aaa.runtime import Engine
from aaa.tools import SchemaTool, ToolHandler, ToolRegistry
from helpers.mcp_server import V2, build

SERVER = Path(__file__).parent.parent / "helpers" / "mcp_server.py"


@pytest.fixture
async def notes():
    server, calls = build()
    if V2:
        async with mcp.Client(server) as session:
            yield MCPClient(session, name="notes"), calls, server
    else:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server._mcp_server) as session:
            yield MCPClient(session, name="notes"), calls, server


async def engine_with(tools, script):
    eng = Engine()
    LLMHandler(FakeLLMClient(script)).attach(eng)
    reg = ToolRegistry(tools)
    ToolHandler(reg).attach(eng)
    return eng, ToolLoop(reg)


async def test_tools_are_discovered_with_their_schemas_and_annotations(notes):
    client, _, _ = notes
    infos = {t.name: t for t in await client.list_tools()}
    assert {"search_notes", "set_title", "delete_note", "explode", "big"} <= set(infos)
    s = infos["search_notes"]
    assert s.read_only is True and s.description == "Search the notes."
    assert s.parameters["properties"]["query"]["type"] == "string" and s.parameters["required"] == [
        "query"
    ]
    assert infos["set_title"].idempotent is True and infos["delete_note"].read_only is None
    assert infos["search_notes"].safe_to_repeat and not infos["delete_note"].safe_to_repeat
    assert infos["delete_note"].is_destructive and not infos["search_notes"].is_destructive


async def test_tools_become_aaa_tools_with_a_conservative_guard_policy(notes):
    client, _, _ = notes
    tools = {t.name: t for t in await client.load_tools()}
    assert all(isinstance(t, SchemaTool) for t in tools.values())
    assert set(tools) >= {"notes__search_notes", "notes__delete_note"}
    delivery = {n: t.guard.get("delivery") for n, t in tools.items()}
    assert delivery["notes__search_notes"] == "at_least_once"  # read-only: safe to repeat
    assert delivery["notes__set_title"] == "at_least_once"  # idempotent
    assert delivery["notes__delete_note"] == "at_most_once"  # no annotations: assume the worst
    distrust = {t.name: t.guard.get("delivery") for t in client.tools(trust_annotations=False)}
    assert set(distrust.values()) == {"at_most_once"}


async def test_include_exclude_prefix_and_name_safety(notes):
    client, _, _ = notes
    await client.list_tools()
    only = client.tools(include=["search_n*", "set_*"], exclude=["set_title"], prefix="n_")
    assert [t.name for t in only] == ["n_search_notes"]
    assert mcp_tool_name("srv__", "weird name!") == "srv__weird_name_"
    long = mcp_tool_name("p_", "x" * 200)
    assert (
        len(long) == 64
        and long == mcp_tool_name("p_", "x" * 200)
        and long != mcp_tool_name("p_", "x" * 199 + "y")
    )


async def test_approval_policy_follows_the_annotations(notes):
    client, _, _ = notes
    await client.list_tools()
    need = {t.name: t.requires_approval for t in client.tools(approval="destructive")}
    assert need["notes__search_notes"] is False  # read-only
    assert need["notes__delete_note"] is True and need["notes__explode"] is True
    assert all(t.requires_approval for t in client.tools(approval="always"))
    assert not any(t.requires_approval for t in client.tools())


async def test_a_model_uses_an_mcp_tool_through_the_engine(notes):
    client, calls, _ = notes

    def script(req):
        last = req.messages[-1]
        if last.role == "user":
            return tool_call("notes__search_notes", query="roadmap", limit=2)
        return text(f"found: {last.content}")

    eng, algo = await engine_with(await client.load_tools(), script)
    async with eng:
        out = await eng.run(algo, {"task": "find roadmap notes"})
    assert out.ok and "note about roadmap #0" in out.value["text"]
    assert calls == [("search_notes", {"query": "roadmap", "limit": 2})]


async def test_bad_arguments_never_reach_the_server(notes):
    client, calls, _ = notes

    def script(req):
        if req.messages[-1].role == "user":
            return tool_call("notes__search_notes", query="x", limit="many")  # wrong type
        return text(req.messages[-1].content or "")

    eng, algo = await engine_with(await client.load_tools(), script)
    async with eng:
        h = await eng.start(algo, {"task": "go"})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    failed = next(e for e in log if e.type == "aaa.tool.failed")
    assert (
        failed.data["error"]["type"] == "InvalidArguments"
        and "limit" in failed.data["error"]["message"]
    )
    assert out.ok and calls == []


async def test_a_server_side_error_is_a_clean_tool_failure(notes):
    client, _, _ = notes

    def script(req):
        if req.messages[-1].role == "user":
            return tool_call("notes__explode", reason="because")
        return text(req.messages[-1].content or "")

    eng, algo = await engine_with(await client.load_tools(), script)
    async with eng:
        h = await eng.start(algo, {"task": "go"})
        await h.result()
        log = await eng.store.read(h.run_id)
    failed = next(e for e in log if e.type == "aaa.tool.failed")
    # mcp 1.x returns the exception text; 2.x deliberately hides internals ("Error executing tool explode")
    assert any(w in failed.data["error"]["message"] for w in ("boom: because", "explode"))


async def test_a_destructive_tool_waits_for_human_approval(notes):
    client, calls, _ = notes

    def script(req):
        if req.messages[-1].role == "user":
            return tool_call("notes__delete_note", note_id="n1")
        return text("done")

    eng, algo = await engine_with(await client.load_tools(approval="destructive"), script)
    async with eng:
        h = await eng.start(algo, {"task": "delete n1"})
        req = None
        for _ in range(200):
            req = next(
                (e for e in await eng.store.read(h.run_id) if e.type == "aaa.approval.requested"),
                None,
            )
            if req:
                break
            await asyncio.sleep(0.02)
        assert req is not None and calls == []  # nothing ran yet
        await h.signal("aaa.approval.granted", {"call_id": req.data["call_id"]})
        assert (await h.result()).ok
    assert calls == [("delete_note", {"note_id": "n1"})]


async def test_long_results_are_truncated_to_protect_the_context(notes):
    client, _, _ = notes
    await client.list_tools()
    big = next(t for t in client.tools(max_result_chars=100) if t.name == "notes__big")
    out = await big.invoke({"n": 5000})
    assert len(out) < 200 and "truncated 4900 characters" in out
    unlimited = next(t for t in client.tools(max_result_chars=None) if t.name == "notes__big")
    assert len(await unlimited.invoke({"n": 5000})) == 5000


async def test_resources_can_be_listed_and_read(notes):
    client, _, _ = notes
    assert any(r["uri"] == "note://welcome" for r in await client.list_resources())
    assert await client.read_resource("note://welcome") == "Welcome to the notes server"


async def test_an_mcp_memory_server_works_as_an_aaa_memory(notes):
    client, _, server = notes
    memory = mcp_memory(
        client, add="add_memory", search="search_memory", forget="delete_all_memories"
    )
    ann, bob = Scope(tenant="acme", user="ann"), Scope(tenant="acme", user="bob")
    await memory.add([MemoryEntry(text="Ann is vegetarian")], ann)
    await memory.add([MemoryEntry(text="Bob eats fish")], bob)
    assert [m.text for m in await memory.search("vegetarian food", ann)] == ["Ann is vegetarian"]
    assert await memory.search("vegetarian", bob) == []  # the server honoured the scope argument
    assert isinstance(memory, Forgettable)
    await memory.forget(ann)
    assert await memory.search("vegetarian", ann) == [] and ann.slug() not in server.memories
    assert await memory.search("fish", bob)


async def test_a_real_stdio_server_subprocess_end_to_end():
    async with MCPClient.stdio(sys.executable, [str(SERVER)], name="notes") as client:
        names = {t.name for t in await client.list_tools()}
        assert "search_notes" in names
        text_, _ = await client.call("search_notes", {"query": "x", "limit": 1})
        assert "note about x #0" in text_
        tools = await client.load_tools()
        assert "notes__search_notes" in {t.name for t in tools}
    with pytest.raises(RuntimeError, match="not connected"):
        client.session  # noqa: B018 - closed on exit


@pytest.fixture
async def http_server():
    """The real server over Streamable HTTP (uvicorn on a free port); records request headers."""
    import socket

    uvicorn = pytest.importorskip("uvicorn")
    server, calls = build()
    app = server.streamable_http_app()
    seen: list[dict[str, str]] = []

    async def spy(scope, receive, send):
        if scope["type"] == "http":
            seen.append({k.decode(): v.decode() for k, v in scope["headers"]})
        await app(scope, receive, send)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(spy, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    task = asyncio.create_task(srv.serve())
    for _ in range(200):
        if srv.started:
            break
        await asyncio.sleep(0.02)
    assert srv.started, "the test HTTP server did not start"
    try:
        yield f"http://127.0.0.1:{port}/mcp", calls, seen
    finally:
        srv.should_exit = True
        await asyncio.wait_for(task, 10)


@pytest.mark.parametrize("with_headers", [False, True])
async def test_a_real_streamable_http_server_end_to_end(http_server, with_headers):
    url, calls, seen = http_server
    headers = {"Authorization": "Bearer sk-test-123"} if with_headers else None
    async with MCPClient.http(url, headers=headers, name="notes") as client:
        tools = await client.load_tools()
        assert "notes__search_notes" in {t.name for t in tools}
        search = next(t for t in tools if t.name == "notes__search_notes")
        assert "note about hi #0" in await search.invoke({"query": "hi", "limit": 1})
    assert calls == [("search_notes", {"query": "hi", "limit": 1})]
    if with_headers:
        assert any(h.get("authorization") == "Bearer sk-test-123" for h in seen)
