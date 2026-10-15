"""The MCP server: the SDK-free core, and the real SDK client (and our own MCPClient) talking to it."""

import sys
from pathlib import Path

import pytest

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.mcp.server import MCPToolServer
from aaa.runtime import Budget, Engine
from aaa.tools import ToolRegistry
from helpers.aaa_mcp_server import registry

HELPER = Path(__file__).parent.parent / "helpers" / "aaa_mcp_server.py"


def err(r):
    return r.is_error if hasattr(r, "is_error") else r.isError


def names(server):
    return [t.name for t in server.list_tools()]


# -- the core --------------------------------------------------------------------------------------
async def test_tools_needing_approval_are_hidden_and_indistinguishable_from_missing():
    s = MCPToolServer(registry())
    assert names(s) == ["add", "boom", "refuse"]
    hidden = await s.call("wire_money", {"amount": 1})
    missing = await s.call("nonexistent", {"amount": 1})
    assert hidden.is_error and missing.is_error
    assert hidden.text.replace("wire_money", "X") == missing.text.replace("nonexistent", "X")


async def test_an_approval_callback_decides_and_fails_closed():
    seen = []

    async def policy(name, args, tenant):
        seen.append((name, args, tenant))
        return args["amount"] <= 100

    s = MCPToolServer(registry(), approval=policy, tenant="acme")
    assert "wire_money" in names(s)
    assert (await s.call("wire_money", {"amount": 50})).text == "sent 50"
    denied = await s.call("wire_money", {"amount": 500})
    assert denied.is_error and "not approved" in denied.text
    assert seen == [("wire_money", {"amount": 50}, "acme"), ("wire_money", {"amount": 500}, "acme")]

    async def broken(name, args, tenant):
        raise RuntimeError("policy service down")

    r = await MCPToolServer(registry(), approval=broken).call("wire_money", {"amount": 1})
    assert r.is_error


def test_the_two_ways_to_expose_approval_tools_do_not_mix():
    with pytest.raises(ValueError):
        MCPToolServer(registry(), approval=lambda *a: None, expose_approval_tools=True)  # type: ignore[arg-type,return-value]
    assert "wire_money" in names(MCPToolServer(registry(), expose_approval_tools=True))


async def test_failures_are_clean_and_never_leak_internals():
    s = MCPToolServer(registry())
    assert (await s.call("add", {"a": 1, "b": 2})).text == "3"
    bad = await s.call("add", {"a": "x", "b": 2})
    assert bad.is_error and "invalid arguments" in bad.text
    extra = await s.call("add", {"a": 1, "b": 2, "c": 3})
    assert extra.is_error
    assert (await s.call("refuse")).text == "not today"
    crash = await s.call("boom")
    assert crash.is_error and "hunter2" not in crash.text and crash.text == "boom failed"


async def test_results_are_capped():
    s = MCPToolServer(registry(), max_result_chars=5)
    assert "truncated" in (await s.call("add", {"a": 10**9, "b": 10**9})).text


async def test_tenant_and_user_come_from_the_host_not_the_arguments():
    from aaa.tools import tool

    @tool
    def whoami(ctx) -> str:
        """Report the caller."""
        return f"{ctx.tenant}/{ctx.meta.get('user')}"

    s = MCPToolServer(ToolRegistry([whoami]), tenant="local", user="dev")
    assert (await s.call("whoami")).text == "local/dev"
    assert (
        await s.call("whoami", tenant="acme", user="ann")
    ).text == "acme/ann"  # the authenticated host
    assert (await s.call("whoami", {"tenant": "evil"})).is_error  # unknown argument


async def test_run_agent_is_opt_in_budgeted_and_tenant_scoped():
    plain = MCPToolServer(registry())
    assert "run_agent" not in names(plain) and (await plain.call("run_agent")).is_error
    with pytest.raises(ValueError):
        MCPToolServer(registry(), algorithms={"x": object()})  # type: ignore[dict-item]

    reg = ToolRegistry([])
    async with Engine() as eng:
        LLMHandler(FakeLLMClient(lambda r: text("pong"))).attach(eng)
        algo = ToolLoop(reg)
        s = MCPToolServer(
            reg,
            engine=eng,
            algorithms={"echo": algo},
            tenant="acme",
            agent_budget=Budget(max_steps=5, deadline_s=10),
        )
        assert "run_agent" in names(s)
        ok = await s.call("run_agent", {"algo": "echo", "input": {"task": "ping"}})
        assert not ok.is_error and '"pong"' in ok.text
        for bad in (
            {"algo": "nope"},
            {"algo": "echo", "tenant": "x"},
            {"algo": "echo", "input": 5},
        ):
            assert (await s.call("run_agent", bad)).is_error
        runs = await eng.store.run_index()
        assert runs and all(r.started.tenant == "acme" for r in runs)
        assert runs[0].started.data["budget"]["max_steps"] == 5


# -- the real SDK talking to it ------------------------------------------------------------------------
mcp = pytest.importorskip("mcp")


async def test_the_official_client_lists_and_calls_our_tools_in_process():
    from aaa.mcp.server import build_server

    server = build_server(MCPToolServer(registry()))
    if hasattr(mcp, "Client"):
        async with mcp.Client(server) as c:
            listed = await c.list_tools()
            res = await c.call_tool("add", {"a": 2, "b": 3})
            bad = await c.call_tool("boom", {})
    else:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server) as c:
            listed = await c.list_tools()
            res = await c.call_tool("add", {"a": 2, "b": 3})
            bad = await c.call_tool("boom", {})
    assert sorted(t.name for t in listed.tools) == ["add", "boom", "refuse"]
    add = next(t for t in listed.tools if t.name == "add")
    schema = getattr(add, "input_schema", None) or add.inputSchema
    assert set(schema["properties"]) == {"a", "b"}
    assert res.content[0].text == "5" and not err(res)
    assert err(bad) and "hunter2" not in bad.content[0].text


async def test_our_own_client_against_a_real_stdio_subprocess():
    from aaa.mcp import MCPClient

    async with MCPClient.stdio(
        sys.executable,
        ["-m", "aaa.cli", "mcp", "serve", "helpers.aaa_mcp_server:build"],
        name="aaa",
        cwd=str(HELPER.parent.parent),
        env={"PYTHONPATH": f"{HELPER.parent.parent}:{HELPER.parent.parent.parent / 'src'}"},
    ) as client:
        listed = {t.name for t in await client.list_tools()}
        assert listed == {"add", "boom", "refuse"}
        out, is_error = await client.call("add", {"a": 20, "b": 22})
        assert out == "42" and not is_error
        tools = await client.load_tools()
        assert "aaa__add" in {t.name for t in tools}
