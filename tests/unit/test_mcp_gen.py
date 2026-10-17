"""The MCP server generator: spec, configuration loading, filtering, compilation, wire shapes, CLI and the serializers."""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from aaa.mcp import wire
from aaa.mcp.compile import acompile_spec, compile_spec, describe
from aaa.mcp.config import Secret, SpecError, deep_merge, env_overlay, load_spec
from aaa.mcp.spec import MCPSpec
from aaa.tools import ToolRegistry, tool

HELPERS = Path(__file__).parent.parent / "helpers"
OBJ = "helpers.aaa_mcp_server:rich_registry"


def spec_of(**over):
    base = {"sources": [{"type": "objects", "ref": OBJ}]}
    return load_spec(deep_merge(base, over), environ={})


async def names(**over):
    core = await acompile_spec(spec_of(**over))
    return sorted(t.name for t in core.list_tools()), core


# ===================================================================== spec and configuration
def test_unknown_keys_and_bad_values_name_the_field():
    with pytest.raises(SpecError, match=r"tools.exclud: Extra inputs are not permitted"):
        load_spec({"tools": {"exclud": ["x"]}}, environ={})
    with pytest.raises(SpecError, match=r"sources.0.objects.ref: String should match"):
        load_spec({"sources": [{"type": "objects", "ref": "no-colon"}]}, environ={})
    with pytest.raises(SpecError, match="tools.rename"):
        load_spec({"tools": {"rename": {"a": "bad name!"}}}, environ={})
    with pytest.raises(SpecError, match="limits.max_tools"):
        load_spec({"limits": {"max_tools": 0}}, environ={})


def test_json_and_toml_files_load_and_unknown_formats_are_refused(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"server": {"name": "from-json"}}))
    (tmp_path / "s.toml").write_text('[server]\nname = "from-toml"\n[tools]\nexclude = ["a_*"]\n')
    (tmp_path / "s.ini").write_text("x")
    assert load_spec(tmp_path / "s.json", environ={}).server.name == "from-json"
    t = load_spec(tmp_path / "s.toml", environ={})
    assert t.server.name == "from-toml" and t.tools.exclude == ["a_*"]
    with pytest.raises(SpecError, match="unknown configuration format"):
        load_spec(tmp_path / "s.ini", environ={})
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(SpecError, match="not valid JSON"):
        load_spec(tmp_path / "bad.json", environ={})
    with pytest.raises(SpecError, match="cannot read"):
        load_spec(tmp_path / "missing.json", environ={})


def test_yaml_is_read_with_safe_load_only(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "s.yaml").write_text("server:\n  name: from-yaml\ntools:\n  include: ['a*']\n")
    assert load_spec(tmp_path / "s.yaml", environ={}).tools.include == ["a*"]
    (tmp_path / "evil.yaml").write_text("server: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(SpecError, match="not valid YAML"):
        load_spec(tmp_path / "evil.yaml", environ={})


def test_precedence_is_file_then_environment_then_flags(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(
        json.dumps({"server": {"name": "file", "title": "File"}, "tools": {"exclude": ["a"]}})
    )
    env = {
        "AAA_MCP__SERVER__NAME": "env",
        "AAA_MCP__TOOLS__EXCLUDE": '["b","c"]',
        "AAA_MCP__LIMITS__MAX_TOOLS": "5",
        "OTHER": "x",
    }
    s = load_spec(f, environ=env)
    assert (s.server.name, s.server.title, s.tools.exclude, s.limits.max_tools) == (
        "env",
        "File",
        ["b", "c"],
        5,
    )
    s = load_spec(f, environ=env, overlay={"server": {"name": "flag"}, "tools": {"exclude": ["z"]}})
    assert (s.server.name, s.server.title, s.tools.exclude) == (
        "flag",
        "File",
        ["z"],
    )  # lists are replaced, not merged
    assert env_overlay({"AAA_MCP__A__B": "true", "AAA_MCP__C": "1.5", "AAA_MCP__D": "text"}) == {
        "a": {"b": True},
        "c": 1.5,
        "d": "text",
    }
    with pytest.raises(SpecError, match="conflicts"):
        env_overlay({"AAA_MCP__A": "1", "AAA_MCP__A__B": "2"})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_a_config_that_imports_code_must_not_be_world_writable(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"sources": [{"type": "objects", "ref": OBJ}]}))
    f.chmod(0o666)
    with pytest.raises(SpecError, match="writable by everyone"):
        load_spec(f, environ={})
    f.chmod(0o644)
    load_spec(f, environ={})
    g = tmp_path / "no-code.json"
    g.write_text("{}")
    g.chmod(0o666)
    load_spec(g, environ={})  # nothing imported, nothing to protect
    assert stat.S_IMODE(g.stat().st_mode) == 0o666


def test_secrets_are_references_never_values():
    s = Secret.parse("env:API_TOKEN")
    assert Secret.parse("${API_TOKEN}").name == "API_TOKEN" and "API_TOKEN" in repr(s)
    assert s.value({"API_TOKEN": "hunter2-value"}) == "hunter2-value" and "hunter2" not in repr(
        s
    ) + str(s)
    with pytest.raises(SpecError, match="never the value"):
        Secret.parse("sk-live-abc123", field="auth.token")
    with pytest.raises(SpecError, match="API_TOKEN is not set"):
        s.value({})


# ===================================================================== compile and filter
async def test_every_tool_is_exposed_by_default_but_approval_tools_stay_hidden():
    exposed, core = await names()
    assert exposed == ["add", "broken_shape", "locate", "shout"]
    assert core.hidden == ["delete_all"] and any("hidden" in line for line in core.report)
    assert (await core.call("delete_all")).is_error  # indistinguishable from an unknown tool
    exposed, core = await names(tools={"expose_approval_tools": True})
    assert "delete_all" in exposed and core.hidden == []


async def test_include_exclude_tags_prefix_rename_and_overrides():
    assert (await names(tools={"include": ["s*", "add"], "exclude": ["shout"]}))[0] == ["add"]
    assert (
        await names(
            sources=[{"type": "objects", "ref": OBJ, "tags": ["geo"]}], tools={"tags": ["geo"]}
        )
    )[0] != []
    assert (await names(tools={"tags": ["nothing"]}))[0] == []
    got, _ = await names(tools={"rename": {"add": "plus"}, "prefix": "t_", "exclude": ["b*"]})
    assert got == ["t_locate", "t_plus", "t_shout"]
    got, core = await names(
        sources=[{"type": "objects", "ref": OBJ, "prefix": "geo_"}],
        tools={
            "include": ["geo_loc*"],
            "overrides": {
                "geo_locate": {
                    "title": "Locate",
                    "description": "Where is it?",
                    "annotations": {"read_only": True},
                }
            },
        },
    )
    (t,) = core.list_tools()
    assert (t.name, t.title, t.description) == ("geo_locate", "Locate", "Where is it?")
    assert t.annotations and t.annotations.read_only is True and t.read_only is True


async def test_mistakes_in_the_policy_are_loud():
    with pytest.raises(SpecError, match="tools.rename: no selected tool is named"):
        await acompile_spec(spec_of(tools={"rename": {"nope": "x"}}))
    with pytest.raises(SpecError, match="tools.overrides: no exposed tool is named"):
        await acompile_spec(spec_of(tools={"overrides": {"nope": {"title": "x"}}}))
    with pytest.raises(SpecError, match="max_tools"):
        await acompile_spec(spec_of(limits={"max_tools": 2}))
    with pytest.raises(SpecError, match=r"two tools are both called 'add'.*objects"):
        await acompile_spec(
            spec_of(sources=[{"type": "objects", "ref": OBJ}, {"type": "objects", "ref": OBJ}])
        )
    with pytest.raises(SpecError, match="cannot import"):
        await acompile_spec(spec_of(sources=[{"type": "objects", "ref": "nonexistent_pkg:x"}]))
    with pytest.raises(SpecError, match="has no attribute"):
        await acompile_spec(
            spec_of(sources=[{"type": "objects", "ref": "helpers.aaa_mcp_server:nope"}])
        )
    with pytest.raises(SpecError, match="expected a ToolRegistry"):
        await acompile_spec(
            spec_of(sources=[{"type": "objects", "ref": "helpers.aaa_mcp_server:Point"}])
        )


async def test_an_override_can_lift_the_approval_requirement_explicitly():
    got, core = await names(tools={"overrides": {"delete_all": {"requires_approval": False}}})
    assert "delete_all" in got and (await core.call("delete_all")).text == "gone"


async def test_the_import_kill_switch_stops_everything_before_any_import(monkeypatch):
    monkeypatch.setenv("AAA_MCP_ALLOW_IMPORT", "0")
    with pytest.raises(SpecError, match="importing is disabled"):
        await acompile_spec(spec_of(sources=[{"type": "objects", "ref": "side_effect_module:x"}]))
    assert "side_effect_module" not in sys.modules
    monkeypatch.delenv("AAA_MCP_ALLOW_IMPORT")
    with pytest.raises(SpecError, match="importing is disabled"):
        await acompile_spec(spec_of(), allow_import=False)


async def test_live_objects_can_be_passed_from_python():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    spec = MCPSpec.model_validate({"sources": [{"type": "live", "key": "mine"}]})
    core = await acompile_spec(spec, live={"mine": ToolRegistry([ping])})
    assert (await core.call("ping")).text == "pong"
    with pytest.raises(SpecError, match="no live object"):
        await acompile_spec(spec)


def test_compile_spec_works_outside_a_loop():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    spec = MCPSpec.model_validate({"sources": [{"type": "live", "key": "mine"}]})
    assert compile_spec(spec, live={"mine": [ping]}).list_tools()[0].name == "ping"


async def test_compile_spec_refuses_to_run_inside_an_event_loop():
    with pytest.raises(RuntimeError, match="acompile_spec"):
        compile_spec(spec_of())


# ===================================================================== tool metadata and structured output
async def test_hints_come_from_what_a_tool_already_says():
    _, core = await names(tools={"expose_approval_tools": True})
    by = {t.name: t for t in core.list_tools()}
    assert by["delete_all"].annotations and by["delete_all"].annotations.destructive is True
    assert by["locate"].annotations and by["locate"].annotations.idempotent is True
    assert by["shout"].annotations is None  # nothing is claimed that the tool did not say
    assert (
        by["locate"].output_schema
        and by["locate"].output_schema["properties"]["x"]["type"] == "number"
    )
    assert by["shout"].output_schema is None


async def test_structured_output_is_sent_validated_and_mirrored_as_text():
    _, core = await names()
    ok = await core.call("locate", {"name": "home"})
    assert ok.structured == {"x": 1.5, "y": 2.5} and json.loads(ok.text) == ok.structured
    bad = await core.call("broken_shape")
    assert (
        bad.is_error and "does not match its output schema" in bad.text and bad.structured is None
    )
    _, off = await names(limits={"structured_output": False})
    assert (await off.call("locate", {"name": "home"})).structured is None
    assert not any(t.output_schema for t in off.list_tools())


async def test_structured_output_is_never_sent_cut_short():
    _, core = await names(limits={"max_result_chars": 1000})
    core.max_result_chars = 10
    r = await core.call("locate", {"name": "home"})
    assert r.structured is None and "truncated" in r.text


# ===================================================================== wire shapes
async def test_wire_shapes():
    _, core = await names(
        server={"title": "Places", "instructions": "Be careful.", "version": "2.1"},
        tools={"overrides": {"shout": {"annotations": {"title": "Shouter", "open_world": False}}}},
    )
    t = {x["name"]: x for x in wire.tools_list_result(core)["tools"]}
    assert t["locate"]["outputSchema"]["type"] == "object" and t["locate"]["annotations"] == {
        "idempotentHint": True
    }
    assert (
        t["shout"]["annotations"] == {"title": "Shouter", "openWorldHint": False}
        and t["shout"]["title"] == "Shouter"
    )
    assert "outputSchema" not in t["shout"] and "annotations" not in t["add"]
    init = wire.initialize_result(core, "2025-06-18")
    assert (
        init["serverInfo"] == {"name": "aaa", "version": "2.1", "title": "Places"}
        and init["instructions"] == "Be careful."
    )
    assert init["capabilities"] == {"tools": {"listChanged": False}}
    ok = wire.call_result_to_wire(await core.call("locate", {"name": "x"}))
    assert ok["structuredContent"] == {"x": 1.5, "y": 2.5} and ok["isError"] is False
    err = wire.call_result_to_wire(await core.call("nope"))
    assert err["isError"] is True and "structuredContent" not in err


async def test_describe_shows_what_the_model_will_see_and_what_is_hidden():
    _, core = await names()
    d = describe(core)
    assert d["hidden"] == ["delete_all"] and {t["name"] for t in d["tools"]} == {
        "add",
        "broken_shape",
        "locate",
        "shout",
    }
    assert d["server"]["name"] == "aaa"


# ===================================================================== serializers agree
mcp = pytest.importorskip("mcp")


def normalized(tool_wire):
    keep = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")
    return {k: tool_wire[k] for k in keep if tool_wire.get(k) is not None}


async def sdk_session(core):
    from aaa.mcp.server import build_server

    server = build_server(core)
    if hasattr(mcp, "Client"):
        return mcp.Client(server)
    from mcp.shared.memory import create_connected_server_and_client_session

    return create_connected_server_and_client_session(server)


async def test_the_sdk_server_and_the_http_handler_describe_tools_identically():
    from aaa_server.mcp_http import handle

    _, core = await names(
        server={"title": "Places", "instructions": "Hi"},
        tools={
            "expose_approval_tools": True,
            "overrides": {"shout": {"title": "Shout", "annotations": {"read_only": True}}},
        },
    )
    http = (
        await handle(
            core, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, tenant="t", user="u"
        )
    )["result"]["tools"]
    async with await sdk_session(core) as c:
        listed = await c.list_tools()
        sdk = [
            normalized(t.model_dump(by_alias=True, exclude_none=True, mode="json"))
            for t in listed.tools
        ]
        ok = await c.call_tool("locate", {"name": "home"})
        bad = await c.call_tool("broken_shape", {})
    assert sorted(sdk, key=lambda t: t["name"]) == sorted(
        (normalized(t) for t in http), key=lambda t: t["name"]
    )
    sc = getattr(ok, "structured_content", None) or getattr(ok, "structuredContent", None)
    assert sc == {"x": 1.5, "y": 2.5}
    http_ok = (
        await handle(
            core,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "locate", "arguments": {"name": "h"}},
            },
            tenant="t",
            user="u",
        )
    )["result"]
    assert http_ok["structuredContent"] == sc
    assert (getattr(bad, "is_error", None) if hasattr(bad, "is_error") else bad.isError) is True


# ===================================================================== command line
def run_cli(*args, env=None):
    import subprocess

    e = {
        **os.environ,
        "PYTHONPATH": f"{HELPERS.parent}:{HELPERS.parent.parent / 'src'}",
        **(env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "aaa.cli", "mcp", *args],
        check=False,
        capture_output=True,
        text=True,
        env=e,
        cwd=HELPERS.parent,
        timeout=60,
    )


def test_cli_check_inspect_and_errors(tmp_path):
    cfg = tmp_path / "s.json"
    cfg.write_text(
        json.dumps({"sources": [{"type": "objects", "ref": OBJ}], "tools": {"exclude": ["b*"]}})
    )
    cfg.chmod(0o644)
    r = run_cli("check", "--config", str(cfg))
    assert r.returncode == 0 and "3 tool(s) exposed" in r.stdout and "delete_all" in r.stdout
    r = run_cli("inspect", "--config", str(cfg), "--prefix", "x_")
    out = json.loads(r.stdout)
    assert (
        r.returncode == 0
        and {t["name"] for t in out["tools"]} == {"x_add", "x_locate", "x_shout"}
        and out["hidden"] == ["x_delete_all"]
    )
    r = run_cli("check", "--config", str(cfg), "--no-compile")
    assert r.returncode == 0 and "nothing was imported" in r.stdout
    assert run_cli("check", "--config", str(cfg), "--no-import-sources").returncode == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"tools": {"oops": 1}}')
    r = run_cli("check", "--config", str(bad))
    assert r.returncode == 2 and "tools.oops" in r.stderr
    r = run_cli(
        "inspect", "--source", OBJ, "--include", "add", env={"AAA_MCP__SERVER__NAME": "from-env"}
    )
    assert json.loads(r.stdout)["server"]["name"] == "from-env"


async def test_a_generated_server_works_over_real_stdio(tmp_path):
    from aaa.mcp import MCPClient

    cfg = tmp_path / "s.json"
    cfg.write_text(
        json.dumps(
            {
                "sources": [{"type": "objects", "ref": OBJ}],
                "server": {"instructions": "x"},
                "tenancy": {"tenant": "acme"},
            }
        )
    )
    cfg.chmod(0o644)
    env = {"PYTHONPATH": f"{HELPERS.parent}:{HELPERS.parent.parent / 'src'}"}
    async with MCPClient.stdio(
        sys.executable,
        ["-m", "aaa.cli", "mcp", "serve", "--config", str(cfg), "--exclude", "broken_*"],
        name="gen",
        cwd=str(HELPERS.parent),
        env=env,
    ) as c:
        assert {t.name for t in await c.list_tools()} == {"add", "locate", "shout"}
        out, is_error = await c.call("shout", {"text": "hi"})
        assert out == "HI" and not is_error
