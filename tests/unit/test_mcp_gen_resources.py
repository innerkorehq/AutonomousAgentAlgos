"""MCP resources and prompts: providers, tenant scoping, traversal, uniform not-found, and every serializer agreeing."""

import json
import os
import sys
from pathlib import Path

import pytest

from aaa.data import Doc, InMemoryIndex
from aaa.mcp import MCPServerBuilder, wire
from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import SpecError, load_spec
from aaa.mcp.server import MCPToolServer
from aaa.mcp.sources.prompts import PromptError
from aaa.mcp.sources.resources import ResourceNotFound, parse_uri
from aaa.mcp.spec import MCPSpec
from aaa.memory import MemoryEntry, Scope
from aaa.memory.fake import FakeMemory
from aaa.tools import ToolRegistry

HELPERS = Path(__file__).parent.parent / "helpers"


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "docs"
    for tenant, name, body in (
        ("acme", "guide.md", "# Acme guide"),
        ("acme", "sub/deep.txt", "deep acme"),
        ("globex", "secret.md", "globex only"),
    ):
        f = root / tenant / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    (root / "acme" / ".env").write_text("TOKEN=hunter2")
    (root / "acme" / "id_rsa").write_text("PRIVATE KEY")
    (root / "outside.txt").write_text("above the tenant dir")
    (root / "acme" / "link").symlink_to(root / "globex" / "secret.md")
    return root


async def server(spec: dict, **kw):
    return await acompile_spec(MCPSpec.model_validate(spec), **kw)


def files_spec(root, **over):
    return {"resources": [{"type": "files", "name": "docs", "root": str(root), **over}]}


# ===================================================================== spec rules
def test_resource_and_prompt_specs_reject_unsafe_or_inconsistent_definitions():
    with pytest.raises(SpecError, match="tenant_key"):
        load_spec({"resources": [{"type": "index", "name": "kb", "ref": "m:a"}]}, environ={})
    load_spec(
        {"resources": [{"type": "index", "name": "kb", "ref": "m:a", "shared": True}]}, environ={}
    )
    load_spec(
        {"resources": [{"type": "index", "name": "kb", "ref": "m:a", "tenant_key": "tenant"}]},
        environ={},
    )
    with pytest.raises(SpecError, match="not declared as arguments"):
        load_spec({"prompts": [{"name": "p", "messages": [{"text": "Hello {who}"}]}]}, environ={})
    with pytest.raises(SpecError, match="not allowed"):
        load_spec(
            {
                "prompts": [
                    {
                        "name": "p",
                        "arguments": [{"name": "who"}],
                        "messages": [{"text": "{who.__class__}"}],
                    }
                ]
            },
            environ={},
        )
    with pytest.raises(SpecError, match="choices are required"):
        load_spec(
            {
                "prompts": [
                    {
                        "name": "p",
                        "arguments": [{"name": "m", "type": "enum"}],
                        "messages": [{"text": "{m}"}],
                    }
                ]
            },
            environ={},
        )
    with pytest.raises(SpecError, match="duplicate argument"):
        load_spec(
            {
                "prompts": [
                    {
                        "name": "p",
                        "arguments": [{"name": "a"}, {"name": "a"}],
                        "messages": [{"text": "x"}],
                    }
                ]
            },
            environ={},
        )


async def test_compile_time_resource_mistakes_are_loud(tmp_path):
    with pytest.raises(SpecError, match="not a directory"):
        await server(files_spec(tmp_path / "missing"))
    with pytest.raises(SpecError, match="share a kind and a name"):
        await server(
            {
                "resources": [
                    {"type": "static", "name": "a", "text": "x"},
                    {"type": "static", "name": "a", "text": "y"},
                ]
            }
        )
    with pytest.raises(SpecError, match="is not an Index"):
        await server(
            {
                "resources": [
                    {
                        "type": "index",
                        "name": "k",
                        "ref": "helpers.aaa_mcp_agents:algo",
                        "shared": True,
                    }
                ]
            }
        )
    with pytest.raises(SpecError, match="two prompts share a name"):
        await server(
            {
                "prompts": [
                    {"name": "p", "messages": [{"text": "x"}]},
                    {"name": "p", "messages": [{"text": "y"}]},
                ]
            }
        )


# ===================================================================== uri handling
@pytest.mark.parametrize(
    "uri",
    [
        "http://files/docs/a",
        "aaa://nothing/docs/a",
        "aaa://files/bad name/a",
        "aaa://files/",
        "aaa://user@files/docs/a",
        "aaa://files/docs/a#frag",
        "not a uri",
        "aaa://files:80/docs/a",
        "",
    ],
)
def test_malformed_uris_are_not_found(uri):
    with pytest.raises(ResourceNotFound):
        parse_uri(uri)


def test_uri_parsing_decodes_the_path_and_query():
    assert parse_uri("aaa://files/docs/a%20b/c.txt") == ("files", "docs", "a b/c.txt", {})
    assert parse_uri("aaa://index/kb/search?q=hello%20world") == (
        "index",
        "kb",
        "search",
        {"q": "hello world"},
    )


# ===================================================================== static and files
async def test_static_resources_are_listed_read_and_clipped():
    core = await server(
        {
            "resources": [
                {
                    "type": "static",
                    "name": "policy",
                    "text": "x" * 5000,
                    "mime_type": "text/markdown",
                    "title": "Policy",
                }
            ],
            "limits": {"max_resource_bytes": 1000},
        }
    )
    (r,) = core.list_resources()
    assert (r.uri, r.title, r.mime_type, r.size) == (
        "aaa://static/policy",
        "Policy",
        "text/markdown",
        5000,
    )
    c = await core.read_resource("aaa://static/policy")
    assert c.text.endswith("[truncated at 1000 bytes]") and c.mime_type == "text/markdown"
    for bad in ("aaa://static/policy/extra", "aaa://static/policy?x=1", "aaa://static/other"):
        with pytest.raises(ResourceNotFound):
            await core.read_resource(bad)


async def test_file_resources_are_tenant_scoped_and_confined(tree):
    core = await server(files_spec(tree))
    acme = sorted(r.name for r in core.list_resources("acme"))
    assert acme == ["guide.md", "sub/deep.txt"]  # no .env, no key, no symlink out
    assert sorted(r.name for r in core.list_resources("globex")) == ["secret.md"]
    assert core.list_resources("nobody") == []
    c = await core.read_resource("aaa://files/docs/guide.md", tenant="acme")
    assert c.text == "# Acme guide" and c.mime_type == "text/markdown"
    assert (
        await core.read_resource("aaa://files/docs/sub/deep.txt", tenant="acme")
    ).text == "deep acme"

    attacks = [
        ("aaa://files/docs/secret.md", "acme"),  # another tenant's file, by name
        ("aaa://files/docs/..%2Fglobex%2Fsecret.md", "acme"),  # traversal out of the tenant dir
        ("aaa://files/docs/../globex/secret.md", "acme"),
        ("aaa://files/docs/..%2Foutside.txt", "acme"),
        ("aaa://files/docs/.env", "acme"),  # hidden
        ("aaa://files/docs/id_rsa", "acme"),  # key-like
        ("aaa://files/docs/link", "acme"),  # symlink out of the root
        ("aaa://files/docs//etc/passwd", "acme"),
        ("aaa://files/docs/guide.md", "../globex"),  # a tenant name trying to climb
        ("aaa://files/docs/guide.md", ".."),
        ("aaa://files/docs/nope.md", "acme"),  # simply missing
        ("aaa://files/docs/", "acme"),
    ]
    messages = set()
    for uri, tenant in attacks:
        with pytest.raises(ResourceNotFound) as e:
            await core.read_resource(uri, tenant=tenant)
        messages.add(repr(e.value))
    assert len(messages) == 1  # forbidden, missing and malformed are indistinguishable


async def test_shared_file_resources_serve_the_root_to_everyone_and_the_pattern_applies(tree):
    core = await server(files_spec(tree, shared=True, pattern="acme/*.md"))
    assert [r.name for r in core.list_resources("anyone")] == ["acme/guide.md"]
    assert (
        await core.read_resource("aaa://files/docs/acme/guide.md", tenant="whoever")
    ).text == "# Acme guide"
    with pytest.raises(ResourceNotFound):
        await core.read_resource(
            "aaa://files/docs/acme/sub/deep.txt", tenant="whoever"
        )  # outside the pattern


async def test_file_reads_are_size_capped(tree):
    (tree / "acme" / "big.txt").write_text("y" * 50_000)
    core = await server(files_spec(tree) | {"limits": {"max_resource_bytes": 2000}})
    c = await core.read_resource("aaa://files/docs/big.txt", tenant="acme")
    assert len(c.text) <= 2000 + 10


async def test_the_listing_is_capped():
    d = Path(os.environ.get("TMPDIR", "/tmp")) / f"aaa-list-{os.getpid()}"
    d.mkdir(exist_ok=True)
    try:
        for i in range(30):
            (d / f"f{i:02}.txt").write_text("x")
        core = await server(
            {
                "resources": [{"type": "files", "name": "d", "root": str(d), "shared": True}],
                "limits": {"max_resources_listed": 10},
            }
        )
        assert len(core.list_resources()) == 10
    finally:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


# ===================================================================== index and memory
async def test_index_resources_filter_by_tenant_and_reject_odd_queries():
    idx = InMemoryIndex()
    await idx.upsert(
        [
            Doc(id="a", text="refund policy for acme", meta={"tenant": "acme"}),
            Doc(id="g", text="refund policy for globex", meta={"tenant": "globex"}),
        ]
    )
    import types as _t

    mod = _t.ModuleType("kbmod")
    mod.kb = idx  # type: ignore[attr-defined]
    sys.modules["kbmod"] = mod
    try:
        core = await server(
            {
                "resources": [
                    {
                        "type": "index",
                        "name": "kb",
                        "ref": "kbmod:kb",
                        "tenant_key": "tenant",
                        "k": 5,
                    }
                ]
            }
        )
        (t,) = core.list_resource_templates()
        assert (
            t.uri_template == "aaa://index/kb/search?q={query}"
            and t.mime_type == "application/json"
        )
        got = json.loads(
            (
                await core.read_resource("aaa://index/kb/search?q=refund%20policy", tenant="acme")
            ).text
        )
        assert [h["id"] for h in got] == ["a"]
        assert [
            h["id"]
            for h in json.loads(
                (await core.read_resource("aaa://index/kb/search?q=refund", tenant="globex")).text
            )
        ] == ["g"]
        for bad in (
            "aaa://index/kb/search",
            "aaa://index/kb/search?q=",
            f"aaa://index/kb/search?q={'x' * 501}",
            "aaa://index/kb/search?q=a&evil=1",
            "aaa://index/kb/other?q=a",
            "aaa://index/nope/search?q=a",
        ):
            with pytest.raises(ResourceNotFound):
                await core.read_resource(bad, tenant="acme")
        shared = await server(
            {"resources": [{"type": "index", "name": "kb", "ref": "kbmod:kb", "shared": True}]}
        )
        assert (
            len(
                json.loads(
                    (
                        await shared.read_resource(
                            "aaa://index/kb/search?q=refund", tenant="anyone"
                        )
                    ).text
                )
            )
            == 2
        )
    finally:
        sys.modules.pop("kbmod", None)


async def test_memory_resources_use_the_callers_tenant_and_user_scope():
    mem = FakeMemory()
    await mem.add([MemoryEntry(text="ann likes tea")], Scope(tenant="acme", user="ann"))
    await mem.add([MemoryEntry(text="bob likes tea too")], Scope(tenant="acme", user="bob"))
    await mem.add([MemoryEntry(text="gus likes tea")], Scope(tenant="globex", user="gus"))
    import types as _t

    mod = _t.ModuleType("memmod")
    mod.mem = mem  # type: ignore[attr-defined]
    sys.modules["memmod"] = mod
    try:
        core = await server({"resources": [{"type": "memory", "name": "m", "ref": "memmod:mem"}]})
        mine = json.loads(
            (
                await core.read_resource("aaa://memory/m/search?q=tea", tenant="acme", user="ann")
            ).text
        )
        assert [m["text"] for m in mine] == ["ann likes tea"]
        theirs = json.loads(
            (
                await core.read_resource("aaa://memory/m/search?q=tea", tenant="globex", user="gus")
            ).text
        )
        assert [m["text"] for m in theirs] == ["gus likes tea"]
    finally:
        sys.modules.pop("memmod", None)


async def test_a_failing_provider_looks_like_a_missing_resource():
    class Broken:
        async def search(self, *a, **k):
            raise RuntimeError("db password is hunter2")

    import types as _t

    mod = _t.ModuleType("brokenmod")
    mod.idx = Broken()  # type: ignore[attr-defined]
    sys.modules["brokenmod"] = mod
    try:
        core = await server(
            {"resources": [{"type": "index", "name": "b", "ref": "brokenmod:idx", "shared": True}]}
        )
        with pytest.raises(ResourceNotFound) as e:
            await core.read_resource("aaa://index/b/search?q=x", tenant="t")
        assert "hunter2" not in repr(e.value) and e.value.__cause__ is None
    finally:
        sys.modules.pop("brokenmod", None)


# ===================================================================== prompts
PROMPTS = {
    "prompts": [
        {
            "name": "triage",
            "title": "Triage a ticket",
            "description": "Classify a support ticket.",
            "arguments": [
                {"name": "ticket", "description": "The ticket text"},
                {"name": "priority", "type": "enum", "choices": ["low", "high"], "required": False},
                {"name": "limit", "type": "integer", "required": False},
            ],
            "messages": [
                {
                    "role": "user",
                    "text": "Triage this ticket (priority {priority}, max {limit} labels): {ticket}. Use {{braces}} literally.",
                },
                {"role": "assistant", "text": "Understood."},
            ],
        }
    ]
}


async def test_prompts_validate_and_fill_their_arguments():
    core = await server(PROMPTS)
    (p,) = core.list_prompts()
    assert p.name == "triage" and [(a.name, a.required) for a in p.arguments] == [
        ("ticket", True),
        ("priority", False),
        ("limit", False),
    ]
    r = core.get_prompt("triage", {"ticket": "printer on fire", "priority": "high", "limit": "3"})
    assert r.description == "Classify a support ticket." and r.messages[1] == (
        "assistant",
        "Understood.",
    )
    assert (
        r.messages[0][1]
        == "Triage this ticket (priority high, max 3 labels): printer on fire. Use {braces} literally."
    )
    assert (
        "max  labels" in core.get_prompt("triage", {"ticket": "x"}).messages[0][1]
    )  # optional arguments left empty
    for args, why in (
        ({}, "missing required"),
        ({"ticket": "x", "who": "y"}, "unknown argument"),
        ({"ticket": "x", "priority": "urgent"}, "one of"),
        ({"ticket": "x", "limit": "many"}, "integer"),
    ):
        with pytest.raises(PromptError, match=why):
            core.get_prompt("triage", args)
    with pytest.raises(PromptError, match="unknown prompt"):
        core.get_prompt("nope")


async def test_user_text_in_a_prompt_argument_is_not_interpreted():
    core = await server(PROMPTS)
    r = core.get_prompt("triage", {"ticket": "{priority} {limit} {ticket.__class__} {{x}}"})
    assert "{ticket.__class__}" in r.messages[0][1] and "{{x}}" in r.messages[0][1]


# ===================================================================== wire and capabilities
async def test_capabilities_appear_only_for_what_is_configured(tree):
    plain = await server({})
    assert wire.capabilities(plain) == {"tools": {"listChanged": False}}
    full = await server(files_spec(tree) | PROMPTS)
    assert wire.capabilities(full) == {
        "tools": {"listChanged": False},
        "resources": {"subscribe": False, "listChanged": False},
        "prompts": {"listChanged": False},
    }
    assert wire.resource_read_result(
        await full.read_resource("aaa://files/docs/guide.md", tenant="acme")
    ) == {
        "contents": [
            {
                "uri": "aaa://files/docs/guide.md",
                "mimeType": "text/markdown",
                "text": "# Acme guide",
            }
        ]
    }
    assert wire.prompts_list_result(full.list_prompts())["prompts"][0]["arguments"][0] == {
        "name": "ticket",
        "description": "The ticket text",
        "required": True,
    }
    w = wire.prompt_get_result(full.get_prompt("triage", {"ticket": "t"}))
    assert (
        w["messages"][0]["content"] == {"type": "text", "text": w["messages"][0]["content"]["text"]}
        and w["description"]
    )


async def test_describe_lists_resources_and_prompts(tree):
    from aaa.mcp.compile import describe

    d = describe(await server(files_spec(tree, shared=True) | PROMPTS))
    assert d["prompts"][0]["name"] == "triage" and d["resource_templates"][0][
        "uriTemplate"
    ].startswith("aaa://files/docs/")


# ===================================================================== serializers agree
mcp = pytest.importorskip("mcp")


async def sdk_session(core):
    from aaa.mcp.server import build_server

    server_ = build_server(core)
    if hasattr(mcp, "Client"):
        return mcp.Client(server_)
    from mcp.shared.memory import create_connected_server_and_client_session

    return create_connected_server_and_client_session(server_)


def dump(model):
    return model.model_dump(by_alias=True, exclude_none=True, mode="json")


async def test_the_sdk_server_and_the_http_handler_serve_resources_and_prompts_identically(tree):
    from aaa_server.mcp_http import handle

    core = await server(
        files_spec(tree, shared=True, pattern="acme/*.md") | PROMPTS | {"server": {"name": "x"}}
    )

    async def http(method, params=None):
        r = await handle(
            core,
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
            tenant="acme",
            user="u",
        )
        assert r is not None
        return r

    async with await sdk_session(core) as c:
        sdk_resources = [dump(r) for r in (await c.list_resources()).resources]
        sdk_templates = (
            [dump(t) for t in (await c.list_resource_templates()).resource_templates]
            if hasattr(await c.list_resource_templates(), "resource_templates")
            else [dump(t) for t in (await c.list_resource_templates()).resourceTemplates]
        )
        read = await c.read_resource("aaa://files/docs/acme/guide.md")
        prompts = [dump(p) for p in (await c.list_prompts()).prompts]
        got = await c.get_prompt("triage", {"ticket": "T"})
        with pytest.raises(Exception, match="not found|resource"):
            await c.read_resource("aaa://files/docs/acme/missing.md")
    h_res = (await http("resources/list"))["result"]["resources"]
    h_tpl = (await http("resources/templates/list"))["result"]["resourceTemplates"]
    h_read = (await http("resources/read", {"uri": "aaa://files/docs/acme/guide.md"}))["result"][
        "contents"
    ]
    h_prompts = (await http("prompts/list"))["result"]["prompts"]
    h_get = (await http("prompts/get", {"name": "triage", "arguments": {"ticket": "T"}}))["result"]
    norm = lambda xs: sorted(xs, key=lambda d: json.dumps(d, sort_keys=True))
    assert norm(sdk_resources) == norm(h_res)
    assert norm(sdk_templates) == norm(h_tpl)
    assert [c.text for c in read.contents] == [c["text"] for c in h_read]
    assert norm(prompts) == norm(h_prompts)
    assert [(m.role, m.content.text) for m in got.messages] == [
        (m["role"], m["content"]["text"]) for m in h_get["messages"]
    ]
    # the HTTP errors: one code for every unreadable resource, a parameter error for a bad prompt
    for uri in ("aaa://files/docs/acme/missing.md", "aaa://files/docs/../x", "junk"):
        assert (await http("resources/read", {"uri": uri}))["error"]["code"] == -32002
    assert (await http("resources/read", {}))["error"]["code"] == -32602
    assert (await http("prompts/get", {"name": "triage", "arguments": {}}))["error"][
        "code"
    ] == -32602
    assert (await http("prompts/get", {"name": "nope"}))["error"]["code"] == -32602


async def test_the_http_handler_tells_each_tenant_only_their_files(tree):
    from aaa_server.mcp_http import handle

    core = await server(files_spec(tree))
    for tenant, expect in (("acme", {"guide.md", "sub/deep.txt"}), ("globex", {"secret.md"})):
        r = await handle(
            core, {"jsonrpc": "2.0", "id": 1, "method": "resources/list"}, tenant=tenant, user="u"
        )
        assert r is not None and {x["name"] for x in r["result"]["resources"]} == expect
    r = await handle(
        core,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "aaa://files/docs/secret.md"},
        },
        tenant="acme",
        user="u",
    )
    assert r is not None and r["error"]["code"] == -32002


async def test_without_resources_or_prompts_those_methods_do_not_exist():
    from aaa_server.mcp_http import handle

    core = MCPToolServer(ToolRegistry([]))
    for m in ("resources/list", "prompts/list"):
        r = await handle(core, {"jsonrpc": "2.0", "id": 1, "method": m}, tenant="t", user="u")
        assert r is not None and r["error"]["code"] == -32601


async def test_resources_and_prompts_work_over_real_stdio(tree, tmp_path):
    from aaa.mcp import MCPClient

    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({"tenancy": {"tenant": "acme"}, **files_spec(tree), **PROMPTS}))
    cfg.chmod(0o644)
    env = {"PYTHONPATH": f"{HELPERS.parent}:{HELPERS.parent.parent / 'src'}"}
    async with MCPClient.stdio(
        sys.executable,
        ["-m", "aaa.cli", "mcp", "serve", "--config", str(cfg)],
        name="g",
        cwd=str(HELPERS.parent),
        env=env,
    ) as c:
        assert sorted(r["name"] for r in await c.list_resources()) == ["guide.md", "sub/deep.txt"]
        assert await c.read_resource("aaa://files/docs/guide.md") == "# Acme guide"
        with pytest.raises(Exception):  # noqa: B017 - the SDK's error type differs between 1.x and 2.x
            await c.read_resource("aaa://files/docs/secret.md")
        prompts = await c.session.list_prompts()
        assert [p.name for p in prompts.prompts] == ["triage"]


def test_the_builder_can_describe_resources_and_prompts_through_a_file(tmp_path):
    b = MCPServerBuilder("x")
    assert b.spec().resources == [] and b.spec().prompts == []
