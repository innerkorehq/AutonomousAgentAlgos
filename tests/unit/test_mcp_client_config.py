"""`aaa mcp client-config`: golden entries per client, names but never values, and the stdio entry really launches the server."""

import json
import os
import select
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aaa.mcp import clientconfig
from aaa.mcp.config import SpecError, load_spec

GOLDEN = Path(__file__).parent.parent / "golden" / "mcp_client_config"
CONFIG = {
    "server": {"name": "order-desk"},
    "sources": [
        {
            "type": "openapi",
            "document": "https://api.example.com/openapi.json",
            "allowed_hosts": ["api.example.com"],
            "auth": {"scheme": "bearer", "secret": "env:ORDERS_API_TOKEN"},
        },
        {
            "type": "mcp",
            "name": "search",
            "transport": "http",
            "url": "https://search.example.com/mcp",
            "allowed_hosts": ["search.example.com"],
            "headers": {"Authorization": "Bearer ${SEARCH_TOKEN}"},
        },
        {
            "type": "mcp",
            "name": "files",
            "transport": "stdio",
            "command": "files-mcp",
            "env_from": ["FILES_KEY"],
        },
    ],
}
CASES = [
    ("claude-desktop", "stdio", None),
    ("cursor", "stdio", None),
    ("generic", "stdio", None),
    ("claude-desktop", "http", "https://mcp.example.com/mcp"),
    ("cursor", "http", "https://mcp.example.com/mcp"),
    ("generic", "http", "https://mcp.example.com/mcp"),
]


def spec():
    return load_spec(CONFIG, environ={})


@pytest.mark.parametrize(("client", "transport", "url"), CASES)
def test_entry_matches_the_golden_file(client, transport, url):
    got = clientconfig.build(
        spec(), "/srv/mcp/server.toml", client=client, transport=transport, url=url
    )
    path = GOLDEN / f"{client}-{transport}.json"
    if os.environ.get("AAA_REGEN_GOLDEN"):
        path.write_text(json.dumps(got, indent=2) + "\n")
    assert got == json.loads(path.read_text())


def test_names_are_listed_and_no_value_is_ever_present(monkeypatch):
    for n in ("ORDERS_API_TOKEN", "SEARCH_TOKEN", "FILES_KEY", "MCP_API_KEY"):
        monkeypatch.setenv(n, "canary-value-123")
    assert clientconfig.env_names(spec()) == ["FILES_KEY", "ORDERS_API_TOKEN", "SEARCH_TOKEN"]
    for client, transport, url in CASES:
        text = json.dumps(
            clientconfig.build(spec(), "/x/s.toml", client=client, transport=transport, url=url)
        )
        assert "canary-value" not in text


def test_overlays_are_carried_into_the_command():
    e = clientconfig.build(spec(), "/x/s.toml", overlays=["/x/prod.toml"])
    assert e["mcpServers"]["order-desk"]["args"][-2:] == ["--overlay", "/x/prod.toml"]


def test_bad_requests_are_refused():
    s = spec()
    with pytest.raises(SpecError, match="unknown client"):
        clientconfig.build(s, "/x", client="vim")
    with pytest.raises(SpecError, match="needs --url"):
        clientconfig.build(s, "/x", transport="http")
    with pytest.raises(SpecError, match="https"):
        clientconfig.build(s, "/x", transport="http", url="http://evil.example.com/mcp")
    with pytest.raises(SpecError, match="--url applies"):
        clientconfig.build(s, "/x", url="https://x.example.com")
    e = clientconfig.build(s, "/x", name="a b/c")
    assert list(e["mcpServers"]) == ["a-b-c"]


def test_cli_validates_first_and_prints_json(tmp_path):
    cfg = tmp_path / "server.toml"
    cfg.write_text(
        '[server]\nname = "s"\n[[sources]]\ntype = "objects"\nref = "helpers.aaa_mcp_server:rich_registry"\n'
    )
    run = lambda *a: subprocess.run(
        [sys.executable, "-m", "aaa.cli", "mcp", "client-config", *a],
        check=False,
        capture_output=True,
        text=True,
    )
    r = run("--config", str(cfg), "--client", "cursor")
    assert r.returncode == 0, r.stderr
    entry = json.loads(r.stdout)["mcpServers"]["s"]
    assert entry["command"] == "aaa" and str(cfg.resolve()) in entry["args"]
    bad = tmp_path / "bad.toml"
    bad.write_text("nonsense = 1\n")
    r = run("--config", str(bad))
    assert r.returncode != 0 and "invalid MCP server configuration" in r.stderr and r.stdout == ""


def test_the_generated_stdio_entry_launches_the_server(tmp_path):
    aaa = Path(sys.executable).parent / "aaa"
    if not aaa.exists():
        pytest.skip("the aaa script is not installed next to this interpreter")
    pytest.importorskip("mcp")
    cfg = tmp_path / "server.toml"
    cfg.write_text(
        '[server]\nname = "s"\n[[sources]]\ntype = "objects"\nref = "helpers.aaa_mcp_server:rich_registry"\n'
    )
    entry = clientconfig.build(load_spec(cfg, environ={}), cfg)["mcpServers"]["s"]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent.parent), *sys.path]),
    }
    cmd = shutil.which("aaa", path=str(aaa.parent)) or str(aaa)
    p = subprocess.Popen(
        [cmd, *entry["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    try:
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        }
        assert p.stdin and p.stdout
        p.stdin.write(json.dumps(init) + "\n")
        p.stdin.flush()
        ready, _, _ = select.select([p.stdout], [], [], 30)
        assert ready, "the server did not answer initialize"
        assert json.loads(p.stdout.readline())["result"]["serverInfo"]["name"] == "s"
    finally:
        p.kill()
        p.wait()
