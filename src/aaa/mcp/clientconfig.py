"""The entry an MCP client needs to connect to a generated server: `aaa mcp client-config`.

Only the NAMES of environment variables appear, as `${NAME}` placeholders (`${env:NAME}` for Cursor), never a value; the client or
the operator supplies them. Remote (HTTP) entries assume the server is run behind aaa-server, whose API key is `MCP_API_KEY`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aaa.mcp.config import SpecError
from aaa.mcp.spec import MCPSpec

CLIENTS = ("claude-desktop", "cursor", "generic")
API_KEY_ENV = "MCP_API_KEY"
_REF = re.compile(r"^env:([A-Za-z_][A-Za-z0-9_]*)$")
_BRACED = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def env_names(spec: MCPSpec) -> list[str]:
    """Names (never values) of the environment variables the configuration reads: API credentials, upstream headers, pass-through."""
    out: set[str] = set()

    def ref(v: str) -> None:
        if m := _REF.match(v.strip()):
            out.add(m.group(1))
        out.update(_BRACED.findall(v))

    for src in spec.sources:
        if src.type == "openapi" and src.auth is not None:
            ref(src.auth.secret)
        elif src.type == "mcp":
            out.update(src.env_from)
            for v in src.headers.values():
                ref(v)
    return sorted(out)


def _placeholder(client: str, name: str) -> str:
    return f"${{env:{name}}}" if client == "cursor" else f"${{{name}}}"


def build(
    spec: MCPSpec,
    config: str | Path,
    *,
    client: str = "generic",
    transport: str = "stdio",
    url: str | None = None,
    name: str | None = None,
    overlays: list[str] | None = None,
) -> dict[str, Any]:
    """`{"mcpServers": {name: entry}}` for the single server `spec` describes."""
    if client not in CLIENTS:
        raise SpecError(f"unknown client {client!r} (use one of {', '.join(CLIENTS)})")
    label = _SAFE_NAME.sub("-", name or spec.server.name).strip("-") or "aaa"
    if transport == "stdio":
        if url:
            raise SpecError("--url applies to --transport http")
        args = ["mcp", "serve", "--config", str(Path(config).resolve())]
        for o in overlays or []:
            args += ["--overlay", str(Path(o).resolve())]
        entry: dict[str, Any] = {"command": "aaa", "args": args}
        names = env_names(spec)
        if names:
            entry["env"] = {n: _placeholder(client, n) for n in names}
    elif transport == "http":
        if not url:
            raise SpecError("--transport http needs --url (where aaa-server serves /mcp)")
        if not url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise SpecError("the URL must be https (plain http only for localhost)")
        bearer = f"Bearer {_placeholder(client, API_KEY_ENV)}"
        if (
            client == "claude-desktop"
        ):  # the desktop app launches local commands; a remote server goes through a bridge
            entry = {
                "command": "npx",
                "args": ["-y", "mcp-remote", url, "--header", f"Authorization:{bearer}"],
                "env": {API_KEY_ENV: _placeholder(client, API_KEY_ENV)},
            }
        else:
            entry = {"url": url, "headers": {"Authorization": bearer}}
    else:
        raise SpecError("transport must be 'stdio' or 'http'")
    return {"mcpServers": {label: entry}}


__all__ = ["CLIENTS", "build", "env_names"]
