# Command line

```text
aaa store check <store>        open the store (bringing an old layout forward) and report runs and tenants
aaa runs <store> [--open]      one line per run
aaa events <store> <run> [--data] [--json]
aaa mcp serve --config server.toml        serve a generated MCP server over stdio (also: check, inspect; see the generator guide)
aaa mcp init [DIR]                     generate a project (interactive, or --yes with flags)
aaa mcp export DIR [--config F|--check|--update]
aaa mcp serve <module:factory>    serve a ToolRegistry / MCPToolServer over MCP stdio (extra `mcp`)
aaa-server serve module:factory [--host 127.0.0.1] [--port 8000]
```

`<store>` is `sqlite:///file.db` (or a `*.db` path), `postgresql://...` or `redis://...`. The `aaa` commands only read events; opening a store may migrate its layout forward (see
[Evolving events and stores](../../guides/evolving-events.md)). `aaa-server` is installed by the separate [`aaa-server`](../../guides/server.md) package.
