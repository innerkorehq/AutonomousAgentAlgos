# MCP: using Model Context Protocol servers

[MCP](https://modelcontextprotocol.io) is the open protocol that lets an agent use tools and data from any compliant server: a
filesystem, a database, GitHub, a memory service, your own internal API. `aaa.mcp` is an MCP **client**: it connects to a server and
turns the server's tools into ordinary AAA tools, so they get argument validation, timeouts, human approval and the
[execution guard](tool-guard.md) like any `@tool`.

```python
--8<-- "examples/mcp_tools.py"
```

```text
weather__forecast        delivery=at_least_once approval=False
weather__reset_station   delivery=at_most_once  approval=True
The forecast says: Lisbon: sunny for the next 3 days
```

Install the client with `pip install "autonomous-agent-algos[mcp]"`. It works with both major versions of the official `mcp` SDK
(1.x and 2.x), which spell several fields differently; the adapter hides that.

## Connecting

```python
async with MCPClient.stdio("uvx", ["some-mcp-server"], env={"API_KEY": "..."}) as srv: ...      # a local subprocess
async with MCPClient.http("https://mcp.example.com/mcp", headers={"Authorization": "Bearer ..."}) as srv: ...
srv = MCPClient(an_open_client_or_session, name="mine")                                         # you manage the connection
```

- **stdio** launches the command as a child process. The child receives only a small safe set of environment variables plus the `env`
  you pass, so put credentials there explicitly.
- **http** speaks Streamable HTTP; `headers` carries credentials.
- Passing your own open `mcp.Client` (2.x) or `ClientSession` (1.x) works for anything else (a custom transport, OAuth, an in-process server for tests); AAA then never closes it.

**One connection per process.** In a [cluster](clusters.md) every worker opens its own connection (a stdio server is one subprocess per worker). Tools
are discovered at startup: call `await srv.list_tools(refresh=True)` and rebuild the registry if a server's tool list changes.

## From server tools to AAA tools

```python
tools = await srv.load_tools(
    prefix="git_",                    # default: "<name>__"; names are made API-safe (a-z, 0-9, _, -; at most 64 characters)
    include=["read_*", "list_*"],     # glob patterns on the server's names
    exclude=["*delete*"],
    approval="destructive",           # "none" (default) | "destructive" | "always"
    max_result_chars=20_000,          # long results are truncated with a marker, protecting the context window
    timeout_s=60,
)
registry = ToolRegistry([*tools, my_other_tool])
```

Each server tool becomes a `SchemaTool`: it carries the server's JSON Schema, and the model's arguments are **validated against it before the
call leaves your process**. A server error comes back as a normal failed tool call the model can react to; binary content (images, audio) is
replaced by a short marker rather than burning context.

### Annotations decide the safety policy

MCP servers may mark tools with hints. AAA uses them, conservatively:

| Server says | `delivery` (what happens if a worker dies mid-call) | `approval="destructive"` |
|---|---|---|
| `readOnlyHint: true` | `at_least_once`: re-run freely | no approval |
| `idempotentHint: true` | `at_least_once` | approval (it still changes things) |
| nothing / anything else | `at_most_once`: never re-run, reported as `OutcomeUnknown` | approval |

The last row follows the MCP spec itself, which says a tool with no annotations must be treated as destructive. The consequence is
deliberate: **with `approval="destructive"`, an un-annotated tool waits for a human**. Annotate your own tools or relax `approval` for servers you trust.

!!! warning "Treat MCP servers as untrusted input"
    - Annotations are hints from the server. A malicious or buggy server can mark a destructive tool read-only. Use
      `trust_annotations=False` for servers you do not control: every tool is then treated as destructive and non-repeatable.
    - Tool **descriptions and results** go into the model's context, so they can carry prompt injection. Use `include`/`exclude` to expose
      only what a run needs, keep `approval` on anything that writes or sends, and cap results with `max_result_chars`.
    - Prefer a dedicated credential per server and per tenant, scoped to what the agent needs.

## Resources

```python
await srv.list_resources()
text = await srv.read_resource("file:///project/README.md")
```

Resource text is returned; binary parts are summarized. Resources are not exposed to the model automatically: wrap what you want in a tool of your own.

## MCP memory servers

Many memory systems (Mem0, Graphiti, Supermemory and others) ship MCP servers. Because their tool names and arguments differ, you say which
tool does what; there is no built-in table to go stale:

```python
from aaa.mcp import mcp_memory

print([t.name for t in await srv.list_tools()])     # look at what the server really offers
memory = mcp_memory(
    srv,
    add="add_memory", search="search_memory", forget="delete_all_memories",   # names from that list
    build_search=lambda query, scope, k: {"query": query, "user_id": scope.slug(), "limit": k},   # only if the defaults differ
)
MemoryHandler(memory).attach(engine)           # now it is an ordinary AAA memory: recall, write-back, erasure
```

The defaults send `user_id=<scope slug>` and read JSON lists, JSON objects holding one (`results`, `memories`, `facts`, `items`...), or plain lines.
See [Long-term memory](memory.md). **Isolation is the server's job here**: AAA passes the scope as an argument, but an MCP server is a separate process
and must honour it. For hard tenant isolation, connect one server (or one credential) per tenant.

## What is verified

The client is tested against a **real** MCP server built with the official SDK, in-process and as a stdio subprocess and over Streamable HTTP
(with credentials), on both SDK major versions (CI runs the 1.x and 2.x legs). Nothing here has been run against a third-party production MCP server;
the protocol is the same, but servers vary in how they annotate tools and report errors. Not yet supported: server-initiated **sampling** and
**elicitation** requests, prompts, resource subscriptions, and OAuth flows (bring your own authenticated client via `MCPClient(session)`).


## Serving your tools over MCP

`MCPToolServer` offers a `ToolRegistry` (and, optionally, whole algorithms) to any MCP client: Claude Desktop, an IDE, another agent. Run it on stdio with `aaa mcp serve myapp.tools:build`
(a function returning a `ToolRegistry` or an `MCPToolServer`) or call `await serve_stdio(server)`; behind [`aaa-server`](server.md) the same object is served at `POST /mcp` with the
server's authentication. Works with both major versions of the `mcp` SDK.

```python
--8<-- "examples/mcp_server.py"
```

```text
offered: ['add', 'lookup', 'run_agent']
add: 5
bad args: True
lookup: lookup failed
wire_money: unknown tool 'wire_money' | error: True
run_agent: {"kind": "success", "value": {"text": "Paris.", "turns": 1}, "reason": null}
```

The defaults assume the caller is not the person who wrote the tools:

- **Tools that require approval are hidden** and a call to one is answered exactly like a call to a tool that does not exist. MCP has no approval channel, so nothing is ever approved
  automatically. To offer one anyway, pass `approval=` (an async callback `(name, arguments, tenant) -> bool` that decides, failing closed if it raises) or, if you accept the
  risk, `expose_approval_tools=True`.
- **Arguments are validated by the tool's schema** before it runs (unknown fields are refused). A tool's `ToolError` message is shown; any other exception is logged on the server and
  reported only as "`<tool>` failed", so connection strings and stack traces do not leave. Results are capped (`max_result_chars`).
- **Tenant and user are fixed by the host**, never by the client: `tenant=`/`user=` on a stdio server, the authenticated principal behind `aaa-server`. A tool that takes a `ctx`
  sees them in `ctx.tenant` / `ctx.meta["user"]`.
- **`run_agent` is off unless you pass `algorithms=`** (and an `engine=`). It runs the named algorithm to completion in the host's tenant under `agent_budget` (default: 50 steps, 200k
  tokens, 120 s), and returns only the outcome. A run that needs a human approval cannot be approved from MCP; the deadline ends it.

Verified: the official SDK client (1.x and 2.x, in process) and `MCPClient` (against a real stdio subprocess, and against the HTTP endpoint on a real socket). Not verified: specific MCP hosts such
as Claude Desktop. The HTTP endpoint implements the stateless subset of Streamable HTTP (one JSON answer per POST, no sessions, no server-initiated messages), which covers `initialize`,
`ping`, `tools/list` and `tools/call`; resources, prompts, sampling and progress notifications are not served.
