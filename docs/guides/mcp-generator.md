# Generating an MCP server

Instead of writing a server in Python, describe the one you want. `aaa.mcp` compiles a **configuration** (a file, flags, environment variables or the Python API) into a running MCP server: it chooses which tools to expose, how they are named and described, which hints clients get, and whether results are structured. The same description serves over stdio (`aaa mcp serve`) and over HTTP behind [`aaa-server`](../server/index.md).

This page covers what exists today: AAA objects as the source, tool selection and metadata, structured output, algorithms exposed as tools, the Python builder and the command line. Resources, prompts, OpenAPI and Python-module sources and project scaffolding are planned and not yet implemented; the [threat model](../security/mcp-generator.md) is already written for them.

```python
--8<-- "examples/mcp_generated.py"
```

```text
offers: desk_forecast | Weather forecast | {'readOnlyHint': True, 'openWorldHint': True}
hidden (need approval): ['desk_erase_history']
structured result: {"city": "Oslo", "celsius": 21.5}
```

## The configuration

TOML and JSON need nothing extra; YAML needs `pip install "autonomous-agent-architecture[mcp-config]"` (read with `safe_load` only). Unknown keys are errors that name the field (`tools.exclud: Extra inputs are not permitted`).

```toml
version = 1

[server]
name = "weather-desk"            # also: title, version, instructions (shown to the client)

[limits]
max_result_chars = 100_000       # text results are cut here, with a marker
max_tools = 200                  # a source that offers more is an error, not silently truncated
structured_output = true         # send structuredContent for tools whose return type is an object

[tenancy]                        # who calls are for on a stdio server (aaa-server uses the authenticated principal instead)
tenant = "mcp"
user = "dev"

[[sources]]
type = "objects"                 # package.module:attribute, see below
ref = "myapp.tools:registry"
tags = ["core"]                  # labels `tools.tags` can select on
prefix = ""                      # added to every tool name from this source

[tools]
include = ["*"]                  # globs on the name the source gave
exclude = ["admin_*"]
tags = []                        # when set, only tools carrying one of these tags
rename = { search_docs = "search" }
prefix = "desk_"                 # added after renaming
expose_approval_tools = false

[tools.overrides.desk_search]    # keyed by the FINAL exposed name
title = "Search the docs"
description = "Full-text search over the documentation."
annotations = { read_only = true, open_world = false }
requires_approval = false        # explicitly lift an approval requirement
```

**Sources.** `type = "objects"` imports `package.module:attribute`; the attribute may be a `ToolRegistry`, an `MCPToolServer`, a `Tool`, a list of tools, or a function (sync or async) returning one of those. From Python you can also pass objects directly with `type = "live"` and `compile_spec(spec, live={"key": registry})`.

**Selection and naming.** `include`/`exclude`/`tags` match the name the source gave (after the source's own `prefix`); `rename` keys are those names; the global `prefix` comes after renaming; `overrides` use the final name. Two tools ending up with the same name is an error that names both origins. A typo in `rename` or `overrides` is an error, not a silent no-op.

**Approval.** A tool that `requires_approval` is **hidden** (and answers exactly like an unknown tool) unless you set `expose_approval_tools`, give it an override `requires_approval = false`, or pass an `approval=` callback to `compile_spec`. Nothing is ever approved for you. `aaa mcp check` and `inspect` list what is hidden.

**Hints.** Without overrides, clients get what the tool already says: an approval-gated tool is `destructiveHint`, a tool with `delivery="at_least_once"` is `idempotentHint`; nothing else is claimed. `annotations` in an override *replace* those. Hints are advice to clients; the server enforces none of them.

**Structured output.** A tool whose return type is a pydantic model, `TypedDict` or dataclass advertises an `outputSchema`, and a successful result carries `structuredContent` (validated against the schema; a result that does not conform is reported as an error, not sent) plus the same JSON as text. A structure that would be cut short by `max_result_chars` is not sent at all, only the truncated text. Other return types (strings, lists, numbers) have no output schema. `SchemaTool` accepts an explicit `output_schema=`, and an override can set one.

## Algorithms as tools

Each configured agent becomes its own tool, with its own name, input schema, budget and hints, instead of the generic `run_agent`.

```toml
[runtime]
engine = "myapp.agents:engine"        # an Engine (or a function returning one) wired with your model and handlers

[limits]
max_concurrent_agents = 4             # agent runs in flight across all agent tools

[[agents]]
name = "ask_support"
algorithm = "myapp.agents:support_agent"     # module:attribute, or a live key from Python
description = "Ask the support agent a customer question."
output = "text"                       # the answer's text; "json" returns the whole outcome
input_template = { task = "Customer {customer}: {question}" }
annotations = { open_world = true }

[agents.budget]
deadline_s = 30                       # mandatory
max_steps = 10
max_cost = 0.25

[agents.input_schema]
type = "object"
required = ["question"]
additionalProperties = false
properties = { question = { type = "string" }, customer = { type = "string" } }
```

- **A deadline is mandatory.** MCP has no way to approve a tool call, so a run that parks on a human approval would otherwise wait forever; the engine ends it at the deadline. Steps, tokens and cost are optional extra limits.
- **The template can only insert validated arguments.** Placeholders are plain names (`{question}`): no attributes, indexes, conversions or format specs, and every placeholder must be a property of the input schema. A value that is exactly one placeholder keeps the argument's own type (a list stays a list).
- **The tenant and user come from the host**, never from the arguments (there is no `tenant` field to fill in): the stdio configuration, or the authenticated principal behind `aaa-server`.
- **Concurrency is capped.** Beyond `max_concurrent_agents` runs in flight, a call waits a few seconds and is then refused with a clear message instead of queuing model spend.
- A failed agent (budget exceeded, cancelled, failed) is an MCP tool error carrying the outcome kind and reason.
- With `aaa mcp serve`, the engine named in `runtime.engine` is started and stopped with the server. Behind `aaa-server`, pass the server's engine to `acompile_spec(spec, engine=...)` and `runtime` is ignored.

## The Python builder

The same configuration as typed, chainable Python. Every method edits an `MCPSpec`, so what you build can be written to a file.

```python
--8<-- "examples/mcp_builder.py"
```

```text
tool: ask_support | None | {'openWorldHint': True}
tool: lookup_order | Look up an order | {'readOnlyHint': True}
answer: Order A1 shipped yesterday.
```

`builder.spec()` returns the validated `MCPSpec`; `builder.dump("server.toml")` writes it (JSON, TOML or YAML by extension, defaults left out) unless it holds live Python objects (`add_tools(registry)`, an algorithm object), which a file cannot carry; `await builder.acompile()` / `builder.compile()` build the server. `dump_spec`/`load_spec` round-trip: a spec written and read back is equal.

## Precedence

Lowest to highest: the file, then `AAA_MCP__*` environment variables, then command-line flags. Levels are separated by double underscores (`AAA_MCP__SERVER__NAME=desk`, `AAA_MCP__TOOLS__EXCLUDE='["a","b"]'`: lists, numbers and booleans are JSON). Dictionaries merge key by key; lists and scalars are **replaced**, so `--exclude x` means exactly `x`.

Secrets never go in the file: a field that needs one takes an `env:NAME` or `${NAME}` reference (`Secret.parse`), and the value is read when needed and never shown in a `repr`.

## The command line

```bash
aaa mcp serve --config server.toml                        # serve over stdio
aaa mcp serve --source myapp.tools:registry --exclude 'admin_*' --tenant acme
aaa mcp check --config server.toml                        # validate, compile, report what would be exposed
aaa mcp inspect --config server.toml                      # print exactly what a client (and its model) will see
aaa mcp serve myapp.tools:build                           # the older form: a factory returning a registry or MCPToolServer
```

Flags: `--source`, `--include`, `--exclude`, `--tag`, `--prefix`, `--name`, `--title`, `--instructions`, `--tenant`, `--user`, `--expose-approval-tools`, `--max-result-chars`, `--no-import-sources`. `serve` writes its messages to stderr, because stdout carries the protocol.

## Over HTTP, behind aaa-server

```python
from aaa.mcp.config import load_spec
from aaa.mcp.compile import acompile_spec

async def build() -> ServerConfig:
    core = await acompile_spec(load_spec("server.toml"), engine=engine)
    return ServerConfig(engine=engine, authenticator=..., mcp=core)
```

`aaa-server` serves the compiled core at `POST /mcp` with the same authentication, rate limits and body limit as everything else; tenant and user come from the principal, not from the file. Titles, annotations, output schemas, structured results, instructions and the server version appear on the wire exactly as over stdio: both are built from one set of wire shapes (`aaa.mcp.wire`).

## Safety

- **Importing runs code.** A source of type `objects` imports your module. Set `AAA_MCP_ALLOW_IMPORT=0` (or `--no-import-sources`) to make any such source an error before anything is imported, and a configuration that imports code is refused if it is writable by everyone.
- Generated servers keep the defaults of [serving tools over MCP](mcp.md#serving-your-tools-over-mcp): failures never leak internals, arguments are validated by the tool's schema, results are capped.
- The full list of threats, including those of the sources that are not built yet, is the [threat model](../security/mcp-generator.md).

## Verified, and not

Tested: configuration loading and precedence (JSON, TOML, YAML, environment, flags), selection/naming/override rules and their error messages, the kill-switch, structured output including the non-conforming and oversized cases, and that the official SDK server, the stdio subprocess and the HTTP handler all describe tools identically. The suite passes on the `mcp` SDK 1.21.2, 1.30 and 2.x. **Not verified:** real MCP hosts (Claude Desktop, IDEs). **The `mcp` extra now requires 1.21.2 or newer**: older 1.x releases no longer import with current pydantic.
