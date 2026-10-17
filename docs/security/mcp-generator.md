# Threat model: the MCP server generator

Written before the code. The generator turns a **configuration** into an MCP server: it can import your modules, call HTTP APIs described by an OpenAPI document, read files and indexes, and run agents, all on behalf of an **MCP client** (and, behind [`aaa-server`](../server/index.md), a remote, authenticated one). The configuration is the new trusted input; what it points at is not.

## Trust boundaries

| Input | Trusted? |
|---|---|
| The **config file**, builder code, CLI flags | Yes, but it is an executable-ish input: it chooses what is imported and fetched. Treated as operator-authored; hardened against accidental injection (T3). |
| **OpenAPI documents** and the descriptions in them | **No.** Their text is shown to a model (T7) and their `servers` URL tells the server where to connect (T1). |
| **Modules named in `module` sources** | The operator's own code, but importing runs it (T2). |
| **MCP client requests** (tool arguments, resource URIs, prompt arguments) | **No**: validated against schemas; tenant and user never come from them. |
| **Upstream API responses, file contents, index documents** | **No**: returned to the client as data, size-capped. |

## Threats and decisions

| # | Threat | Decision | Ships in |
|---|---|---|---|
| T1 | **OpenAPI SSRF and credential forwarding.** A spec (or a redirect) points the server at internal addresses, or leaks the API credential to another host. | A non-empty `allowed_hosts` is mandatory; every call goes through the SSRF-safe fetcher (resolve once, all addresses public, connection pinned, redirects re-checked). Credentials come only from `env:` references and are attached **only** to the configured server's host, never after a redirect; redirects to another host are refused. `Authorization`, `Cookie` and `Host` can never be set from tool arguments. Private networks need an explicit opt-in that logs a warning. | M5 (shipped) |
| T2 | **Arbitrary module import.** A config names a module whose import has side effects, or an attacker-writable config makes the server import attacker code. | `module` sources need an explicit module and allow-list (default `__all__`, never `_names`, only functions defined in that module). Introspection never calls the functions. A kill-switch (`--no-import-sources`, `AAA_MCP_ALLOW_IMPORT=0`) turns every importing source into a startup error, and a config file writable by others is refused. Module tools require approval unless the operator opts out per tool. | M1 (objects), M4 |
| T3 | **Config injection.** YAML tags that construct objects, `${ENV}` expansion in unexpected fields, template injection into generated files. | `yaml.safe_load` only; the spec models forbid unknown keys; environment references are expanded only in the fields that declare them; generated files never interpolate user text into Python source (names are validated as slugs; YAML is written from the validated model). | M1, M6 |
| T4 | **Secrets in files, logs and errors.** | Secrets exist only as `env:NAME` references; the resolved value lives in a type whose `repr` is redacted; `check`, `inspect` and `export` print reference names only; a scaffold test plants a canary secret and searches the generated tree. A missing variable is a startup error, not a runtime one. | M1, M6 |
| T5 | **Resource exfiltration.** Path traversal or symlinks out of a files root, one tenant reading another's documents or memory, URI-template variables smuggling paths. | File resources use the confined `FileAccess`; index and memory providers filter by the call's tenant; file roots are per-tenant unless `shared: true` is set explicitly; template variables must match a strict pattern; an unknown resource and a forbidden one are indistinguishable. | M3 |
| T6 | **Cost amplification.** An agent tool, a huge upstream response or a fan-out of calls. | Every agent tool needs an explicit budget; a global cap on concurrent agent runs; response and request byte caps and timeouts; `max_tools`; per-source concurrency limits. | M2, M5 |
| T7 | **Prompt injection through descriptions.** An OpenAPI `description` is shown to the model as a tool description. | Descriptions are length-capped and treated as untrusted; `description_override` replaces them; `aaa mcp inspect` prints exactly what the model will see so an operator can review it. Not fully solvable: a spec you do not control puts its words in your model's context. | M5 (shipped) |
| T8 | **Approval bypass by defaults.** Exposing every operation of an API makes destructive calls available. | Non-GET operations and module functions default to `requires_approval`, which keeps them hidden (indistinguishable from unknown tools) unless the operator opts in; a startup banner lists every state-changing operation exposed without approval. | M4, M5 |
| T9 | **Enumeration through errors.** Distinguishing "forbidden" from "missing" resources, tools or prompts. | The same response for both. | M1, M3 |

## Also decided

- **Protocol floor.** The generated server supports the MCP SDK 1.x line from **1.21.2** (older 1.x releases no longer import with current pydantic) and 2.x; the stdio and HTTP paths are tested against both.
- **No new log fields.** The generator adds nothing to the event log; the frozen v0.2.0 fixtures stay green.
- **Out of scope:** authenticating MCP clients on stdio (the process boundary is the trust boundary), OAuth for upstream APIs, and what the exposed tools themselves do once called (see the tool, guardrail and server threat models).
