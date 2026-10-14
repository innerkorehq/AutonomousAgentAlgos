# Security

AAA runs model-driven code paths next to your systems. This page lists what it protects, what it leaves to you, and the
assumptions behind each.

## What is in the event log

The log is a complete record of each run: **prompts, model outputs, tool arguments and tool results**. If a customer's
personal data, an API response or a secret passes through a tool, it is in the log (and in snapshots and, if enabled, blobs).

- Treat the event store and blob store as **sensitive data stores**: encryption at rest, network isolation, least-privilege
  credentials, backups protected like the data.
- `Engine(redactor=...)` scrubs events **before** they are persisted or delivered ([Redaction](../guides/privacy.md#redaction)).
  It is a pattern-based safety net, not a guarantee; keep sensitive data out of events where you can.
- [`engine.erase`, `purge` and retention policies](../guides/privacy.md#erasure) remove finished runs (payloads, snapshots, blobs, outbox
  copies). Nothing is deleted unless you configure it. They do not reach your own backups or logs.
- Blobs and the event store are **not encrypted** by AAA; use encryption at rest.

## Secrets

- LLM keys are referenced as `${ENV_VAR}` in `LLMConfig` and resolved when the client is built. They are never written to events.
  Do not put literal keys in config files you commit.
- A key a *tool* returns, or one the user pastes into a prompt, will be in the log.
- Logs (`aaa.*` loggers) include exception tracebacks, which may include argument values.

## Tools: the real attack surface

A tool runs with **your process's privileges**. Prompt injection, where text the agent reads (a web page, an email, a document)
tries to steer the model, is the central risk, and AAA's controls are structural, not magical:

| Control | What it gives you |
|---|---|
| Argument validation | The model cannot pass unknown or mistyped arguments; Pydantic rejects them before your code runs. |
| `tool_names` / small registries | The model can only call what you exposed. Least privilege per algorithm. |
| `requires_approval=True` | A human sees the exact call. Enforced outside the model. |
| `delivery="at_most_once"` | Irreversible actions never repeat automatically. |
| Budgets, deadlines, `max_turns` | A manipulated model cannot run unboundedly or spend without limit. |
| Timeouts and pools | One slow or malicious dependency cannot exhaust the worker. |

Design tools so a successful injection is **bounded in damage**: read-only by default, narrow scopes, per-user credentials rather
than a shared admin token, and approval for anything that writes, sends or deletes. Treat tool *results* as untrusted input.

!!! danger "AAA does not sandbox code"
    If a tool or validator executes model-generated code, run it in an isolated sandbox you control. The library ships none.

## Authorization of events

Events are plain data. **Anything that can append to a run's stream can approve a tool call, cancel the run, or drive an FSM.**
AAA does not authenticate the appender. Therefore:

- Restrict database/Redis write access to your services.
- Put authentication and authorization in front of any endpoint that signals runs ("approve", "cancel").
- Validate the signed-in user may act on that run and that `call_id` before calling `run.signal(...)`.

## Multi-tenancy

Events carry a `tenant`, and [`engine.tenant(name)`](../guides/privacy.md#tenancy) gives a scope that namespaces run ids and
checks ownership on every lookup, so one customer's code cannot read or erase another's runs through it. That is *application-level*
isolation: all tenants still share one store and the scope trusts the tenant name you pass. For hard isolation use **separate databases** (Postgres), **separate key prefixes or instances** (Redis `prefix=`), and separate
blob roots per tenant, and give each tenant's workers only their own credentials.

## Network and gateways

- Use HTTPS to gateways. `LLMConfig.base_url` is taken at face value; do not build it from untrusted input.
- A gateway sees all prompts and completions. Choose and secure it accordingly.
- Preset headers (`x-portkey-api-key`, `cf-aig-authorization`, ...) are sent on every request.

## Dependencies

The core depends on `pydantic`, `anyio` and `httpx`. Postgres, Redis and LiteLLM support are optional extras. Pin and scan dependencies as
you do for any service.

## Reporting a vulnerability

Please do not file a public issue with exploit details. Use GitHub's **private vulnerability reporting** (repository *Security* tab,
*Report a vulnerability*) if it is enabled. If it is not, open an issue titled "Security contact request" **without details** and a maintainer
will arrange a private channel.
