# Threat model: the HTTP server (`aaa-server`)

Written before the code (milestone M6). The server puts an engine behind a network port. The engine already isolates tenants (`TenantScope`), validates tool arguments, enforces budgets
and asks for approval; the server's job is to make sure **only the right caller reaches those mechanisms, as the right tenant, and cannot use the port to hurt the host or other tenants.**

## Trust boundaries

| Comes from | Trusted? |
|---|---|
| The authenticated **principal** (tenant, subject, scopes) | Yes: it is the only source of tenant and user |
| Request body, query string and headers | **No**: validated, size-limited, and never allowed to name a tenant, user or metadata |
| The model's output and tool results | No (as everywhere in AAA) |
| The event log | Yes, but only a filtered view of it leaves the process |

## Threats and decisions

| # | Threat | Decision |
|---|---|---|
| 1 | **Authentication bypass or timing leaks.** | API keys are stored as SHA-256 digests and compared in constant time; a missing, malformed or unknown credential gives the same `401` with a `WWW-Authenticate: Bearer` header and no hint of which part failed. Keys and `Authorization` headers are never logged. |
| 2 | **Cross-tenant access (IDOR).** Guessing or reusing another tenant's run or session id. | Every operation goes through `TenantScope` / `TenantSessions` bound to the principal's tenant. A foreign id and a nonexistent id both return the **same `404`**, so existence does not leak. A test sends every endpoint a foreign id. |
| 3 | **Tenant or user spoofing** through the body (`{"tenant": "acme"}`), a header (`X-Tenant`) or metadata. | Request models reject unknown fields (`422`). Tenant, user and the run's trusted `meta` come only from the principal. Tests post spoofed fields to every endpoint. |
| 4 | **JWT pitfalls**: `alg: none`, HS/RS confusion, missing `exp`, key-id spam, a token naming its own key URL. | The accepted algorithms are an explicit allow-list set by the operator, never read from the token; `exp`, `iss` and `aud` are required; `jku`/`x5u` headers are ignored (the JWKS URL is fixed by configuration); an unknown `kid` triggers at most one JWKS refresh per interval. |
| 5 | **Resource exhaustion**: huge bodies, floods, endless runs, many open streams. | A body size limit (1 MiB default) enforced before parsing; a token-bucket rate limit per principal; a cap on concurrent runs and open streams per tenant; **budgets are server-side**: the server sets a default `Budget` per tenant and a request may only lower it, never raise it. Slow-client protection (header/body timeouts, TLS) belongs to the reverse proxy in front: the docs say so. |
| 6 | **Approval abuse.** Approving a call that was never requested (pre-signing), or approving without the right. | The approvals endpoint needs the `approve` scope and only accepts a `call_id` for which an approval request is **currently pending** in that tenant's run. |
| 7 | **Information disclosure through events.** The raw log holds the operator's system prompt, memory contents and guard internals. | The stream is a **projection**, not the log: by default only run lifecycle, token deltas, tool names/outcomes, approval requests and session replies leave, with payloads trimmed. Showing more is an explicit operator setting. Errors are generic with a request id; no stack traces. |
| 8 | **Idempotency-Key abuse.** Replaying a key with a different request, or using keys to collide with others. | Keys are scoped to the tenant (run ids are namespaced), limited in length and charset, and a repeat with a different request body is `409`, not a silent return of the old run. |
| 9 | **CSRF and CORS.** | There are no cookies or ambient credentials; only `Authorization: Bearer`. CORS is off unless the operator lists origins explicitly (never a wildcard together with credentials). |
| 10 | **Long-lived streams outliving their credentials.** | Authentication is checked when a stream opens; a stream ends after a maximum duration (default 15 minutes) and the client reconnects with `Last-Event-ID`, which re-authenticates. A client disconnect stops the stream. |
| 11 | **Quota bypass by racing** the check against the start. | Documented: quotas are checked before a run starts and recorded when it completes, so concurrent starts can overshoot by at most the concurrency cap. Per-run budgets are the hard limit. |
| 12 | **Information at rest in the process**: unauthenticated endpoints. | Only `/healthz`, which returns `{"ok": true}` and no version or configuration. |
| 13 | **Binding to the world by accident.** | `aaa-server serve` listens on `127.0.0.1` by default; exposing it is a flag you type. TLS terminates at a reverse proxy (documented), not in the library. |

## Out of scope

Authenticating end users (OIDC login flows, sessions in browsers), TLS termination, DDoS protection, WAF rules, and what the algorithms and tools themselves do once a run starts
(see the tool, guardrail and multimodal threat models).
