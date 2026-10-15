# Stability and deprecation policy

AAA is **pre-1.0**. This page says what that means in practice, so you can decide what to depend on.

## Versioning

`MAJOR.MINOR.PATCH`, interpreted like this while the major version is 0:

- **Patch** (0.2.0 -> 0.2.1): bug fixes and security fixes. No API changes, no log-format changes.
- **Minor** (0.2 -> 0.3): new features, and the only place a **deprecation may complete** (a removal) or a stable API may change incompatibly. Every such change is called out under
  *Breaking* in the [changelog](../about/changelog.md).
- **1.0** will mean the "stable" modules below follow strict semantic versioning.

## What is stable and what is experimental

| Status | Modules |
|---|---|
| **Stable** (changes are deprecated first) | `aaa.runtime` (Engine, RunHandle, Budget, Ledger), `aaa.events` (Event, stores, buses, Projection), `aaa.core` (Outcome, errors), `aaa.llm` types and the `LLMClient` protocols, `aaa.tools` (Tool, ToolRegistry, ToolHandler), `aaa.algos` public classes, `aaa.session`, `aaa.memory` protocols, `aaa.vector` protocols and filters, `aaa.guard` protocols and policy, `aaa.eval` core, `aaa.observe` core |
| **Provisional** (may change in a minor release with a changelog note) | adapters for third-party libraries and services (they follow those projects), `aaa.score`, `aaa.safe`, `aaa.trace` attribute names (the OpenTelemetry GenAI conventions are themselves unstable) |
| **Experimental** (may change or disappear in any release) | anything the docs mark experimental, including A2A support once it ships |

Names starting with an underscore, and anything not documented in the API reference, are private.

## The event log

The log is your data, so its format is held to a stricter rule than the code:

- **Additive only.** New event types and new optional fields with defaults may appear in any minor release. A field is never removed or retyped, and an event type is never reused for something else.
- Code that reads the log (every `Projection`, every algorithm) must tolerate every shape the log has ever held. Stores never rewrite or upcast stored events.
- A `Projection` whose state shape changes bumps its `version`; old snapshots are then ignored and rebuilt from the log.
- A change that would break this rule is made as a **new event type** and the old one keeps being read.

## Deprecation

- A deprecated API keeps working and emits a `DeprecationWarning` (see `aaa._compat.deprecated`) that names the replacement and the version it will be removed in.
- It stays for **at least two minor releases** after the release that deprecates it, and the changelog lists it under *Deprecated* when it is deprecated and *Breaking* when it is removed.
- Security fixes may shorten this when keeping the old behaviour would leave users exposed.

## Python and dependency support

Supported Python versions are those in CI (3.11, 3.12, 3.13). A Python version is dropped only in a minor release, announced in the changelog one release earlier.
Optional third-party integrations are tested against the newest release of each library every week; when one breaks, the adapter is fixed or its supported range is
narrowed in the next patch release.
