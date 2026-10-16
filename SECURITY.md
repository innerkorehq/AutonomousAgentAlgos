# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private vulnerability reporting:
<https://github.com/innerkorehq/AutonomousAgentArchitecture/security/advisories/new>

Include what you found, how to reproduce it, and the version. You will get an acknowledgement within a few working days. A fix is released as a
patch version of the latest minor release, and the advisory credits you unless you prefer otherwise.

## Supported versions

Only the **latest minor release** receives security fixes (the project is pre-1.0; see the [stability policy](docs/policy/stability.md)).

## What counts

In scope: tenant isolation or erasure failures, secrets or personal data written to the log or exported in traces when they should not be,
guardrail or approval bypasses that the documentation says are enforced, and unsafe defaults in shipped tools.

Not in scope: the documented limits (for example, guardrails reduce risk and are not a security boundary; the subprocess sandbox is dev-only
and not isolation; hosted-service adapters send text to that service by design). See [operations / security](docs/operations/security.md).
