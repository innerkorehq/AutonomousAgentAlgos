# aaa-server

HTTP/SSE server for [AAA](https://github.com/innerkorehq/AutonomousAgentAlgos) agents ([`autonomous-agent-algos`](https://pypi.org/project/autonomous-agent-algos/)): bearer authentication, tenants taken only from the credential, rate limits and quotas, server-side budgets, human approvals over HTTP, resumable event streams, plus MCP and (experimental) A2A endpoints.

**Documentation:** [overview](https://innerkorehq.github.io/AutonomousAgentAlgos/server/) · [server guide](https://innerkorehq.github.io/AutonomousAgentAlgos/guides/server/) · [threat model](https://innerkorehq.github.io/AutonomousAgentAlgos/security/server/) · [API reference](https://innerkorehq.github.io/AutonomousAgentAlgos/reference/api/server/) · [library docs](https://innerkorehq.github.io/AutonomousAgentAlgos/) · [changelog](https://innerkorehq.github.io/AutonomousAgentAlgos/about/changelog/)

```bash
pip install aaa-server                  # needs autonomous-agent-algos>=0.3.0 (installed automatically)
pip install "aaa-server[jwt,redis]"    # + JWT/JWKS auth, + the shared Redis rate limiter
```

```python
# myapp/server.py
from aaa.algos import ToolLoop
from aaa.runtime import Budget, Engine
from aaa.tools import ToolRegistry
from aaa_server import ApiKeyAuthenticator, ServerConfig

async def build() -> ServerConfig:
    engine = Engine()   # + attach your LLM and tool handlers; use Postgres in production
    return ServerConfig(
        engine=engine,
        authenticator=ApiKeyAuthenticator.from_keys({"s3cret-key": ("acme", "ann")}),
        algorithms={"agent": ToolLoop(ToolRegistry([]))},
        budget=Budget(max_steps=20, deadline_s=60),
    )
```

```bash
aaa-server serve myapp.server:build     # 127.0.0.1:8000 by default
```

Read the [threat model](https://innerkorehq.github.io/AutonomousAgentAlgos/security/server/) before exposing it: put a TLS-terminating reverse proxy in front.
Status: **alpha (0.1)**. MIT licensed. Source: <https://github.com/innerkorehq/AutonomousAgentAlgos/tree/master/packages/aaa-server>.
