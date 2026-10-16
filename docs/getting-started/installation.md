# Installation

AAA requires **Python 3.11 or newer** and is fully asynchronous (`asyncio`).

## From PyPI

```bash
pip install autonomous-agent-algos            # or: uv add autonomous-agent-algos
pip install "autonomous-agent-algos[postgres,redis,mcp]"   # with optional extras (table below)
```

## From source

To work on the library itself, install from the repository:

=== "uv"

    ```bash
    git clone https://github.com/innerkorehq/AutonomousAgentAlgos.git
    cd AutonomousAgentAlgos
    uv sync
    ```

=== "pip"

    ```bash
    pip install "git+https://github.com/innerkorehq/AutonomousAgentAlgos.git"
    ```

The core depends only on `pydantic`, `anyio` and `httpx`. Everything else is an optional extra:

| Extra | Installs | You need it for |
|---|---|---|
| `postgres` | `asyncpg` | `PostgresEventStore` and its bus backend |
| `redis` | `redis` | `RedisEventStore` and its bus backend |
| `litellm` | `litellm` | the in-process LiteLLM adapter (`provider="litellm"`) |
| `mem0` | `mem0ai` | `Mem0Memory` ([memory](../guides/memory.md)) |
| `mcp` | `mcp` | `MCPClient`: use any MCP server's tools ([MCP](../guides/mcp.md)) |
| `graphiti` | `graphiti-core` | `GraphitiMemory` (also needs Neo4j, FalkorDB or Neptune) |
| `langgraph` / `langmem` | `langgraph` / `langmem` | `LangGraphStoreMemory`, and LangMem's tools and managers on the same store |
| *(separate package)* `aaa-server` | `aaa-server` (extras `jwt`, `redis`) | the [HTTP server](../server/index.md) ([PyPI](https://pypi.org/project/aaa-server/)): `pip install "aaa-server[jwt,redis]"` |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings` | building this documentation |

```bash
pip install "autonomous-agent-algos[postgres,redis] @ git+https://github.com/innerkorehq/AutonomousAgentAlgos.git"
```

SQLite persistence uses only the standard library, so it needs no extra.

## Verify the install

Run the quickstart. It needs no API key and no network:

```bash
python examples/quickstart.py
```

```text
success - 17 x 23 = 391
```

## Running the test suite

```bash
uv sync                      # dev dependencies include asyncpg and redis
uv run pytest -q
```

The Postgres and Redis tests start **throwaway servers** on random ports in temporary directories, so they never
touch your own databases. They need the `initdb`/`postgres` and `redis-server` binaries. If those are missing the
corresponding tests are skipped; to use servers you already run, set `AAA_TEST_POSTGRES_URL` and/or
`AAA_TEST_REDIS_URL`. On macOS with Homebrew the Postgres binaries live under `/opt/homebrew/opt/postgresql@N/bin`;
point `AAA_TEST_PG_BIN` there if they are not on your `PATH`.

## Building these docs locally

```bash
uv sync --extra docs
uv run mkdocs serve          # http://127.0.0.1:8000
```
