# Long-term memory

A run's event log remembers *that run*. **Memory** is what an agent knows across runs: a user's preferences, decisions made last
week, facts about a customer. AAA does not reinvent that layer. Libraries such as **Mem0**, **Graphiti** (the engine behind Zep),
**LangMem** and **Supermemory** already extract, consolidate and index memories. AAA gives them one small interface, wires them
into the event log, tenancy and erasure, and lets you plug in any other library with a few lines.

```python
--8<-- "examples/memory_agent.py"
```

```text
first run stored: ['Ann is vegetarian and lives in Lisbon']
ann, new run: A vegetarian place in Lisbon: Ao 26.
bob, same tenant: Which cuisine do you like?
ann at another tenant: Which cuisine do you like?
forgot: {'NativeMemory': 1}
ann after forgetting: Which cuisine do you like?
```

## The pieces

| Piece | What it is |
|---|---|
| `Memory` protocol | `add(entries, scope)` and `search(query, scope)`. Everything else is an optional capability: `Updatable`, `Deletable`, `Forgettable`, `Profiled`, `Listable`. |
| `Scope` | *Whose* memory: `tenant`, `user`, `agent`, `session`, `namespace`. Each adapter maps it to its library's own notion. |
| Backends | `NativeMemory` (built in, offline), `Mem0Memory`, `GraphitiMemory`, `SupermemoryMemory`, `LangGraphStoreMemory` (LangMem), `CallableMemory` (anything else), `FakeMemory` (tests). |
| `MemoryHandler` | Serves `aaa.memory.requested` events: recall, remember, forget. Optional **write-back** when a run finishes. |
| `memory_tools(memory)` | `remember`, `recall` (and opt-in `forget`) tools the model can call. |
| `ToolLoop(recall=True)` | Looks memories up *before* the first model call and puts them in the prompt. |

## Using it

### Automatic recall

```python
memory = Mem0Memory(AsyncMemory())
MemoryHandler(memory).attach(engine)
algo = ToolLoop(registry, recall=True)            # or RecallConfig(k=8, max_chars=1500)
await engine.tenant("acme").run(algo, {"task": "..."}, user="ann")
```

The loop emits an `aaa.memory.requested` event, waits for `aaa.memory.completed`, and sends the first model request with the
recalled facts as a system message. If the memory backend fails or is slow, the run **continues without memories**: a failing
memory service never fails a run.

### Remembering

Three ways, which combine:

- **The model decides** (hot path): give it `memory_tools(memory)`. It calls `remember("Ann prefers aisle seats")` when it learns something.
- **Write-back** (background): `MemoryHandler(memory, write_back=WriteBack())` stores each successful run's task and answer when the run
  finishes. Pass `WriteBack(build=...)` to store something smarter (a summary, extracted facts).
- **Your code**: `await memory.add([MemoryEntry(text=...)], Scope(tenant="acme", user="ann"))`.

### Whose memory: scope comes from the run, never from the model

`MemoryHandler`, the tools and automatic recall build the scope from the run's **tenant** and its **metadata**
(`start(..., user="ann")` on a tenant scope, or `meta={"user": ..., "agent": ...}` on the engine). The model never supplies a scope:
the `remember` and `recall` tools have no tenant or user parameter, so a prompt injection cannot read another customer's memories.
Use opaque ids for `user` (not e-mail addresses): the redactor protects the `user` and `agent` metadata keys, but your own logs may not.

## Why events

Memory calls go through the log (`aaa.memory.requested` → `aaa.memory.completed` / `failed`) instead of being a library call inside your
algorithm. That buys:

- **Replay**: the recalled items are recorded, so replaying a run shows what the original saw even after the memory changed.
- **Privacy**: requests and results pass through the engine's [redactor](privacy.md#redaction) before storage, and what reaches the
  provider is what was stored.
- **Crash safety**: a backend that cannot de-duplicate (`idempotent = False`, e.g. Mem0) gets `at_most_once` writes. The handler records an
  attempt marker first; a worker that died mid-write is reported as `OutcomeUnknown` instead of silently writing twice. Backends with
  idempotency keys (Graphiti episode `uuid`, Supermemory `customId`, `NativeMemory`, the LangGraph store) are retried freely.
- **Spend**: memory events that report `usage`/`cost` count towards the run's budget ledger.

## Backends

All four are optional extras, imported lazily: `pip install "autonomous-agent-architecture[mem0]"` (also `graphiti`, `langgraph`, `langmem`).
Supermemory needs none: it is plain HTTP.

=== "Mem0"

    ```python
    from mem0 import AsyncMemory            # open source; or AsyncMemoryClient for the hosted platform
    from aaa.memory.mem0 import Mem0Memory

    memory = Mem0Memory(AsyncMemory())      # configure Mem0's own LLM, embedder and vector store on it
    ```

    Mem0 extracts facts and decides ADD/UPDATE/DELETE with **its own** LLM, so that spend is outside AAA's budget and gateway. It has no
    idempotency key, so writes are `at_most_once`. No point-in-time recall (`as_of` is ignored). Entity ids are tenant-qualified
    (`Scope.slug()`); tenant-wide operations use Mem0's wildcard entity filter, the least-exercised path: test
    `forget(Scope(tenant=...))` against your Mem0 version. Written against `mem0ai` 2.x, where identity travels inside `filters`.

=== "Graphiti / Zep"

    ```python
    from graphiti_core import Graphiti
    from aaa.memory.graphiti import GraphitiMemory

    g = Graphiti("bolt://localhost:7687", "neo4j", "password")  # Neo4j, FalkorDB or Neptune
    await g.build_indices_and_constraints()
    memory = GraphitiMemory(g)
    ```

    A temporal knowledge graph: facts carry `valid_at`/`invalid_at`, so `search(..., as_of=date)` answers "what was true then", and by
    default only currently-valid facts are returned. One Graphiti `group_id` per scope; tenant-wide scopes expand to every group of the
    tenant. Writes use a deterministic episode `uuid`. You configure the graph database, LLM (structured output required) and embedder.
    No `update`: facts are derived from episodes.

=== "Supermemory"

    ```python
    from aaa.memory.supermemory import SupermemoryMemory

    memory = SupermemoryMemory(api_key=os.environ["SUPERMEMORY_API_KEY"])
    memory = SupermemoryMemory(api_key="...", base_url="http://localhost:6767")   # self-hosted
    ```

    `ingest="documents"` (default) lets Supermemory extract memories from content, asynchronously (`AddResult.pending`, or `wait=True`
    to poll); `customId` makes it idempotent. `ingest="memories"` stores text verbatim, synchronously. One `containerTag` per scope;
    `profile()` returns the static and dynamic profile. Tenant-wide `forget` deletes container tags, which the API restricts to
    organisation owners/admins. The v3/v4 API is announced to shut down on 2026-12-31; all paths live in one dict (`PATHS`).

=== "LangMem / LangGraph"

    ```python
    from langgraph.store.memory import InMemoryStore
    from aaa.memory.langgraph import LangGraphStoreMemory

    store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
    memory = LangGraphStoreMemory(store)    # or a Postgres/Redis/hosted BaseStore

    # let LangMem's own tools and managers use the same memories:
    create_search_memory_tool(namespace=memory.namespace_for(scope), store=store)
    ```

    Any LangGraph `BaseStore` becomes an AAA memory, and LangMem's records are read transparently. Namespaces are hierarchical
    (`(root, tenant, user, agent, ...)`), so a tenant-wide search is a prefix search. Without an embedding index on the store, recall
    ranks by word overlap.

=== "MCP server"

    ```python
    from aaa.mcp import MCPClient, mcp_memory

    memory = mcp_memory(srv, add="add_memory", search="search_memory")   # tool names from `srv.list_tools()`
    ```

    Any memory service with an MCP server works without an adapter; see [MCP](mcp.md#mcp-memory-servers).

=== "Anything else"

    ```python
    from aaa.memory import CallableMemory

    memory = CallableMemory(
        add=lambda entries, scope: lib.store([e.text for e in entries], user=scope.user),
        search=lambda query, scope, k: lib.find(query, user=scope.user, limit=k),
        forget=lambda scope: lib.delete_user(scope.user),     # optional: switches on erasure support
    )
    ```

    Functions may be sync or async and may return loose shapes (strings, dicts with `text`/`memory`/`fact`/`content`). Each optional
    function turns on its capability, so `isinstance(memory, Forgettable)` is true exactly when you supplied `forget`.
    Set `idempotent=True` only if your `add` honours the idempotency `key` it is given.

!!! note "`NativeMemory` is keyword search unless you give it an embedder"
    `NativeMemory()` finds memories that **share a word** with the query. For semantic recall pass
    `NativeMemory(embedder=LLMEmbedder(client))` (hybrid keyword + vector), or use one of the services above. It is in-process, so use it
    for development, tests and single-process apps. `llm_extractor(client)` makes it store distilled facts instead of raw text; that one
    LLM call is not routed through the engine's `LLMHandler`, so it is not in the run's budget.

## Erasure and right to be forgotten

Memories are knowledge about a *person*, not part of one run, so `engine.erase(run_id)` does not remove them. Use the memory side:

```python
await engine.tenant("acme").forget_memory(user="ann")   # one user, from every registered backend
await engine.tenant("acme").erase_all()                 # runs AND (by default) the tenant's memories
```

`MemoryHandler(...).attach(engine)` registers the backend (or pass `Engine(memory=[...])`). The report lists what each backend removed
and which backends *cannot* forget (`unsupported`), so a gap is visible rather than silent. See [Tenancy, redaction, retention](privacy.md#erasure).

## What is and is not verified

- All backends pass one shared contract suite (scoping between tenants and users, idempotent writes, delete/forget, point-in-time
  recall where supported) against faithful stand-ins.
- The calls each adapter makes are replayed against the **real** `Mem0`, `Graphiti` and `LangGraph` signatures in CI (an `adapters`
  job that also runs weekly against the newest releases), and the LangGraph adapter runs on LangGraph's real `InMemoryStore`.
- Supermemory is checked against its published OpenAPI spec, not a live account. Nothing here has been run against a live Mem0,
  Graphiti or Supermemory service: these libraries change quickly and their docs disagree with each other, so treat a first run against
  your own deployment as part of adopting an adapter.
