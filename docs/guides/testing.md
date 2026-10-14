# Testing your agents

Agent code is hard to test when it depends on a live model and on timing. AAA is built so you can test it deterministically,
fast, with no network.

## 1. A scripted model

`FakeLLMClient` stands in for the LLM. Give it a function of the request, a list of responses (exceptions in the list are
raised, so error paths are testable), and it records every request it saw.

```python
from aaa.llm import FakeLLMClient
from aaa.llm.fake import text, tool_call


def model(request):  # decide from the conversation, like a real model would
    last = request.messages[-1]
    return tool_call("lookup", id=7) if last.role == "user" else text(f"found {last.content}")


llm = FakeLLMClient(model)
...
assert llm.requests[1].messages[-1].role == "tool"  # what the model was shown on its second turn
```

```python
FakeLLMClient([text("a"), RateLimited("slow down"), text("b")])  # scripted sequence with a failure
```

Decide from the *content* of the request rather than the call count when anything runs in parallel (plan steps,
`SelfConsistency` samples), because call order across concurrent runs is not fixed.

## 2. A full run in a test

```python
import pytest
from aaa.algos import ToolLoop
from aaa.runtime import Engine

...


async def test_agent_looks_things_up():
    registry = ToolRegistry([lookup])
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        run = await engine.start(ToolLoop(registry), {"task": "find 7"})
        outcome = await run.result()
        log = await engine.store.read(run.run_id)

    assert outcome.ok and "found" in outcome.value["text"]
    assert [e.type for e in log].count("aaa.tool.completed") == 1
```

Assert on the **event list**, not only the answer: it tells you *how* the agent got there (which tools, how many LLM calls, whether
an approval was requested).

## 3. Record once, replay forever

`ReplayLLMClient(events)` serves the answers recorded in a run's log, in order. Record a real run (or a scripted one), keep
the events, and replay it as a golden test with no network and no scripting.

```python
--8<-- "examples/testing_replay.py"
```

```text
replayed identically: 5
```

## 4. Testing tools

Tools are plain functions plus validation, so call them directly; errors are exactly what the model would see:

```python
assert await multiply.invoke({"a": 6, "b": 7}) == 42
with pytest.raises(ToolArgsError):
    await multiply.invoke({"a": "six"})
```

## 5. Testing failure and recovery

Use a durable store and `engine.abort()` to simulate a crash, then recover:

```python
engine = await Engine(store=SQLiteEventStore(path)).__aenter__()
...
await engine.abort()  # nothing drained, nothing released
engine2 = await Engine(store=SQLiteEventStore(path)).__aenter__()
handles = await engine2.recover([algo])
```

[`examples/persistence_recovery.py`](persistence.md#recovering-after-a-crash) is a complete worked example. For cluster
behavior, `tests/unit/test_multiprocess.py` shows real-process `SIGKILL` tests.

## 6. Time

Deadlines, tool timeouts and lease TTLs use real time. In tests set them small (`Budget(deadline_s=0.2)`,
`@tool(timeout_s=0.05)`, `GuardConfig(lease_ttl_s=1.0, heartbeat_s=0.2)`) rather than waiting.

## 7. Verifying against a real model

A scripted model proves your wiring; it cannot prove the model behaves. Run `scripts/live_smoke.py` against your gateway
([how](../about/limitations.md#verify-a-hosted-gateway-yourself)), and keep a small set of live evals for behavior you care about.

## pytest setup

AAA's own suite uses the `anyio` pytest plugin in auto mode:

```toml
[tool.pytest.ini_options]
anyio_mode = "auto"
```

`pytest-asyncio` works equally well. Nothing in AAA requires a specific plugin.

## How these docs stay correct

Every example in these docs lives in [`examples/`](https://github.com/innerkorehq/AutonomousAgentAlgos/tree/master/examples) and is
**executed by the test suite**, with its output asserted (`tests/unit/test_examples.py`). If an API change breaks a documented
example, the build fails.
