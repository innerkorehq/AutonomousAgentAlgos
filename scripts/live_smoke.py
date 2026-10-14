"""Live gateway smoke test: does AAA actually work against a real OpenAI-compatible endpoint?

    AAA_LIVE_BASE_URL=http://127.0.0.1:11434/v1  AAA_LIVE_MODEL=gemma4:latest  \\
    [AAA_LIVE_API_KEY=...] [AAA_LIVE_HEADERS='{"x-portkey-api-key": "..."}'] \\
    [AAA_LIVE_CHAT_PATH=/chat/completions] [AAA_LIVE_EMBED_MODEL=...] \\
        uv run python scripts/live_smoke.py

Works for any gateway preset in aaa.llm.gateways (set the base URL/headers they document).
Each check prints PASS / FAIL / SKIP with latency; exit code is non-zero if anything FAILED.
Costs real tokens (a handful of tiny requests)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback

from aaa.algos import ToolLoop
from aaa.llm import (
    LLMConfig,
    LLMHandler,
    LLMRequest,
    Message,
    OpenAICompatClient,
    ToolSpec,
    collect,
)
from aaa.llm import errors as E
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry, tool

BASE = os.environ.get("AAA_LIVE_BASE_URL", "")
MODEL = os.environ.get("AAA_LIVE_MODEL", "")
KEY = os.environ.get("AAA_LIVE_API_KEY")
HEADERS = json.loads(os.environ.get("AAA_LIVE_HEADERS", "{}"))
CHAT_PATH = os.environ.get("AAA_LIVE_CHAT_PATH", "/chat/completions")
EMBED_MODEL = os.environ.get("AAA_LIVE_EMBED_MODEL")
TIMEOUT = float(os.environ.get("AAA_LIVE_TIMEOUT", "240"))

results: list[tuple[str, str, float, str]] = []


def cfg(**kw) -> LLMConfig:
    d = {
        "model": MODEL,
        "base_url": BASE,
        "api_key": KEY,
        "headers": HEADERS,
        "chat_path": CHAT_PATH,
        "timeout_s": TIMEOUT,
    }
    d.update(kw)
    return LLMConfig(**d)


def check(name):
    def deco(fn):
        async def run():
            t = time.monotonic()
            try:
                note = await fn()
                results.append(
                    (
                        name,
                        "SKIP" if str(note).startswith("skip:") else "PASS",
                        time.monotonic() - t,
                        str(note or ""),
                    )
                )
            except Exception as e:  # noqa: BLE001 - a smoke test reports, it does not crash
                tb = traceback.format_exc().strip().splitlines()[-1]
                results.append(
                    (name, "FAIL", time.monotonic() - t, f"{type(e).__name__}: {e}"[:300] or tb)
                )
            print(
                f"  {results[-1][1]:4}  {name}  ({results[-1][2]:.1f}s)  {results[-1][3][:160]}",
                flush=True,
            )

        run.__name__ = fn.__name__
        return run

    return deco


@check("plain completion returns text + token usage")
async def t_complete():
    c = OpenAICompatClient(cfg())
    r = await c.complete(
        LLMRequest(
            messages=[Message(role="user", content="Reply with exactly one word: pong")],
            params={"temperature": 0, "max_tokens": 400},
        )
    )
    assert (r.message.content or "").strip(), f"empty content (finish_reason={r.finish_reason})"
    assert r.usage.total_tokens > 0, "gateway returned no usage"
    await c.aclose()
    return (
        f"{(r.message.content or '').strip()[:40]!r} tokens={r.usage.total_tokens} model={r.model}"
    )


@check("streaming yields incremental chunks and the same text")
async def t_stream():
    c = OpenAICompatClient(cfg())
    req = LLMRequest(
        messages=[Message(role="user", content="Count from 1 to 8, separated by commas.")],
        params={"temperature": 0, "max_tokens": 400},
    )
    chunks = [ch async for ch in c.stream(req)]
    text = "".join(ch.text for ch in chunks)
    assert sum(1 for ch in chunks if ch.text) > 1, "text arrived in a single chunk (not streaming?)"
    assert any(d in text for d in ["1", "2", "3"]), f"unexpected text {text[:80]!r}"
    r = await collect(c.stream(req))
    usage = "usage-in-stream" if any(ch.usage for ch in chunks) else "NO usage in stream"
    await c.aclose()
    return f"{len(chunks)} chunks, {usage}, folded={len(r.message.content or '')} chars"


@check("structured output (JSON schema) parses and validates")
async def t_structured():
    from aaa.algos.base import parse_json_obj

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
        "required": ["city", "population_millions"],
        "additionalProperties": False,
    }
    c = OpenAICompatClient(cfg())
    r = await c.complete(
        LLMRequest(
            messages=[
                Message(
                    role="user",
                    content="Give the city Tokyo and its approximate population in millions as JSON.",
                )
            ],
            response_schema=schema,
            params={"temperature": 0, "max_tokens": 400},
        )
    )
    obj = parse_json_obj(r.message.content)
    assert (
        obj
        and isinstance(obj.get("city"), str)
        and isinstance(obj.get("population_millions"), (int, float))
    ), r.message.content
    await c.aclose()
    return json.dumps(obj)[:80]


ADD = ToolSpec(
    name="add",
    description="Add two integers and return the sum.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
)


@check("native tool calling: model emits a well-formed call")
async def t_tool_call():
    c = OpenAICompatClient(cfg())
    r = await c.complete(
        LLMRequest(
            messages=[
                Message(
                    role="user",
                    content="Use the add tool to compute 1234 + 4321. Do not answer without calling it.",
                )
            ],
            tools=[ADD],
            params={"temperature": 0, "max_tokens": 600},
        )
    )
    assert r.message.tool_calls, (
        f"no tool call; content={r.message.content!r} finish={r.finish_reason}"
    )
    call = r.message.tool_calls[0]
    assert call.name == "add" and call.raw_arguments is None, call
    assert call.arguments == {"a": 1234, "b": 4321} or sorted(call.arguments.values()) == [
        1234,
        4321,
    ], call.arguments
    await c.aclose()
    return f"{call.name}({call.arguments}) id={call.id!r}"


@check("end-to-end ToolLoop through Engine + LLMHandler + ToolHandler")
async def t_e2e():
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    reg = ToolRegistry([multiply])
    client = OpenAICompatClient(cfg())
    async with Engine() as eng:
        LLMHandler(client).attach(eng)
        ToolHandler(reg).attach(eng)
        h = await eng.start(
            ToolLoop(
                reg,
                params={"temperature": 0, "max_tokens": 600},
                max_turns=6,
                system="You are precise. Always use the multiply tool for arithmetic, then state the result.",
            ),
            {"task": "What is 17 times 23? Use the tool."},
            budget=Budget(max_steps=30, deadline_s=TIMEOUT),
        )
        out = await h.result()
        log = await eng.store.read(h.run_id)
    await client.aclose()
    used = [e for e in log if e.type == "aaa.tool.completed"]
    assert out.ok, f"{out.kind}: {out.reason}"
    assert used and used[0].data["result"] == 391, "tool was not used correctly"
    assert "391" in out.value["text"], out.value["text"]
    toks = sum(
        e.data.get("usage", {}).get("total_tokens", 0) for e in log if e.type == "aaa.llm.completed"
    )
    return f"answer={out.value['text'].strip()[:60]!r} turns={out.value['turns']} tokens={toks}"


@check("streamed tool call: argument fragments reassemble into valid JSON")
async def t_stream_tool():
    c = OpenAICompatClient(cfg())
    req = LLMRequest(
        messages=[
            Message(
                role="user",
                content="Use the add tool to compute 77 + 25. Do not answer without calling it.",
            )
        ],
        tools=[ADD],
        params={"temperature": 0, "max_tokens": 600},
    )
    r = await collect(c.stream(req))
    assert r.message.tool_calls, f"no tool call in stream; content={r.message.content!r}"
    call = r.message.tool_calls[0]
    assert call.raw_arguments is None and sorted(call.arguments.values()) == [25, 77], (
        call.arguments,
        call.raw_arguments,
    )
    await c.aclose()
    return f"{call.name}({call.arguments})"


@check("PlanAndExecute with a real planner (nested strict schema) and real executor")
async def t_plan():
    from aaa.algos import PlanAndExecute

    @tool
    def square(n: int) -> int:
        """Return n squared."""
        return n * n

    reg = ToolRegistry([square])
    client = OpenAICompatClient(cfg())
    async with Engine() as eng:
        LLMHandler(client, concurrency=2).attach(eng)
        ToolHandler(reg).attach(eng)
        executor = ToolLoop(
            reg,
            params={"temperature": 0, "max_tokens": 600},
            max_turns=5,
            system="Solve the step. Use the square tool for squares. Reply with just the result.",
        )
        algo = PlanAndExecute(executor, max_parallel=1, max_replans=0)
        algo.model = None
        h = await eng.start(
            algo,
            {"task": "Compute the square of 6 and the square of 9, then add the two results."},
            budget=Budget(max_steps=80, deadline_s=TIMEOUT),
        )
        out = await h.result()
        log = await eng.store.read(h.run_id)
    await client.aclose()
    steps = [e for e in log if e.type == "aaa.task.started"]
    assert out.ok, f"{out.kind}: {out.reason}"
    assert len(steps) >= 2, f"planner produced {len(steps)} steps"
    return f"{len(steps)} steps, final={out.value['text'].strip()[:70]!r} (expected 117)"


@check("GenerateTestRepair against a real model with a deterministic validator")
async def t_repair():
    from aaa.algos import GenerateTestRepair
    from aaa.validate import ValidationHandler, Verdict

    def has_three_items(candidate: str, ctx: dict) -> Verdict:
        try:
            items = json.loads(candidate)
        except ValueError as e:
            return Verdict(
                ok=False, feedback=f"not valid JSON ({e}); reply with ONLY a JSON array, no prose"
            )
        ok = isinstance(items, list) and len(items) == 3 and all(isinstance(i, str) for i in items)
        return Verdict(
            ok=ok,
            feedback="" if ok else f"need a JSON array of exactly 3 strings, got {candidate[:80]}",
        )

    client = OpenAICompatClient(cfg())
    async with Engine() as eng:
        LLMHandler(client).attach(eng)
        ValidationHandler({"default": has_three_items}).attach(eng)
        out = await eng.run(
            GenerateTestRepair(params={"temperature": 0, "max_tokens": 400}, max_iterations=3),
            {"task": "List three fruits as a JSON array of strings."},
            budget=Budget(deadline_s=TIMEOUT),
        )
    await client.aclose()
    assert out.ok, f"{out.kind}: {out.reason}"
    return f"{out.value['candidate'][:50]!r} after {out.value['iterations']} attempt(s)"


@check("handler streams aaa.llm.delta events from the real gateway")
async def t_handler_stream():
    class Ask:
        name = "ask"

        def subscriptions(self):
            return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

        async def on_event(self, e, ctx):
            if e.type == "aaa.run.started":
                req = LLMRequest(
                    messages=[Message(role="user", content="Name three primary colors.")],
                    stream=True,
                    params={"temperature": 0, "max_tokens": 300},
                )
                return [
                    e.caused(
                        "aaa.llm.requested", {"request": req.model_dump(mode="json"), "tag": "s"}
                    )
                ]
            return [
                e.caused(
                    "aaa.run.completed",
                    {"kind": "success" if e.type.endswith("completed") else "failed"},
                )
            ]

    client = OpenAICompatClient(cfg())
    async with Engine() as eng:
        LLMHandler(client).attach(eng)
        h = await eng.start(Ask(), {})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    await client.aclose()
    deltas = [e.data["text"] for e in log if e.type == "aaa.llm.delta"]
    assert out.ok and len(deltas) > 1, f"{len(deltas)} deltas, outcome={out.kind}"
    return f"{len(deltas)} delta events: {''.join(deltas)[:50]!r}"


@check("unknown model -> non-retryable error (not Transient)")
async def t_bad_model():
    c = OpenAICompatClient(cfg())
    try:
        await c.complete(
            LLMRequest(
                model="definitely-not-a-real-model-xyz",
                messages=[Message(role="user", content="hi")],
            )
        )
    except E.LLMError as e:
        assert not e.retryable, (
            f"{type(e).__name__} is marked retryable; would be retried pointlessly"
        )
        await c.aclose()
        return f"{type(e).__name__}"
    raise AssertionError("gateway accepted a nonexistent model")


@check("bad credentials -> AuthError (skipped if the gateway is unauthenticated)")
async def t_bad_auth():
    if not KEY:
        return "skip: no AAA_LIVE_API_KEY configured, gateway likely unauthenticated"
    c = OpenAICompatClient(cfg(api_key="sk-this-is-not-valid", headers={}))
    try:
        await c.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    except E.AuthError:
        return "AuthError"
    except E.LLMError as e:
        return f"got {type(e).__name__} (expected AuthError; gateway may map auth differently)"
    raise AssertionError("invalid key was accepted")


@check("unreachable endpoint -> TransientError (retryable)")
async def t_unreachable():
    c = OpenAICompatClient(cfg(base_url="http://127.0.0.1:9/v1", timeout_s=3))
    try:
        await c.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    except E.TransientError:
        return "TransientError"
    raise AssertionError("expected TransientError")


@check("embeddings")
async def t_embed():
    if not EMBED_MODEL:
        return "skip: set AAA_LIVE_EMBED_MODEL to test"
    c = OpenAICompatClient(cfg())
    r = await c.embed(["hello world", "goodbye"], model=EMBED_MODEL)
    assert len(r.vectors) == 2 and len(r.vectors[0]) > 8
    await c.aclose()
    return f"dim={len(r.vectors[0])}"


async def main() -> int:
    if not BASE or not MODEL:
        print("set AAA_LIVE_BASE_URL and AAA_LIVE_MODEL (see the docstring)", file=sys.stderr)
        return 2
    print(
        f"gateway: {BASE}{CHAT_PATH}   model: {MODEL}   auth: {'api key' if KEY else 'none'}   extra headers: {sorted(HEADERS)}\n"
    )
    for t in (
        t_complete,
        t_stream,
        t_structured,
        t_tool_call,
        t_stream_tool,
        t_e2e,
        t_plan,
        t_repair,
        t_handler_stream,
        t_bad_model,
        t_bad_auth,
        t_unreachable,
        t_embed,
    ):
        await t()
    n = {k: sum(1 for r in results if r[1] == k) for k in ("PASS", "FAIL", "SKIP")}
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
