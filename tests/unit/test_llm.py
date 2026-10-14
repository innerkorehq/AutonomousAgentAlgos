import json

import httpx
import pytest

from aaa.core import OutcomeKind
from aaa.events import Event
from aaa.llm import (
    CircuitBreakerClient,
    FakeLLMClient,
    LLMConfig,
    LLMHandler,
    LLMRequest,
    Message,
    OpenAICompatClient,
    RetryingClient,
    ToolSpec,
    build_client,
    gateways,
    register_provider,
)
from aaa.llm import errors as E
from aaa.llm.config import expand_env
from aaa.llm.fake import ReplayLLMClient, text, tool_call
from aaa.runtime import Engine

REQ = LLMRequest(messages=[Message(role="user", content="hi")])


def mock_client(handler, cfg=None, **kw):
    cfg = cfg or LLMConfig(model="m", base_url="https://gw.test/v1", **kw)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatClient(cfg, http_client=http)


def ok_body(content="yo", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "model": "m",
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


async def test_request_mapping_and_response_parsing():
    seen = {}

    def h(req: httpx.Request):
        seen["url"], seen["body"], seen["auth"] = (
            str(req.url),
            json.loads(req.content),
            req.headers["authorization"],
        )
        return httpx.Response(
            200,
            json=ok_body(
                None,
                [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a": 1}'},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "g", "arguments": "{oops"},
                    },
                ],
            ),
        )

    c = mock_client(h, api_key="sk-literal", price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    r = await c.complete(
        LLMRequest(
            messages=[Message(role="user", content="x")],
            tools=[ToolSpec(name="f", parameters={"type": "object"})],
            response_schema={"type": "object"},
            params={"temperature": 0},
        )
    )
    assert (
        seen["url"] == "https://gw.test/v1/chat/completions" and seen["auth"] == "Bearer sk-literal"
    )
    assert seen["body"]["tools"][0]["function"]["name"] == "f" and seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert r.message.tool_calls[0].arguments == {"a": 1}
    assert (
        r.message.tool_calls[1].raw_arguments == "{oops" and r.message.tool_calls[1].arguments == {}
    )
    assert r.usage.total_tokens == 15 and r.cost == pytest.approx(20 / 1_000_000)


@pytest.mark.parametrize(
    "status,body,exc",
    [
        (429, "slow", E.RateLimited),
        (503, "x", E.TransientError),
        (401, "no", E.AuthError),
        (400, "maximum context length exceeded", E.ContextOverflow),
        (400, "content_policy_violation", E.ContentFiltered),
        (404, "nope", E.FatalError),
    ],
)
async def test_error_classification(status, body, exc):
    c = mock_client(lambda r: httpx.Response(status, text=body, headers={"retry-after": "2"}))
    with pytest.raises(exc):
        await c.complete(REQ)


async def test_timeout_is_transient():
    def h(r):
        raise httpx.ConnectTimeout("t")

    with pytest.raises(E.TransientError):
        await mock_client(h).complete(REQ)


async def test_retry_then_success_and_nonretryable_not_retried():
    calls, slept = {"n": 0}, []

    def h(r):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json=ok_body())

    async def sleep(s):
        slept.append(s)

    c = RetryingClient(mock_client(h), max_attempts=3, sleep=sleep)
    assert (await c.complete(REQ)).message.content == "yo" and len(slept) == 2
    calls["n"] = 0
    bad = RetryingClient(mock_client(lambda r: httpx.Response(401)), sleep=sleep)
    with pytest.raises(E.AuthError):
        await bad.complete(REQ)


async def test_circuit_breaker_opens_and_recovers():
    t = {"now": 0.0}
    fake = FakeLLMClient([E.TransientError("x")] * 2 + [text("ok")])
    cb = CircuitBreakerClient(fake, failure_threshold=2, cooldown_s=10, clock=lambda: t["now"])
    for _ in range(2):
        with pytest.raises(E.TransientError):
            await cb.complete(REQ)
    assert cb.state == "open"
    with pytest.raises(E.CircuitOpen):
        await cb.complete(REQ)
    t["now"] = 11
    assert (await cb.complete(REQ)).message.content == "ok" and cb.state == "closed"


def test_presets_and_env_expansion(monkeypatch):
    p = gateways.portkey("gpt-4o", provider="openai")
    assert p.base_url == "https://api.portkey.ai/v1" and p.headers["x-portkey-provider"] == "openai"
    cf = gateways.cloudflare("openai/gpt-5-mini", "ACC")
    assert cf.base_url == "https://gateway.ai.cloudflare.com/v1/ACC/default/compat"
    assert gateways.vercel("openai/gpt-5-mini").base_url == "https://ai-gateway.vercel.sh/v1"
    assert gateways.kong("m", "http://k:8000/chat").chat_path == ""
    assert gateways.mlflow("ep").base_url.endswith("/gateway/mlflow/v1")
    assert gateways.bifrost("openai/gpt-4o-mini").base_url == "http://localhost:8080/v1"
    assert expand_env("Bearer ${T}", {"T": "abc"}) == "Bearer abc"
    with pytest.raises(E.LLMConfigError):
        expand_env("${NOPE_NOT_SET}", {})
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cft")
    monkeypatch.setenv("PROVIDER_API_KEY", "pk")
    seen = {}

    def h(req):
        seen.update(req.headers)
        seen["url"] = str(req.url)
        return httpx.Response(200, json=ok_body())

    import asyncio

    asyncio.run(mock_client(h, cf).complete(REQ))
    assert seen["cf-aig-authorization"] == "Bearer cft" and seen["authorization"] == "Bearer pk"


async def test_kong_posts_to_route_url_exactly():
    seen = {}

    def h(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json=ok_body())

    await mock_client(h, gateways.kong("m", "http://k:8000/chat")).complete(REQ)
    assert seen["url"] == "http://k:8000/chat"


async def test_custom_provider_registry():
    register_provider("mine", lambda cfg: FakeLLMClient([text("custom")]))
    c = build_client(LLMConfig(provider="mine", model="x"))
    assert (await c.complete(REQ)).message.content == "custom"
    with pytest.raises(E.LLMConfigError):
        build_client(LLMConfig(provider="nope", model="x"))


async def test_handler_through_engine_records_events_budget_and_replay():
    class Ask:
        name = "ask"

        def subscriptions(self):
            return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

        async def on_event(self, e: Event, ctx):
            if e.type == "aaa.run.started":
                return [e.caused("aaa.llm.requested", {"request": REQ.model_dump(mode="json")})]
            if e.type == "aaa.llm.failed":
                return [
                    e.caused(
                        "aaa.run.completed", {"kind": "failed", "reason": e.data["error"]["type"]}
                    )
                ]
            return [
                e.caused(
                    "aaa.run.completed",
                    {"kind": "success", "value": e.data["response"]["message"]["content"]},
                )
            ]

    fake = FakeLLMClient([text("hello", total_tokens=7)])
    async with Engine() as eng:
        LLMHandler(fake).attach(eng)
        h = await eng.start(Ask(), {})
        out = await h.result()
        log = await eng.store.read(h.run_id)
    assert out.value == "hello"
    assert [e.type for e in log] == [
        "aaa.run.started",
        "aaa.llm.requested",
        "aaa.llm.completed",
        "aaa.run.completed",
    ]
    assert log[2].causation_id == log[1].id and log[2].data["usage"]["total_tokens"] == 7
    async with Engine() as eng2:
        LLMHandler(ReplayLLMClient(log)).attach(eng2)
        assert (await eng2.run(Ask(), {})).value == "hello"
    async with Engine() as eng3:
        LLMHandler(FakeLLMClient([E.AuthError("bad key")])).attach(eng3)
        o = await eng3.run(Ask(), {})
        assert o.kind is OutcomeKind.FAILED and o.reason == "AuthError"
    assert tool_call("f", x=1).message.tool_calls[0].arguments == {"x": 1}


SSE = (
    'data: {"model":"m","choices":[{"delta":{"content":"Hel"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"f","arguments":"{\\"a\\""}}]}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
    "data: [DONE]\n\n"
)


async def test_sse_stream_collects_text_tool_calls_and_usage():
    from aaa.llm import collect

    seen = {}

    def h(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"})

    c = mock_client(h)
    chunks = [ch async for ch in c.stream(REQ)]
    assert "".join(ch.text for ch in chunks) == "Hello" and seen["body"]["stream"] is True
    r = await collect(c.stream(REQ))
    assert r.message.content == "Hello" and r.message.tool_calls[0].arguments == {"a": 1}
    assert r.usage.total_tokens == 7 and r.finish_reason == "tool_calls"


async def test_stream_http_error_is_classified():
    with pytest.raises(E.RateLimited):
        [x async for x in mock_client(lambda r: httpx.Response(429, text="slow")).stream(REQ)]


async def test_handler_streams_deltas_and_wrappers_forward_stream():
    class Ask:
        name = "ask"

        def subscriptions(self):
            return ["aaa.run.started", "aaa.llm.completed"]

        async def on_event(self, e, ctx):
            if e.type == "aaa.run.started":
                r = LLMRequest(messages=REQ.messages, stream=True)
                return [
                    e.caused(
                        "aaa.llm.requested", {"request": r.model_dump(mode="json"), "tag": "x"}
                    )
                ]
            return [e.caused("aaa.run.completed", {"kind": "success", "value": e.data["tag"]})]

    client = RetryingClient(FakeLLMClient([text("one two three")]))
    assert hasattr(client, "stream") and not hasattr(RetryingClient(object()), "stream")
    async with Engine() as eng:
        LLMHandler(client).attach(eng)
        h = await eng.start(Ask(), {})
        assert (await h.result()).value == "x"
        log = await eng.store.read(h.run_id)
    deltas = [e.data["text"] for e in log if e.type == "aaa.llm.delta"]
    assert "".join(deltas) == "one two three" and len(deltas) == 3
    done = next(e for e in log if e.type == "aaa.llm.completed")
    assert done.data["response"]["message"]["content"] == "one two three"
