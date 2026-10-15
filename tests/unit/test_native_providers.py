"""Native Anthropic and Gemini clients. One scenario suite runs against both through a mock server that answers in each provider's own
wire format; shapes are additionally validated against the vendors' SDK types where those SDKs are installed (they are not dependencies)."""

import json

import httpx
import pytest

from aaa.algos import ToolLoop
from aaa.core.outcome import OutcomeKind
from aaa.data.blob import InMemoryBlobStore
from aaa.llm import LLMHandler
from aaa.llm.anthropic import AnthropicClient
from aaa.llm.config import LLMConfig
from aaa.llm.errors import (
    AuthError,
    ContentFiltered,
    ContextOverflow,
    LLMConfigError,
    RateLimited,
    TransientError,
)
from aaa.llm.factory import build_client
from aaa.llm.gemini import GeminiClient
from aaa.llm.parts import AudioPart, FilePart, ImagePart
from aaa.llm.streaming import collect
from aaa.llm.types import LLMRequest, Message, ToolSpec
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
ADD = ToolSpec(
    name="add",
    description="add two numbers",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)
SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


# -- a mock server per provider ---------------------------------------------------------------
def a_reply(content, stop="end_turn", usage=None):
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage
        or {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def g_reply(parts, finish="STOP", usage=None):
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": finish, "index": 0}
        ],
        "usageMetadata": usage
        or {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        "modelVersion": "gemini-test",
    }


class Server:
    """Records requests; `script(payload)` returns a provider-shaped dict (or an httpx.Response)."""

    def __init__(self, script):
        self.script, self.requests = script, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request, body))
        out = self.script(body)
        return out if isinstance(out, httpx.Response) else httpx.Response(200, json=out)


def make(provider: str, script, **cfg):
    server = Server(script)
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    config = LLMConfig(provider=provider, model="m", api_key="secret-key", **cfg)
    client = (
        AnthropicClient(config, http_client=http)
        if provider == "anthropic"
        else GeminiClient(config, http_client=http)
    )
    return client, server


PROVIDERS = ["anthropic", "gemini"]


def text_reply(provider, s="Hello"):
    return (
        a_reply([{"type": "text", "text": s}])
        if provider == "anthropic"
        else g_reply([{"text": s}])
    )


def tool_reply(provider):
    if provider == "anthropic":
        return a_reply(
            [
                {"type": "text", "text": "adding"},
                {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1, "b": 2}},
            ],
            stop="tool_use",
        )
    return g_reply(
        [{"text": "adding"}, {"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}}]
    )


# -- shared behaviour -------------------------------------------------------------------------
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_text_reply_usage_cost_and_auth(provider):
    client, server = make(
        provider, lambda b: text_reply(provider), price_in_per_mtok=1.0, price_out_per_mtok=2.0
    )
    r = await client.complete(
        LLMRequest(
            messages=[
                Message(role="system", content="Be brief."),
                Message(role="user", content="hi"),
            ]
        )
    )
    assert r.message.content == "Hello" and r.finish_reason == "stop"
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens) == (10, 5, 15)
    assert r.cost == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6)
    req, _ = server.requests[0]
    assert "secret-key" not in str(req.url)  # keys travel in headers, never in the URL
    assert req.headers["x-api-key" if provider == "anthropic" else "x-goog-api-key"] == "secret-key"


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_tool_call_round_trip_through_the_wire_format(provider):
    def script(body):
        sent_result = (
            json.dumps(body).count("tool_result" if provider == "anthropic" else "functionResponse")
            > 0
        )
        return text_reply(provider, "3") if sent_result else tool_reply(provider)

    client, server = make(provider, script)
    first = await client.complete(
        LLMRequest(messages=[Message(role="user", content="1+2?")], tools=[ADD])
    )
    assert first.finish_reason == "tool_calls" and first.message.content == "adding"
    call = first.message.tool_calls[0]
    assert (call.name, call.arguments) == ("add", {"a": 1, "b": 2})
    history = [
        Message(role="user", content="1+2?"),
        first.message,
        Message(role="tool", tool_call_id=call.id, name="add", content="3"),
    ]
    second = await client.complete(LLMRequest(messages=history, tools=[ADD]))
    assert second.message.content == "3" and second.finish_reason == "stop"
    body = server.requests[1][1]
    if provider == "anthropic":
        assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
        use, result = body["messages"][1]["content"][-1], body["messages"][2]["content"][0]
        assert use["type"] == "tool_use" and result == {
            "type": "tool_result",
            "tool_use_id": use["id"],
            "content": "3",
        }
        assert body["tools"][0]["input_schema"]["required"] == ["a", "b"]
    else:
        assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
        assert body["contents"][1]["parts"][-1]["functionCall"] == {
            "name": "add",
            "args": {"a": 1, "b": 2},
        }
        assert body["contents"][2]["parts"][0]["functionResponse"] == {
            "name": "add",
            "response": {"output": "3"},
        }
        assert body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["required"] == [
            "a",
            "b",
        ]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_structured_output(provider):
    def script(body):
        if provider == "anthropic":
            assert (
                body["tool_choice"] == {"type": "tool", "name": "aaa_respond"}
                and body["tools"][0]["input_schema"] == SCHEMA
            )
            return a_reply(
                [{"type": "tool_use", "id": "toolu_1", "name": "aaa_respond", "input": {"n": 7}}],
                stop="tool_use",
            )
        assert (
            body["generationConfig"]["responseMimeType"] == "application/json"
            and body["generationConfig"]["responseJsonSchema"] == SCHEMA
        )
        return g_reply([{"text": '{"n": 7}'}])

    client, _ = make(provider, script)
    r = await client.complete(
        LLMRequest(messages=[Message(role="user", content="give n")], response_schema=SCHEMA)
    )
    assert (
        json.loads(r.message.content or "") == {"n": 7}
        and not r.message.tool_calls
        and r.finish_reason == "stop"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_system_prompt_and_params(provider):
    client, server = make(provider, lambda b: text_reply(provider), params={"temperature": 0.2})
    await client.complete(
        LLMRequest(
            messages=[
                Message(role="system", content="S1"),
                Message(role="system", content="S2"),
                Message(role="user", content="hi"),
            ],
            params={"max_tokens": 50, "stop": "END", "frequency_penalty_typo": 1},
        )
    )
    body = server.requests[0][1]
    if provider == "anthropic":
        assert (
            body["system"] == [{"type": "text", "text": "S1\n\nS2"}]
            and body["max_tokens"] == 50
            and body["temperature"] == 0.2
            and body["stop_sequences"] == ["END"]
        )
        assert "frequency_penalty_typo" not in body and all(
            m["role"] != "system" for m in body["messages"]
        )
    else:
        assert body["systemInstruction"] == {"parts": [{"text": "S1\n\nS2"}]}
        assert body["generationConfig"] == {
            "temperature": 0.2,
            "maxOutputTokens": 50,
            "stopSequences": ["END"],
        }


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_attachments_on_the_wire(provider):
    client, server = make(provider, lambda b: text_reply(provider))
    msg = Message(
        role="user",
        content="look",
        parts=[
            ImagePart.from_bytes(PNG, "image/png"),
            ImagePart.from_url("https://example.com/a.png", "image/png"),
            FilePart.from_bytes(b"%PDF-1.4", "application/pdf", name="d.pdf"),
        ],
    )
    await client.complete(LLMRequest(messages=[msg]))
    body = server.requests[0][1]
    if provider == "anthropic":
        blocks = body["messages"][0]["content"]
        assert (
            blocks[1]["source"]["type"] == "base64"
            and blocks[1]["source"]["media_type"] == "image/png"
        )
        assert blocks[2]["source"] == {"type": "url", "url": "https://example.com/a.png"}
        assert (
            blocks[3]["type"] == "document"
            and blocks[3]["source"]["media_type"] == "application/pdf"
            and blocks[3]["title"] == "d.pdf"
        )
    else:
        parts = body["contents"][0]["parts"]
        assert (
            parts[1]["inlineData"]["mimeType"] == "image/png"
            and parts[2]["fileData"]
            == {"mimeType": "image/png", "fileUri": "https://example.com/a.png"}
            and "inlineData" in parts[3]
        )


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_errors_map_to_aaa_error_classes(provider):
    def err(status, kind_a, kind_g, msg="x", headers=None):
        body = (
            {"type": "error", "error": {"type": kind_a, "message": msg}}
            if provider == "anthropic"
            else {"error": {"code": status, "message": msg, "status": kind_g}}
        )
        return httpx.Response(status, json=body, headers=headers or {})

    cases = [
        (
            err(429, "rate_limit_error", "RESOURCE_EXHAUSTED", headers={"retry-after": "7"}),
            RateLimited,
        ),
        (err(500, "api_error", "INTERNAL"), TransientError),
        (
            err(529 if provider == "anthropic" else 503, "overloaded_error", "UNAVAILABLE"),
            TransientError,
        ),
        (err(401, "authentication_error", "UNAUTHENTICATED"), AuthError),
        (err(403, "permission_error", "PERMISSION_DENIED"), AuthError),
        (
            err(
                400,
                "invalid_request_error",
                "INVALID_ARGUMENT",
                msg="prompt is too long: 250000 tokens > 200000 maximum"
                if provider == "anthropic"
                else "The input token count exceeds the maximum number of tokens allowed",
            ),
            ContextOverflow,
        ),
    ]
    for response, expected in cases:
        client, _ = make(provider, lambda b, r=response: r)
        with pytest.raises(expected) as ei:
            await client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
        if expected is RateLimited:
            assert ei.value.retry_after == 7.0


async def test_provider_specific_refusals():
    client, _ = make("gemini", lambda b: g_reply([], finish="SAFETY"))
    with pytest.raises(ContentFiltered):
        await client.complete(LLMRequest(messages=[Message(role="user", content="x")]))
    client, _ = make("gemini", lambda b: {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})
    with pytest.raises(ContentFiltered, match="blocked"):
        await client.complete(LLMRequest(messages=[Message(role="user", content="x")]))
    client, _ = make("anthropic", lambda b: a_reply([], stop="refusal"))
    with pytest.raises(ContentFiltered):
        await client.complete(LLMRequest(messages=[Message(role="user", content="x")]))


# -- Anthropic specifics ----------------------------------------------------------------------
async def test_anthropic_merges_tool_results_and_puts_them_before_user_text():
    client, server = make("anthropic", lambda b: text_reply("anthropic"))
    from aaa.llm.types import ToolCall

    calls = [
        ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),
        ToolCall(id="c2", name="add", arguments={"a": 2, "b": 2}),
    ]
    msgs = [
        Message(role="user", content="two sums"),
        Message(role="assistant", tool_calls=calls),
        Message(role="tool", tool_call_id="c1", content="2"),
        Message(role="tool", tool_call_id="c2", content="4"),
        Message(role="user", content="thanks, now explain"),
    ]
    await client.complete(LLMRequest(messages=msgs, tools=[ADD]))
    turns = server.requests[0][1]["messages"]
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert [b["type"] for b in turns[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert turns[1]["content"][0]["id"] == "c1"


async def test_anthropic_requires_max_tokens_and_supports_prompt_caching_and_usage():
    client, server = make(
        "anthropic",
        lambda b: a_reply(
            [{"type": "text", "text": "ok"}],
            usage={
                "input_tokens": 3,
                "output_tokens": 2,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 50,
            },
        ),
    )
    client.cache = True
    r = await client.complete(
        LLMRequest(
            messages=[
                Message(role="system", content="long system"),
                Message(role="user", content="q"),
            ],
            tools=[ADD],
        )
    )
    body = server.requests[0][1]
    assert body["max_tokens"] == 4096 and body["system"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"} and body["messages"][-1][
        "content"
    ][-1]["cache_control"] == {"type": "ephemeral"}
    assert (r.usage.prompt_tokens, r.usage.cached_tokens, r.usage.completion_tokens) == (
        153,
        50,
        2,
    )  # cache reads and writes count as prompt tokens
    assert server.requests[0][0].headers["anthropic-version"] == "2023-06-01"


async def test_anthropic_has_no_embeddings_and_no_audio():
    client, _ = make("anthropic", lambda b: text_reply("anthropic"))
    assert client.modalities == {"text", "image", "file"}
    with pytest.raises(LLMConfigError, match="embeddings"):
        await client.embed(["x"])


# -- Gemini specifics -------------------------------------------------------------------------
async def test_gemini_orders_function_responses_like_the_calls_and_echoes_call_ids():
    client, server = make("gemini", lambda b: text_reply("gemini"))
    from aaa.llm.types import ToolCall

    calls = [
        ToolCall(id="c1", name="add", arguments={"a": 1}, provider_id="gid-1"),
        ToolCall(id="c2", name="mul", arguments={"a": 2}),
    ]
    msgs = [
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=calls),
        Message(role="tool", tool_call_id="c2", name="mul", content="B"),
        Message(role="tool", tool_call_id="c1", name="add", content="A"),
    ]
    await client.complete(LLMRequest(messages=msgs))
    contents = server.requests[0][1]["contents"]
    assert (
        contents[1]["parts"][0]["functionCall"]["id"] == "gid-1"
        and "id" not in contents[1]["parts"][1]["functionCall"]
    )
    resp = [p["functionResponse"] for p in contents[2]["parts"]]
    assert [r["name"] for r in resp] == ["add", "mul"] and resp[0][
        "id"
    ] == "gid-1"  # results follow call order, whatever order they arrived in


async def test_gemini_ignores_thought_parts_and_counts_thinking_as_output():
    client, _ = make(
        "gemini",
        lambda b: g_reply(
            [{"text": "(thinking...)", "thought": True}, {"text": "answer"}],
            usage={
                "promptTokenCount": 4,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 10,
                "totalTokenCount": 17,
                "cachedContentTokenCount": 2,
            },
        ),
    )
    r = await client.complete(LLMRequest(messages=[Message(role="user", content="q")]))
    assert (
        r.message.content == "answer"
        and r.usage.completion_tokens == 13
        and r.usage.cached_tokens == 2
        and r.usage.total_tokens == 17
    )


async def test_gemini_embeddings():
    def script(body):
        assert body["requests"][0]["model"] == "models/text-embedding-004" and body["requests"][1][
            "content"
        ] == {"parts": [{"text": "b"}]}
        return {"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]}

    client, server = make("gemini", script)
    r = await client.embed(["a", "b"], "models/text-embedding-004")
    assert r.vectors == [[0.1, 0.2], [0.3, 0.4]] and str(server.requests[0][0].url).endswith(
        "/models/text-embedding-004:batchEmbedContents"
    )


async def test_gemini_tool_choice_mapping():
    client, server = make("gemini", lambda b: text_reply("gemini"))
    await client.complete(
        LLMRequest(
            messages=[Message(role="user", content="x")],
            tools=[ADD],
            params={"tool_choice": "required"},
        )
    )
    assert server.requests[0][1]["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}


# -- streaming --------------------------------------------------------------------------------
def sse(*events):
    out = ""
    for name, data in events:
        out += (f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n"
    return httpx.Response(200, content=out.encode(), headers={"content-type": "text/event-stream"})


A_STREAM = (
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hel"},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "lo"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_9",
                "name": "add",
                "input": {},
            },
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"a": 1,'},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ' "b": 2}'},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 20},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_streaming_text_and_tool_calls_assemble_into_the_same_response(provider):
    if provider == "anthropic":
        stream = sse(*A_STREAM)
    else:
        stream = sse(
            (
                None,
                g_reply([{"text": "Hel"}], finish=None)
                | {
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": "Hel"}]}, "index": 0}
                    ]
                },
            ),
            (
                None,
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": "lo"},
                                    {"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}},
                                ],
                            },
                            "finishReason": "STOP",
                            "index": 0,
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 30,
                    },
                },
            ),
        )
    client, server = make(provider, lambda b: stream)
    r = await collect(
        client.stream(LLMRequest(messages=[Message(role="user", content="1+2?")], tools=[ADD]))
    )
    assert r.message.content == "Hello" and r.finish_reason == "tool_calls"
    assert [(c.name, c.arguments) for c in r.message.tool_calls] == [("add", {"a": 1, "b": 2})]
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (10, 20)
    req, body = server.requests[0]
    assert (
        (body.get("stream") is True)
        if provider == "anthropic"
        else str(req.url).endswith(":streamGenerateContent?alt=sse")
    )


async def test_a_stream_error_event_is_raised():
    client, _ = make(
        "anthropic",
        lambda b: sse(
            ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
        ),
    )
    with pytest.raises(TransientError, match="overloaded"):
        await collect(client.stream(LLMRequest(messages=[Message(role="user", content="x")])))


# -- the whole agent loop, through the engine -------------------------------------------------
@tool
def add(a: int, b: int) -> int:
    """add"""
    return a + b


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_tool_loop_runs_end_to_end_on_each_provider(provider):
    seen_results = []

    def script(body):
        raw = json.dumps(body)
        marker = "tool_result" if provider == "anthropic" else "functionResponse"
        if marker in raw:
            seen_results.append(raw)
            return text_reply(provider, "The answer is 3")
        return tool_reply(provider)

    client, _ = make(provider, script)
    eng = Engine(blobs=InMemoryBlobStore())
    LLMHandler(client).attach(eng)
    reg = ToolRegistry([add])
    ToolHandler(reg).attach(eng)
    async with eng:
        out = await eng.run(
            ToolLoop(reg),
            {
                "task": "what is 1+2?",
                "attachments": [
                    ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n" + b"1" * 6000, "image/png")
                ],
            },
        )
    assert out.ok and "3" in out.value["text"] and seen_results


async def test_anthropic_rejects_audio_before_any_request():
    client, server = make("anthropic", lambda b: text_reply("anthropic"))
    eng = Engine(blobs=InMemoryBlobStore())
    LLMHandler(client).attach(eng)
    reg = ToolRegistry([])
    ToolHandler(reg).attach(eng)
    async with eng:
        out = await eng.run(
            ToolLoop(reg),
            {
                "task": "transcribe",
                "attachments": [AudioPart.from_bytes(b"RIFF" + b"0" * 6000, "audio/wav")],
            },
        )
    assert (
        out.kind is OutcomeKind.FAILED
        and "UnsupportedModality" in (out.reason or "")
        and server.requests == []
    )


# -- configuration ----------------------------------------------------------------------------
def test_providers_are_registered_and_configured_declaratively():
    a = build_client(LLMConfig(provider="anthropic", model="claude-x", api_key="k"))
    g = build_client(LLMConfig(provider="gemini", model="gemini-x", api_key="k"))
    assert isinstance(a, AnthropicClient) and a.base_url == "https://api.anthropic.com"
    assert (
        isinstance(g, GeminiClient)
        and g.base_url == "https://generativelanguage.googleapis.com/v1beta"
    )
    custom = build_client(
        LLMConfig(
            provider="gemini",
            model="m",
            base_url="https://proxy.internal/v1beta/",
            modalities=["image"],
        )
    )
    assert custom.base_url == "https://proxy.internal/v1beta" and custom.modalities == {"image"}


# -- shapes against the vendors' own SDK types (skipped where the SDK is not installed) ---------------------
def test_anthropic_requests_and_replies_match_the_sdk_types():
    pytest.importorskip("anthropic")
    from anthropic.types import Message as SdkMessage
    from anthropic.types import MessageParam, TextBlockParam, ToolParam
    from pydantic import TypeAdapter

    from aaa.llm.anthropic import to_anthropic
    from aaa.llm.types import ToolCall

    msgs = [
        Message(role="system", content="S"),
        Message(
            role="user",
            content="look",
            parts=[
                ImagePart.from_bytes(PNG, "image/png"),
                ImagePart.from_url("https://example.com/a.png", "image/png"),
                FilePart.from_bytes(b"%PDF-1.4", "application/pdf", name="d.pdf"),
            ],
        ),
        Message(
            role="assistant",
            content="ok",
            tool_calls=[ToolCall(id="toolu_1", name="add", arguments={"a": 1, "b": 2})],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="3"),
    ]
    system, turns = to_anthropic(msgs, cache=True)
    for t in turns:
        TypeAdapter(MessageParam).validate_python(t)
    for s in system or []:
        TypeAdapter(TextBlockParam).validate_python(s)
    TypeAdapter(ToolParam).validate_python(
        {"name": "add", "description": "d", "input_schema": ADD.parameters}
    )
    for reply in (
        tool_reply("anthropic"),
        text_reply("anthropic"),
        a_reply(
            [{"type": "tool_use", "id": "toolu_1", "name": "aaa_respond", "input": {"n": 7}}],
            stop="tool_use",
        ),
    ):
        SdkMessage.model_validate(
            reply
        )  # our fake server speaks the real format, so the parser is tested against reality


def test_gemini_requests_and_replies_match_the_sdk_types():
    pytest.importorskip("google.genai")
    from google.genai import types

    from aaa.llm.gemini import to_gemini
    from aaa.llm.types import ToolCall

    msgs = [
        Message(role="system", content="S"),
        Message(
            role="user",
            content="look",
            parts=[
                ImagePart.from_bytes(PNG, "image/png"),
                ImagePart.from_url("https://example.com/a.png", "image/png"),
            ],
        ),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1})]),
        Message(role="tool", tool_call_id="c1", name="add", content="3"),
    ]
    system, contents = to_gemini(msgs)
    types.Content.model_validate(system)
    for c in contents:
        types.Content.model_validate(c)
    client, _ = make("gemini", lambda b: text_reply("gemini"))
    payload = client._payload(
        LLMRequest(
            messages=msgs,
            tools=[ADD],
            response_schema=SCHEMA,
            params={"temperature": 0.1, "max_tokens": 9, "stop": "x"},
        )
    )
    types.GenerateContentConfig.model_validate(payload["generationConfig"])
    types.Tool.model_validate(payload["tools"][0])
    for reply in (tool_reply("gemini"), text_reply("gemini")):
        types.GenerateContentResponse.model_validate(reply)


def test_anthropic_stream_events_match_the_sdk_types():
    pytest.importorskip("anthropic")
    from anthropic.types import RawMessageStreamEvent
    from pydantic import TypeAdapter

    ta = TypeAdapter(RawMessageStreamEvent)
    for _name, event in A_STREAM:
        ta.validate_python(event)
