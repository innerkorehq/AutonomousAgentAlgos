"""The OpenAI Responses API client, over a mock transport. Request and response shapes follow OpenAI's published API;
test_openai_conformance (below, needs the `openai` package) checks them against the official SDK's own types."""

import json

import httpx
import pytest

from aaa.llm import LLMConfig, build_client
from aaa.llm.errors import ContextOverflow, StateNotFound
from aaa.llm.openai_responses import OpenAIResponsesClient, input_items
from aaa.llm.types import LLMRequest, Message, ToolCall, ToolSpec

BASE = "https://api.example.test/v1"


def make(handler, **kw):
    cfg = LLMConfig(
        provider="openai_responses", model="gpt-x", base_url=BASE, api_key="sk-test", **kw
    )
    return OpenAIResponsesClient(
        cfg, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def reply(output, *, rid="resp_1", usage=None, **extra):
    return {
        "id": rid,
        "object": "response",
        "status": "completed",
        "model": "gpt-x",
        "output": output,
        "usage": usage
        or {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 60},
        },
        **extra,
    }


def message(txt):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": txt, "annotations": []}],
    }


async def test_a_plain_request_maps_to_the_responses_api():
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=reply([message("Hello!")]))

    client = make(
        handler, price_in_per_mtok=1.0, price_cached_in_per_mtok=0.1, price_out_per_mtok=2.0
    )
    r = await client.complete(
        LLMRequest(
            messages=[
                Message(role="system", content="be brief"),
                Message(role="system", content="and kind"),
                Message(role="user", content="hi"),
            ],
            tools=[
                ToolSpec(
                    name="lookup",
                    description="d",
                    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                )
            ],
            params={"max_tokens": 50, "temperature": 0.2},
            response_schema=None,
        )
    )
    body = json.loads(seen[0].content)
    assert (
        seen[0].url.path == "/v1/responses" and seen[0].headers["authorization"] == "Bearer sk-test"
    )
    assert body["instructions"] == "be brief\n\nand kind" and body["input"] == [
        {"role": "user", "content": "hi"}
    ]
    assert (
        body["model"] == "gpt-x"
        and body["store"] is True
        and body["max_output_tokens"] == 50
        and "max_tokens" not in body
    )
    assert body["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "d",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            "strict": False,
        }
    ]
    assert "previous_response_id" not in body
    assert r.message.content == "Hello!" and r.state_id == "resp_1" and r.finish_reason == "stop"
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens) == (
        100,
        20,
        60,
    )
    assert r.cost == pytest.approx(
        (40 * 1.0 + 60 * 0.1 + 20 * 2.0) / 1e6
    )  # cache hits are billed at the discount


async def test_continuing_sends_previous_response_id_and_only_new_items():
    seen: list[dict] = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=reply([message("ok")], rid="resp_2"))

    client = make(handler)
    await client.complete(
        LLMRequest(
            messages=[
                Message(role="system", content="be brief"),
                Message(role="tool", tool_call_id="call_abc", content="42"),
            ],
            continue_from="resp_1",
        )
    )
    body = seen[0]
    assert body["previous_response_id"] == "resp_1"
    assert body["instructions"] == "be brief", (
        "instructions are not carried by the provider: they are sent every time"
    )
    assert body["input"] == [
        {"type": "function_call_output", "call_id": "call_abc", "output": "42"}
    ]


async def test_tool_calls_keep_the_providers_ids_and_results_are_matched_to_them():
    out = [
        message("calling"),
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_prov_1",
            "name": "lookup",
            "arguments": '{"q": "x"}',
            "status": "completed",
        },
    ]
    client = make(lambda req: httpx.Response(200, json=reply(out)))
    r = await client.complete(LLMRequest(messages=[Message(role="user", content="go")]))
    (call,) = r.message.tool_calls
    assert (call.id, call.provider_id, call.name, call.arguments) == (
        "call_prov_1",
        "call_prov_1",
        "lookup",
        {"q": "x"},
    )
    assert r.finish_reason == "tool_calls" and r.message.content == "calling"
    # a full re-send after AAA renamed the call: results still name the PROVIDER's id
    renamed = call.model_copy(update={"id": "call_aaa_0"})
    items = input_items(
        [
            Message(role="user", content="go"),
            Message(role="assistant", tool_calls=[renamed]),
            Message(role="tool", tool_call_id="call_aaa_0", content="7"),
        ]
    )
    assert items[1] == {
        "type": "function_call",
        "call_id": "call_prov_1",
        "name": "lookup",
        "arguments": '{"q": "x"}',
    }
    assert items[2] == {"type": "function_call_output", "call_id": "call_prov_1", "output": "7"}


async def test_malformed_tool_arguments_are_reported_not_guessed():
    out = [{"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{oops"}]
    client = make(lambda req: httpx.Response(200, json=reply(out)))
    r = await client.complete(LLMRequest(messages=[Message(role="user", content="go")]))
    assert r.message.tool_calls[0].raw_arguments == "{oops"


async def test_errors_are_classified_and_a_missing_state_is_its_own_error():
    def gone(req):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "previous_response_not_found",
                    "message": "Previous response with id 'resp_1' not found.",
                }
            },
        )

    with pytest.raises(StateNotFound):
        await make(gone).complete(
            LLMRequest(messages=[Message(role="user", content="x")], continue_from="resp_1")
        )

    def long(req):
        return httpx.Response(
            400, json={"error": {"message": "This model's maximum context length is 128000 tokens"}}
        )

    with pytest.raises(ContextOverflow):
        await make(long).complete(LLMRequest(messages=[Message(role="user", content="x")]))


async def test_store_false_is_a_stateless_client():
    seen: list[dict] = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=reply([message("ok")]))

    client = make(handler)
    stateless = OpenAIResponsesClient(client.config, store=False, http_client=client._http)
    assert stateless.server_state is False
    r = await stateless.complete(LLMRequest(messages=[Message(role="user", content="x")]))
    assert seen[0]["store"] is False and r.state_id is None


async def test_forgetting_state_deletes_each_response_and_tolerates_missing_ones():
    calls: list[tuple[str, str]] = []

    def handler(req):
        calls.append((req.method, req.url.path))
        return httpx.Response(404 if req.url.path.endswith("gone") else 200, json={"deleted": True})

    n = await make(handler).forget_state(["resp_a", "resp_gone", "resp_b"])
    assert n == 2 and calls == [
        ("DELETE", "/v1/responses/resp_a"),
        ("DELETE", "/v1/responses/resp_gone"),
        ("DELETE", "/v1/responses/resp_b"),
    ]


async def test_streaming_assembles_text_tool_calls_usage_and_the_state_handle():
    from aaa.llm.streaming import collect

    events = [
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "function_call", "call_id": "call_9", "name": "lookup"},
        },
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{"q":'},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": ' "z"}'},
        {
            "type": "response.completed",
            "response": reply([{"type": "function_call"}], rid="resp_s"),
        },
    ]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    client = make(
        lambda req: httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    r = await collect(
        client.stream(LLMRequest(messages=[Message(role="user", content="go")], stream=True))
    )
    assert (
        r.message.content == "Hello" and r.state_id == "resp_s" and r.finish_reason == "tool_calls"
    )
    assert (r.message.tool_calls[0].id, r.message.tool_calls[0].arguments) == ("call_9", {"q": "z"})
    assert r.usage.cached_tokens == 60


def test_the_factory_builds_it_and_chat_completions_clients_report_cached_tokens():
    client = build_client(LLMConfig(provider="openai_responses", model="m", base_url=BASE))
    assert isinstance(client, OpenAIResponsesClient) and client.server_state is True


async def test_chat_completions_clients_report_cached_prompt_tokens_too():
    from aaa.llm.openai_compat import OpenAICompatClient

    body = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }
    cfg = LLMConfig(
        model="m",
        base_url=BASE,
        price_in_per_mtok=1.0,
        price_cached_in_per_mtok=0.25,
        price_out_per_mtok=2.0,
    )
    c = OpenAICompatClient(
        cfg,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        ),
    )
    r = await c.complete(LLMRequest(messages=[Message(role="user", content="x")]))
    assert r.usage.cached_tokens == 80 and r.cost == pytest.approx(
        (20 * 1.0 + 80 * 0.25 + 5 * 2.0) / 1e6
    )
    assert ToolCall  # (imported for the type used above in this module)
